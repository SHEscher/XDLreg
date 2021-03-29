# %% Import
from meta_functions import *

import string
from skimage import draw
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # due to BigSur issue with "PyQt5" / "MacOSX" backend
# print(matplotlib.get_backend())
import concurrent.futures
import keras

from LRP import apply_colormap  # , analyze_model
from apply_heatmap import create_cmap, gregoire_black_firered
from train_kerasMRInet import crop_model_name


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# # Set global params
max_age = 80
min_age = 20  # or e.g. 4 for developemental factors (here: size of head)

# Set Paths
p2fileroot = "/data/pt_02238/DeepAge/" if check_system() == "MPI" else "../../../"
p2simulation = os.path.join(p2fileroot, "Results/Simulation/")
if not os.path.exists(p2simulation):
    os.mkdir(p2simulation)


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# # 2D data

def make_pumpkin(age, img_size=(98, 98)):
    """Create elliptic shape of random (head) size. Thickness grows proportionally. Add noise."""

    # Size of pumpkin with some random variance
    p_size = (40 + 2*np.clip(age, 0, 20),  # i.e. for age >= 20, the standard brain size is (80, 65) + v
              25 + 2*np.clip(age, 0, 20))
    p_size += np.random.normal(loc=0, scale=p_size[0]/20, size=2)  # v

    # Draw outer ellipse
    rr, cc = draw.ellipse(r=img_size[0]//2, c=img_size[1]//2,
                          r_radius=p_size[0]/2, c_radius=p_size[1]/2,
                          rotation=0)

    # Draw inner ellipse
    rr_inner, cc_inner = draw.ellipse(r=img_size[0]//2, c=img_size[1]//2,
                                      r_radius=p_size[0]/5, c_radius=p_size[1]/5, rotation=0)

    # Define grid/image (as np.arary)
    pumpkin = np.zeros((98, 98), dtype=np.float32)

    # Create elliptic shape with hole
    pumpkin[rr, cc] = .8  # set outer ellipse to .8 (max will be 1)
    pumpkin[rr_inner, cc_inner] = 0  # create hole with inner ellipse

    # Add noise
    noise = 1 - np.random.randn(*pumpkin.shape) * 0.05
    pumpkin = np.multiply(pumpkin, noise)

    return pumpkin


def random_name():
    # PH + 3 Chars + 9 digit number
    rand_name = "PH" + "".join(np.random.choice(a=list(string.ascii_letters), size=3, replace=True))
    rand_name += str(np.random.randint(0, 10 ** 8)).zfill(9)  # with 1 leading zero
    return rand_name


class PumpkinHead:
    def __init__(self, age, name=None):
        self.age = age
        self.name = random_name() if name is None else name
        self.pumpkin_brain = make_pumpkin(age=age)
        self.n_lesions = None
        self.lesion_coords = []
        self.n_atrophies = None
        self.atrophy_coords = []
        self.grow()

    def grow(self):
        """Run several ageing processes on self.pumpkin_brain ad function of self.age"""
        self.add_lesions()
        self.add_atrophies()

    def add_atrophies(self, **kwargs):
        """
        Atrophies are probabilistically applied to surface area including inner surfaces
        Reduce image intensity in a certain range up to zero (i.e. maximal reduction)
        """

        max_atrophies = max_age * kwargs.pop("max_atrophies", 5)  # keep max_age here to clip below
        expected_n_atrophies = max_atrophies * self.age/max_age

        # Compute normal (int) distrubtion around expected value
        distr = np.round(np.random.normal(loc=expected_n_atrophies,
                                          scale=kwargs.pop("scale", 1.5),
                                          size=2001))

        distr = distr[(0 <= distr) & (distr <= max_atrophies)]

        if len(distr) == 0:
            # Values are out of range: set n_lesions to the respective range-extreme of expected value
            self.n_lesions = 0 if expected_n_atrophies <= 0 + 5 else max_atrophies  # 0 + small margin

        else:

            all_probs = np.zeros(max_atrophies + 1)
            all_probs[np.unique(distr).astype(int)] = np.unique(distr, return_counts=True)[1]
            all_probs = all_probs / np.sum(all_probs)

            self.n_atrophies = np.random.choice(a=range(max_atrophies + 1), size=1, p=all_probs).item()

        # Add lesions
        ctn_atrophies = 0
        while ctn_atrophies < self.n_atrophies:
            # Get area of pumpkin head
            non_zeros_idx = np.nonzero(self.pumpkin_brain)

            # Choose random location within pumpkin
            idx = np.random.randint(low=0, high=len(non_zeros_idx[0]), size=1).item()

            xi, yi = non_zeros_idx[0][idx], non_zeros_idx[1][idx]

            # Look at surrounding of coordinate: if at boarders of pumpkin add an atrophy
            if self.pumpkin_brain[xi - 1: xi + 2, yi - 1: yi + 2].min() == 0:

                # Apply atrophy with probability: The more zeros around, the more likely the atrophy
                n_values = len(self.pumpkin_brain[xi - 1: xi + 2, yi - 1: yi + 2].flatten())
                n_zeros = n_values - np.count_nonzero(self.pumpkin_brain[xi - 1: xi + 2, yi - 1: yi + 2])
                prob_atrophy = n_zeros / (n_values-1)

                if np.random.binomial(n=1, p=prob_atrophy):
                    self.pumpkin_brain[xi, yi] = 0

                    self.atrophy_coords.append((xi, yi))  # add location of athropy to list

                    ctn_atrophies += 1

    def add_lesions(self, **kwargs):
        """
        Probabilistically add lesions within the self.pumpkin_brain of a certain size
        Increase image intensity clearly in a certain range.
        """

        max_lesions = 40
        expected_n_lesions = self.age - max_lesions

        # Compute normal (int) distrubtion around expected N-lesions
        distr = np.round(np.random.normal(loc=expected_n_lesions,
                                          scale=kwargs.pop("scale", 1),
                                          size=2001))
        distr = distr[(0 <= distr) & (distr <= max_lesions)]  # throw values out that exceed range
        # we expect for age 80: 40 lesions; for age 70: 30 lesions, and for age <= 40: 0 lesions

        if len(distr) == 0:
            # Values are out of range: set n_lesions to the respective range-extreme of expected value
            self.n_lesions = 0 if expected_n_lesions <= 0+5 else max_lesions  # 0 + small margin
        else:

            all_probs = np.zeros(max_lesions+1)
            all_probs[np.unique(distr).astype(int)] = np.unique(distr, return_counts=True)[1]
            all_probs = all_probs/np.sum(all_probs)

            self.n_lesions = np.random.choice(a=range(max_lesions+1), size=1, p=all_probs).item()

        # Get area of pumpkin head
        non_zeros_idx = np.nonzero(self.pumpkin_brain)

        # Add lesions
        ctn_lesions = 0
        while ctn_lesions < self.n_lesions:

            idx = np.random.randint(low=0, high=len(non_zeros_idx[0]), size=1).item()

            xi, yi = non_zeros_idx[0][idx], non_zeros_idx[1][idx]

            # Look at surrounding of coordinate: if not at boarders of pumpkin add a lesion
            if self.pumpkin_brain[xi-2: xi+3, yi-2: yi+3].min() > 0:

                lesion = np.clip(a=self.pumpkin_brain[xi - 1: xi + 2, yi - 1: yi + 2] + (
                        .2 + np.random.normal(scale=.025)),
                                 a_min=0, a_max=1)

                self.pumpkin_brain[xi - 1: xi + 2, yi - 1: yi + 2] = lesion

                self.lesion_coords.append((xi, yi))

                ctn_lesions += 1

    def exhibition(self, **kwargs):
        plt.figure(num=f"{self.name} | age={self.age}")
        cmap = kwargs.pop("cmap", "gist_heat")
        plt.imshow(self.pumpkin_brain, cmap=cmap, **kwargs)
        plt.show()

# # TODO 3D data
# For instance, use MNI template, and binarize it then apply some 'age'-related changes (e.g. lesions)"""


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# # Create dataset

class PumpkinSet:

    def __init__(self, n_samples, uniform=True, age_bias=None, skew_factor=.8,
                 name=None, save=True):
        self._n_samples = n_samples
        self._age_distribution = None
        self._is_uniform = uniform
        self._draw_sample_distribution(uniform=uniform, age_bias=age_bias, skew_factor=skew_factor)
        self._data = [None] * n_samples
        self._generate_data()

        if name is None:
            self.name = datetime.now().strftime('%Y-%m-%d_%H-%M_') + f"N-{n_samples}_" \
                                                                     f"{'' if uniform else 'non-'}uniform"
        else:
            self.name = name

        if save:
            self.save()

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def is_uniform(self):
        return self._is_uniform

    @property
    def age_distribution(self):
        return self._age_distribution

    @property
    def data(self):
        return self._data

    def _draw_sample_distribution(self, uniform: bool, age_bias=None, skew_factor=.8):
        if uniform:
            self._age_distribution = np.random.choice(a=np.arange(min_age,
                                                                  max_age + 1),
                                                      size=self.n_samples, replace=True)
            # round(np.random.uniform(min_age, max_age, self.n_samples)).astype(int)  # undesired boarders

        else:
            assert age_bias is not None, "age bias must be given for non-uniform age distribution"

            # Crate uniform and non-uniform part & add them
            n_uni = int(skew_factor * self.n_samples)
            uni_part = np.round(np.random.uniform(min_age, max_age, n_uni)).astype(int)
            nonuni_part = np.round(np.random.normal(loc=age_bias, scale=4, size=self.n_samples - n_uni))

            self._age_distribution = np.append(uni_part, nonuni_part)

            # Replace values which exceed range
            idx_offrange = np.where(
                (self._age_distribution < min_age) | (self._age_distribution > max_age))

            self._age_distribution[idx_offrange] = np.random.choice(a=np.arange(min_age, max_age + 1),
                                                                    size=len(idx_offrange[0]),
                                                                    replace=True)

            # Shuffle order
            np.random.shuffle(self._age_distribution)

    def _generate_data(self):

        try:
            cprint(f"Start creating the pumpkin dataset of {self.name} ...", 'b')
            start_time = datetime.now()
            # Worker n_CPU * 2 works best
            with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()*2) as executor:
                heads = executor.map(PumpkinHead, self.age_distribution)
            self._data = list(heads)
            cprint(f"Created {self.n_samples} pumpkins in "
                   f"{chop_microseconds(datetime.now() - start_time)} [hh:min:sec].", 'b')

        except Exception:
            start_time = datetime.now()
            for i, age in enumerate(self.age_distribution):
                self._data[i] = PumpkinHead(age=age)
                loop_timer(start_time=start_time, loop_length=self.n_samples, loop_idx=i,
                           loop_name="Create Pumpkin Dataset")

    def save(self):
        save_obj(obj=self, name=self.name, folder=p2simulation)

    def data2numpy(self, for_keras=True):
        # len(self.data) == self.n_samples
        ydata = np.array([self.data[i].age for i in range(self.n_samples)])

        img_shape = self.data[0].pumpkin_brain.shape
        xdata = np.empty(shape=(self.n_samples, *img_shape))

        for i in range(self.n_samples):
            xdata[i] = self.data[i].pumpkin_brain

        if for_keras:
            xdata = xdata[..., np.newaxis]

        print("xdata.shape:", xdata.shape)  # TEST
        print("ydata.shape:", ydata.shape)  # TEST

        return xdata, ydata


def load_pumpkin_set(name, folder=p2simulation):
    return load_obj(name=name, folder=folder)


def get_pumpkin_set(n_samples=2000, uniform=True, age_bias=None):

    for file in os.listdir(p2simulation):
        if f"N-{n_samples}" in file and f"{'_' if uniform else 'non-'}uniform" in file:
            cprint(f"Found & load following file: {file} ...", 'b')
            return load_pumpkin_set(name=file)

    else:
        cprint(f"No dataset found. Start creating it ...", 'b')
        return PumpkinSet(n_samples=n_samples, uniform=uniform, age_bias=age_bias, save=True)

# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# # ConvNet


def create_simulation_model(name="PumpkinNet", target_bias=None, input_shape=(98, 98), class_task=False,
                            leaky_relu=True, batch_norm=False):
    """
    This is a 2D adaptatiton of the model created in MRInet.py
    """

    if target_bias is not None:
        cprint(f"\nGiven target bias is {target_bias:.3f}\n", "y")

    if leaky_relu and not batch_norm:
        actfct = None
    else:
        actfct = "relu"

    kmodel = keras.Sequential(name=name)  # OR: Sequential([keras.layer.Conv3d(....), layer...])

    # 3D-Conv
    if batch_norm:
        kmodel.add(keras.layers.BatchNormalization(input_shape=input_shape + (1,)))
        kmodel.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="SAME",
                                       activation=actfct))
    else:
        kmodel.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="SAME",
                                       activation=actfct, input_shape=input_shape + (1,)))
        # auto-add batch:None, last: channels
    if leaky_relu:
        kmodel.add(keras.layers.LeakyReLU(alpha=.2))  # lrelu
    kmodel.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="SAME"))

    if batch_norm:
        kmodel.add(keras.layers.BatchNormalization())
    kmodel.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="SAME", activation=actfct))
    if leaky_relu:
        kmodel.add(keras.layers.LeakyReLU(alpha=.2))
    kmodel.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="SAME"))

    if batch_norm:
        kmodel.add(keras.layers.BatchNormalization())
    kmodel.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="SAME", activation=actfct))
    if leaky_relu:
        kmodel.add(keras.layers.LeakyReLU(alpha=.2))
    kmodel.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="SAME"))

    if batch_norm:
        kmodel.add(keras.layers.BatchNormalization())
    kmodel.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="SAME", activation=actfct))
    if leaky_relu:
        kmodel.add(keras.layers.LeakyReLU(alpha=.2))
    kmodel.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="SAME"))

    # 3D-Conv (1x1x1)
    if batch_norm:
        kmodel.add(keras.layers.BatchNormalization())
    kmodel.add(keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding="SAME", activation=actfct))
    if leaky_relu:
        kmodel.add(keras.layers.LeakyReLU(alpha=.2))
    kmodel.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="SAME"))

    if batch_norm:
        kmodel.add(keras.layers.BatchNormalization())

    # FC
    kmodel.add(keras.layers.Flatten())
    kmodel.add(keras.layers.Dropout(rate=.5))
    kmodel.add(keras.layers.Dense(units=64, activation=actfct))
    if leaky_relu:
        kmodel.add(keras.layers.LeakyReLU(alpha=.2))

    # Output
    if not class_task:
        kmodel.add(keras.layers.Dense(
            units=1, activation='linear',
            # add target bias == 57.317 (for age), or others
            use_bias=True,
            bias_initializer="zeros" if target_bias is None else keras.initializers.Constant(
                value=target_bias)))

    else:
        kmodel.add(keras.layers.Dense(units=2,
                                      activation='softmax',  # in binary case. also: 'sigmoid'
                                      use_bias=False))  # default: True

    # Compile
    kmodel.compile(optimizer=keras.optimizers.Adam(5e-4),  # ="adam",
                   loss="mse",
                   metrics=["accuracy"] if class_task else ["mae"])

    # Summary
    kmodel.summary()

    return kmodel


def split_simulation_data(xdata, ydata, return_idx=False, only_test=False):

    dsize = len(ydata)  # n samples

    # Get split indices
    idx_train = (0, int(.8 * dsize))
    idx_val = (int(.8 * dsize), int(.9 * dsize))
    idx_test = (int(.9 * dsize), dsize)

    if return_idx:
        return idx_test if only_test else (idx_train, idx_val, idx_test)

    # Split data
    x_train = xdata[:idx_train[1]]
    x_val = xdata[idx_val[0]:idx_val[1]]
    x_test = xdata[idx_test[0]:]
    y_train = ydata[:idx_train[1]]
    y_val = ydata[idx_val[0]:idx_val[1]]
    y_test = ydata[idx_test[0]:]

    if only_test:
        return x_test, y_test
    else:
        return x_train, x_val, x_test, y_train, y_val, y_test


def train_simulation_model(pumpkin_set, leaky_relu=False, epochs=80, batch_size=4):

    # Prep data for model
    xdata, ydata = pumpkin_set.data2numpy(for_keras=True)
    x_train, x_val, x_test, y_train, y_val, y_test = split_simulation_data(xdata=xdata, ydata=ydata,
                                                                           only_test=False)

    # Create model
    _model_name = f"PumpkinNet_{'leaky' if leaky_relu else ''}ReLU_{pumpkin_set.name.split('_')[-1]}"
    model = create_simulation_model(name=_model_name,
                                    target_bias=np.mean(ydata),
                                    leaky_relu=leaky_relu)

    # Create folders
    if not os.path.exists(os.path.join(p2simulation, model.name)):
        os.mkdir(os.path.join(p2simulation, model.name))

    # # Save model progress (callbacks)
    # See also: https://www.tensorflow.org/tutorials/keras/save_and_load
    callbacks = [keras.callbacks.ModelCheckpoint(
        filepath=f"{p2simulation}{model.name}" + "_{epoch}.h5",
        save_best_only=True,
        save_weights_only=False,
        period=10,
        monitor="val_loss",
        verbose=1),
        keras.callbacks.TensorBoard(log_dir=f"{p2simulation}{model.name}/")]
    # , keras.callbacks.EarlyStopping()]

    # # Train the model
    cprint('Fit model on training data ...', 'b')

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(x_val, y_val))

    # Save final model (weights+architecture)
    model.save(filepath=f"{p2simulation}{model.name}_final.h5")  # HDF5 file

    # Report training metrics
    # print('\nhistory dict:', history.history)
    np.save(file=f"{p2simulation}{model.name}_history", arr=history.history)

    # # Evaluate the model on the test data
    cprint(f'\nEvaluate {model.name} on test data ...', 'b')
    performs = model.evaluate(x_test, y_test, batch_size=1)  # , verbose=2)
    cprint(f'test loss, test performance: {performs}', 'y')

    return model.name


def plot_simulation_heatmaps(model_name, n_subjects=20, subset="test",
                             analyzer_type="lrp.sequential_preset_a", pointers=True,
                             cbar=False, true_scale=False):

    # Get model
    _model = keras.models.load_model(os.path.join(p2simulation, model_name + "_final.h5"))

    # Get relevance maps
    rel_obj = create_relevance_dict(model_name=model_name, subset=subset, analyzer_type=analyzer_type)

    # Prep data
    pdata = get_pumpkin_set(n_samples=2000, uniform="non-uni" not in model_name)
    _x, _y = pdata.data2numpy(for_keras=True)
    if subset == "test":
        xdata, ydata = split_simulation_data(xdata=_x, ydata=_y, only_test=True)
        didx = split_simulation_data(xdata=_x, ydata=_y, return_idx=True, only_test=True)
    else:
        # xdata, ydata = ...
        raise NotImplementedError("Not implemented yet for other subsets than 'test'")

    for sub in range(n_subjects):
        # sub = 0
        img = xdata[sub].copy()
        img = img[np.newaxis, ...]
        a = rel_obj[sub]
        # plt.imshow(a)

        col_a = apply_colormap(R=a, inputimage=img.squeeze(), cmapname='black-firered',
                               cintensifier=5., gamma=.2, true_scale=true_scale)
        sub_y = ydata[sub]
        sub_yt = _model.predict(img).item()

        fig = plt.figure(num=f"S{sub}, age={int(sub_y)}, pred={sub_yt:.2f}")
        ax = fig.add_subplot(1, 1, 1)
        aximg = plt.imshow(col_a[0], cmap=create_cmap(gregoire_black_firered))

        if cbar:
            cbar_range = (-1, 1) if not true_scale else (-col_a[2], col_a[2])
            caxbar = fig.colorbar(aximg, ax=ax, fraction=0.048, pad=0.04)  # shrink=0.8, aspect=50)
            caxbar.set_ticks(np.linspace(0, 1, 7), True)
            caxbar.ax.set_yticklabels(labels=[f"{tick:.2g}" for tick in np.linspace(
                cbar_range[0], cbar_range[1], len(caxbar.get_ticks()))])

        plt.tight_layout()

        for fm in ["png", "pdf"]:
            plt.savefig(os.path.join(p2simulation, _model.name,
                                     f"LRP_S{sub}_age-{sub_y}_pred-{sub_yt:.1f}.{fm}"))

        if pointers:
            phead = pdata.data[didx[0]+sub]
            # phead.exhibition()
            # cntr = np.array(col_a[0].shape[:-1]) // 2  # center of image

            # Mark atrophies
            for coord in phead.atrophy_coords:
                plt.plot(coord[1], coord[0], "s", color="#D3F5D4",  # "lightgreen"
                         ms=2, alpha=.9)  # ms=4: full-pixel

            # Arrows to lesions
            for coord in phead.lesion_coords:
                # Shadow
                plt.annotate(s='', xy=coord[::-1],
                             # xytext=(coord[::-1] + cntr)//2,  # arrows come from center
                             xytext=np.array(coord[::-1]) + [-4.6, 5.4],
                             arrowprops=dict(arrowstyle='simple', color="black",
                                             alpha=.5))

                # Arrow
                plt.annotate(s='', xy=coord[::-1],
                             # xytext=(coord[::-1] + cntr)//2,  # arrows come from center
                             xytext=np.array(coord[::-1]) + [-5, 5],
                             arrowprops=dict(arrowstyle='simple', color="#E3E7E3",  # "lightgreen"
                                             alpha=.9))

            plt.tight_layout()

            for fm in ["png", "pdf"]:
                plt.savefig(os.path.join(p2simulation, _model.name,
                                         f"LRP_S{sub}_age-{sub_y}_pred-{sub_yt:.1f}_pointer.{fm}"))
        plt.close()


def create_relevance_dict(model_name, subset="test", analyzer_type="lrp.sequential_preset_a", save=True):

    try:
        rel_obj = load_obj(name=model_name + f"_relevance-maps_{subset}-set", folder=p2simulation)
    except FileNotFoundError:

        import innvestigate

        _model = keras.models.load_model(os.path.join(p2simulation, model_name + "_final.h5"))

        _x, _y = get_pumpkin_set(n_samples=2000,
                                 uniform="non-uni" not in model_name).data2numpy(for_keras=True)
        if subset == "test":
            xdata, ydata = split_simulation_data(xdata=_x, ydata=_y, only_test=True)
        else:
            # xdata, ydata = ...
            raise NotImplementedError("Implement for other subsets if required!")

        analyzer = innvestigate.create_analyzer(analyzer_type, _model, disable_model_checks=True,
                                                neuron_selection_mode="max_activation")

        rel_obj = analyzer.analyze(xdata, neuron_selection=None).squeeze()
        # can do multiple samples in e.g. xdata.shape (200, 98, 98, 1)

        # rel_dict = {}
        # start = datetime.now()
        # for sub in range(len(ydata)):
        #     img = xdata[sub].copy()
        #     img = img[np.newaxis, ...]
        #     # Generate relevance map
        #     a = analyze_model(mri=img, analyzer_type=analyzer_type, model_=_model, norm=False)
        #     # Fill dict
        #     rel_dict.update({sub: a})
        #     loop_timer(start_time=start, loop_length=len(ydata), loop_idx=sub,
        #                loop_name="Generate Heatmaps")

        if save:
            save_obj(obj=rel_obj, name=model_name + f"_relevance-maps_{subset}-set", folder=p2simulation)

    return rel_obj


# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<
# # Testing

if __name__ == "__main__":

    import seaborn as sns

    # # Create heatmaps for all models on testset
    for fn in os.listdir(p2simulation):
        # find(fname="final.h5", folder=p2simulation, typ="file", exclusive=False, fullname=False,
        #      abs_path=True, verbose=False)
        if "final.h5" in fn:
            model_name = crop_model_name(model_name=fn)
            cprint(f"\nCreate heatmaps for {model_name}\n", col="p", fm="bo")
            rel_obj = create_relevance_dict(model_name=model_name, subset="test", save=True)

            # Plot heatmaps for N random tori
            plot_simulation_heatmaps(model_name=model_name, n_subjects=20, subset="test", pointers=True,
                                     true_scale=False)

            # Check sum relevance depending on model prdiction
            model = keras.models.load_model(os.path.join(p2simulation, fn))

            pdata = get_pumpkin_set(n_samples=2000, uniform="non-uni" not in model_name)
            x, y = pdata.data2numpy(for_keras=True)
            xtest, ytest = split_simulation_data(xdata=x, ydata=y, only_test=True)

            pred = model.predict(xtest)
            perf = np.mean(np.abs(pred-ytest[..., np.newaxis]))  # MAE
            print(f"{model_name} with MAE of {perf:.2f}")

            # Compute sum relevance
            sumR = [np.sum(rel_obj[sub]) for sub in range(len(ytest))]

            # Plot Sum.R as function of prediction
            cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="dark", as_cmap=True)
            fig, ax = plt.subplots()
            plt.title(model_name)
            ax.scatter(pred, sumR, c=np.sign(sumR), cmap=cmap)  # cmap="bwr"
            plt.hlines(y=0, xmin=plt.xlim()[0], xmax=plt.xlim()[1], alpha=.5, linestyles="dashed")
            plt.vlines(x=y.mean(),  # target bias
                       ymin=plt.ylim()[0], ymax=plt.ylim()[1], label=f"target bias = {y.mean():.1f}",
                       colors="pink",
                       alpha=.5, linestyles="dotted")
            plt.xlabel("prediction")
            plt.ylabel("sum relevance")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(p2simulation, model_name, "Sum_relevance_over_Prediction.png"))
            plt.savefig(os.path.join(p2simulation, model_name, "Sum_relevance_over_Prediction.pdf"))
            # plt.show()
            plt.close()

# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o  END
