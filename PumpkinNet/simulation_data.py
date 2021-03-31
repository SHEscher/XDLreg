"""
Create simulation data.

Author: Simon M. Hofmann | <[firstname].[lastname][at]pm.me> | 2021
"""

# %% Import

import os
import string
import timeit
from datetime import datetime, timedelta
import concurrent.futures

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import draw

from utils import root_path, cprint, chop_microseconds, save_obj, load_obj, loop_timer, function_timed

# %% Set global params << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >

# Set params
max_age = 80
min_age = 20  # OR, e.g., min_age=4 for developmental factors (here: size of head)

# Set paths
p2data = os.path.join(root_path, "Data")


# %% Create image data ("Pumpkins") << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >>

def make_pumpkin(age: int, img_size: tuple = (98, 98)) -> np.ndarray:
    """
    Create elliptic shape of random (head) size. Thickness grows proportionally. Add noise.
    This is template for the PumpkinHead class.
    :param age: age of pumpkin, here only for general 'head' size of pumpkin.
    :param img_size: image size
    :return: pumpkin head
    """
    # Size of pumpkin with some random variance
    p_size = (40 + 2 * np.clip(age, 0, 20),  # i.e. for age >= 20, standard 'brain' size is (80, 65) + v
              25 + 2 * np.clip(age, 0, 20))
    p_size += np.random.normal(loc=0, scale=p_size[0] / 20, size=2)  # v

    # Draw outer ellipse
    rr, cc = draw.ellipse(r=img_size[0] // 2, c=img_size[1] // 2,
                          r_radius=p_size[0] / 2, c_radius=p_size[1] / 2,
                          rotation=0)

    # Draw inner ellipse
    rr_inner, cc_inner = draw.ellipse(r=img_size[0] // 2, c=img_size[1] // 2,
                                      r_radius=p_size[0] / 5, c_radius=p_size[1] / 5, rotation=0)

    # Define grid/image (as np.arary)
    pumpkin = np.zeros((98, 98), dtype=np.float32)

    # Create elliptic shape with hole
    pumpkin[rr, cc] = .8  # set outer ellipse to .8 (max will be 1)
    pumpkin[rr_inner, cc_inner] = 0  # create hole with inner ellipse

    # Add noise
    noise = 1 - np.random.randn(*pumpkin.shape) * 0.05
    pumpkin = np.multiply(pumpkin, noise)

    return pumpkin


def random_name() -> str:
    """
    Create random name for pumpkin head. This simulates the pseudonym in a human study (e.g., with MRI)
    :return random name
    """
    # PH + 3 Chars + 9 digit number
    rand_name = "PH" + "".join(np.random.choice(a=list(string.ascii_letters), size=3, replace=True))
    rand_name += str(np.random.randint(0, 10 ** 8)).zfill(9)  # with 1 leading zero
    return rand_name


class PumpkinHead:
    """Class of ageing pumpkin heads."""

    def __init__(self, age, name=None):
        """
        Create instance of PumpkinHead.
        :param age: age of pumpkin.
        :param name: name of pumpkin. If not given (None), will be automatically generated.
        """
        self.age = age
        self.name = random_name() if name is None else name
        self.pumpkin_brain = make_pumpkin(age=age)
        self.n_lesions = None
        self.lesion_coords = []
        self.n_atrophies = None
        self.atrophy_coords = []
        self.grow()

    def grow(self):
        """
        Run several ageing processes on self.pumpkin_brain as function of self.age
        """
        self.add_lesions()
        self.add_atrophies()

    def add_atrophies(self, **kwargs):
        """
        Linearly add atrophies to self.pumpkin_brain.
        Atrophies are probabilistically applied to surface area including inner surfaces
        Reduce image intensity in a certain range up to zero (i.e. maximal reduction).
        """

        max_atrophies = max_age * kwargs.pop("max_atrophies", 5)  # keep max_age here to clip below
        expected_n_atrophies = max_atrophies * self.age / max_age

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
                prob_atrophy = n_zeros / (n_values - 1)

                if np.random.binomial(n=1, p=prob_atrophy):
                    self.pumpkin_brain[xi, yi] = 0

                    self.atrophy_coords.append((xi, yi))  # add location of athropy to list

                    ctn_atrophies += 1

    def add_lesions(self, **kwargs):
        """
        Probabilistically, and non-linearly add lesions within the self.pumpkin_brain of a certain size.
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
            self.n_lesions = 0 if expected_n_lesions <= 0 + 5 else max_lesions  # 0 + small margin
        else:
            all_probs = np.zeros(max_lesions + 1)
            all_probs[np.unique(distr).astype(int)] = np.unique(distr, return_counts=True)[1]
            all_probs = all_probs / np.sum(all_probs)

            self.n_lesions = np.random.choice(a=range(max_lesions + 1), size=1, p=all_probs).item()

        # Get area of pumpkin head
        non_zeros_idx = np.nonzero(self.pumpkin_brain)

        # Add lesions
        ctn_lesions = 0
        while ctn_lesions < self.n_lesions:

            idx = np.random.randint(low=0, high=len(non_zeros_idx[0]), size=1).item()

            xi, yi = non_zeros_idx[0][idx], non_zeros_idx[1][idx]

            # Look at surrounding of coordinate: if not at boarders of pumpkin add a lesion
            if self.pumpkin_brain[xi - 2: xi + 3, yi - 2: yi + 3].min() > 0:
                lesion = np.clip(a=self.pumpkin_brain[xi - 1: xi + 2, yi - 1: yi + 2] + (
                        .2 + np.random.normal(scale=.025)),
                                 a_min=0, a_max=1)

                self.pumpkin_brain[xi - 1: xi + 2, yi - 1: yi + 2] = lesion

                self.lesion_coords.append((xi, yi))

                ctn_lesions += 1

    def exhibition(self, **kwargs):
        """
        Plot pumpkin head.
        :param kwargs: matplotlib.pyplot.imshow specific **kwargs.
        """
        plt.figure(num=f"{self.name} | age={self.age}")
        cmap = kwargs.pop("cmap", "gist_heat")
        plt.imshow(self.pumpkin_brain, cmap=cmap, **kwargs)
        plt.show()


# TODO 3D case: for instance, use MNI template and apply changes as function of age

# %% Create dataset << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

def generate_set_name(n_samples: int, uniform: str, age_bias) -> str:
    """
    Generate the name of a simulated dataset as function of given arguments.
    :param n_samples: number of samples in dataset
    :param uniform: whether dataset is uniformly distributed
    :param age_bias: None (only for uniform datasets) OR certain age int/flaot (uniform / non-uniform)
    :return: name of dataset (will be used as suffiix after time-stamp)
    """
    if not uniform:
        assert age_bias is not None, "For non-uniform datasets, age_bias must be provided."
    return f"N-{n_samples}_{'' if uniform else 'non-'}uniform" + ("" if uniform else f"{age_bias:.1f}")


class PumpkinSet:
    """Dataset class of simulated head images ('pumpkins')."""

    def __init__(self, n_samples: int, uniform: bool = True, age_bias: float = None,
                 skew_factor: float = .8, save: bool = True):
        """
        Create instance of PumpkinSet class.
        :param n_samples: number of samples in dataset
        :param uniform: whether dataset is uniformly distributed
        :param age_bias: for non-uniform datasets provide age-bias.
        :param skew_factor: parameter which influences the data distribution (only for non-uniform)
        :param save: Whether to save dataset externally.
        """
        self._n_samples = n_samples
        self._age_distribution = None
        self._age_bias = age_bias
        self._is_uniform = uniform
        self._draw_sample_distribution(uniform=uniform, age_bias=age_bias, skew_factor=skew_factor)
        self._data = [None] * n_samples
        self._name = datetime.now().strftime('%Y-%m-%d_%H-%M_') + generate_set_name(n_samples, uniform,
                                                                                   age_bias)
        self._generate_data()
        if save:
            self.save()

    @property
    def name(self) -> str:
        """Return name of simulated dataset."""
        return self._name

    @property
    def n_samples(self) -> int:
        """Return number of samples in simulated dataset."""
        return self._n_samples

    @property
    def is_uniform(self) -> bool:
        """Return whether simulated dataset is uniformaly distributed."""
        return self._is_uniform

    @property
    def age_distribution(self):
        """Return age distribution in simulated dataset."""
        return self._age_distribution

    @property
    def age_bias(self):
        """Return age bias in simulated dataset (None for uniformly distributed datasets)."""
        # Note this is not equal to the distribution mean
        return self._age_bias

    @property
    def data(self):
        """Return the simulated image data of class PumpkinHead."""
        return self._data

    def display_distrubtion(self) -> None:
        """Plot the age distribution of the simulated dataset."""
        plt.figure(f"Age distribution in {self.name}", figsize=(6, 4), dpi=150)
        h = sns.histplot(self.age_distribution, binwidth=1, kde=True, color="cadetblue", alpha=.5)
        ymax = int(h.axes.get_ylim()[-1] * .9)
        plt.vlines(x=30.5, ymin=0, ymax=ymax, colors="red", ls="dotted", label="age-bias")
        plt.vlines(np.mean(self.age_distribution), ymin=0, ymax=ymax, colors="orange", ls="dashed",
                   label="mean")
        plt.vlines(np.median(self.age_distribution), ymin=0, ymax=ymax, colors="green", ls="dashed",
                   label="median")
        h.axes.set_xlabel("age")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _draw_sample_distribution(self, uniform: bool, age_bias: float = None,
                                  skew_factor: float = .8) -> None:
        """
        Draw sample distrubtion either uniformly or non-uniformly (with certain age-bias and skew-factor).
        :param uniform: whether dataset is uniformly distributed
        :param age_bias: for non-uniform datasets provide age-bias.
        :param skew_factor: parameter which influences the data distribution (only for non-uniform)
        """
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

    @function_timed
    def _generate_data(self) -> None:
        """
        Generate image data for dataset (N=self.n_samples).
        Depending on sample size, this can take a while.
        """
        try:
            cprint(f"Start creating the pumpkin dataset of {self.name} ...", 'b')
            t = timeit.timeit(
                stmt="PumpkinHead(np.random.randint(low=min_age, high=max_age+1))",
                setup="from PumpkinNet.simulation_data import PumpkinHead, np, min_age, max_age",
                number=3)/3  # approx. time to create one PumpkinHead
            # Estimated time to create dataset
            t_est = (t * self.n_samples) / (os.cpu_count()*0.95)  # .95, here, factor to start process
            cprint(f"Creating whole dataset via parallel processing will take about: "
                   f"{chop_microseconds(timedelta(seconds=t_est))} [h:m:s] ...", col='y')

            start_time = datetime.now()
            # Worker n_CPU * 2 works best
            with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
                heads = executor.map(PumpkinHead, self.age_distribution)
            self._data = list(heads)
            cprint(f"Created {self.n_samples} pumpkins in "
                   f"{chop_microseconds(datetime.now() - start_time)} [hh:min:sec].", 'b')
            exit()  # TODO REMOVE AFTER TESTING

        except Exception as e:
            # print(e)
            start_time = datetime.now()
            for i, age in enumerate(self.age_distribution):
                self._data[i] = PumpkinHead(age=age)
                loop_timer(start_time=start_time, loop_length=self.n_samples, loop_idx=i,
                           loop_name="Create Pumpkin Dataset")

    def save(self) -> None:
        """Save this instance of PumpkinSet (self) externally."""
        if not os.path.exists(p2data):
            os.mkdir(p2data)
        save_obj(obj=self, name=self.name, folder=p2data)

    def data2numpy(self, for_keras: bool = True):
        """
        Convert dataset to numpy arrays.
        :param for_keras: add empty axis to image-data, such that a Keras model can take them as input.
        :return: dataset as numpy arrays (x: images; y: age-vector)
        """
        ydata = np.array([self.data[i].age for i in range(self.n_samples)])

        img_shape = self.data[0].pumpkin_brain.shape
        xdata = np.empty(shape=(self.n_samples, *img_shape))

        for i in range(self.n_samples):
            xdata[i] = self.data[i].pumpkin_brain

        if for_keras:
            xdata = xdata[..., np.newaxis]

        return xdata, ydata


def get_pumpkin_set(n_samples: int = 2000, uniform: bool = True, age_bias: float = None):
    """
    Get dataset (class PumpkinSet) with given properties (**kwargs) either from memory, or generates it.
    :param n_samples: number of samples in dataset
    :param uniform: whether dataset is uniformly distributed
    :param age_bias: for non-uniform datasets provide age-bias.
    :return: dataset (class PumpkinSet)
    """
    assert n_samples >= 100, "Simulated dataset can't be smaller than a 100 samples."

    df_files = os.listdir(p2data) if os.path.exists(p2data) else []
    f_suffix = generate_set_name(n_samples, uniform, age_bias)

    for file in df_files:
        if f_suffix in file:
            cprint(f"Found & loaded following file: {file} ...", 'b')
            return load_obj(name=file, folder=p2data)
            # return load_pumpkin_set(name=file)

    else:
        cprint(f"No dataset found. Start creating it ...", 'b')
        return PumpkinSet(n_samples=n_samples, uniform=uniform, age_bias=age_bias, save=True)


# %% Prepare data for model training << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >

def split_simulation_data(xdata: np.ndarray, ydata: np.ndarray, return_idx: bool = False,
                          only_test: bool = False):
    """
    Split simulation data into training, validation, and test set.
    :param xdata: image data
    :param ydata: age data (target)
    :param return_idx: whether to return index (for test set only) or indeces where data was split.
    :param only_test: whether to return only data of test set.
    :return: split data (training, validation, test; OR only test)
    """
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

# << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< END
