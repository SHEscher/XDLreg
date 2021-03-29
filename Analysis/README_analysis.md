# LIFE – Analysis

`[Last update: 2020-04-15]`

Status (optional): analysis phase / *work in progress*
Begin of project: Sep 1, 2018

## Description of Analysis
*List relevant information one need to know about the (raw) data. Could include how data was aquired*

### Modelling

#### MRInet
In `./Analysis/Modelling/MRInet/` are the implementation of 3D-convolutional neural networks (CNN) models for the prediction from MRI images.  

#### LRP
In `./Analysis/Modelling/LRP/` one can find the implementation of the layer-wise relevance propagation algorithm (LRP), which highlights information in the input space being relevant for the given model prediction. <br>
Here, the LRP will be applied on the predictions of the MRInet (above)
extracting voxels in the MRI image that were relevant for the
prediction.

### Subanalysis
The subfolder `./Analysis/Subanalysis/`  contains mainly jupyter notebooks for subanalyses, for instance for checking different interpretation methods for the deep learning model.

#### Convert `jupyter` notebooks to presentation slides

For presentations of the notebooks `Subanalysis/` one first needs to define for each cell in a notebook whether it is a slide, and what kind (see this [blog-post](https://medium.com/learning-machine-learning/present-your-data-science-projects-with-jupyter-slides-75f20735eb0f)).
Then one can convert the notebook to slides via the following methods:

- Run command in shell and open resulting html file:
```shell
jupyter nbconvert NOTEBOOK.ipynb --to slides --SlidesExporter.reveal_scroll=True
```
- Use [`RISE`](https://rise.readthedocs.io/en/stable/dev/index.html) inside the local jupyter server @ http://localhost:8888/ which allows for online code-execution
- Use bash file from within the folder and open resulting html, however this will export all notebooks in folder to html-slides:
```bash
bash export_all_notebookes_to_slides.sh
```
- Export the notebook from `jupyter`'s local server as shown in the image below (via `Reveal.js`).
Note: If vertical scrolling on slides is required (usually necessary for long cell-outputs), one need to switch the scroll-option to `true` in the exported html file via an editor, by setting
```html
function setScrollingSlide(){
  var scroll = true
  ...
}
```  

![how-to-export](./Subanalysis/Export_to_presentation.png)

## COPYRIGHT/LICENCSE
*One could add information about code sharing, license and copy right issues*
