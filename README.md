# HelmNet: Introduction to Computer Vision

Image classification project that detects whether a worker is wearing a safety helmet, using convolutional neural networks and VGG-16 transfer learning.

## Contents

| File | Description |
| --- | --- |
| `JyothiK_HelmNet.ipynb` | Project notebook with code, outputs, observations and recommendations |
| `images.npy` | Image dataset: 631 RGB images, 200 x 200 pixels |
| `Labels.csv` | Labels for the images: 1 = with helmet, 0 = without helmet |

## Running the notebook

The notebook was written for Google Colab and loads the data from Google Drive as `images_proj.npy` and `Labels_proj.csv`. To run it with the files in this repository, update the two paths in the data-loading cell to point to `images.npy` and `Labels.csv`.
