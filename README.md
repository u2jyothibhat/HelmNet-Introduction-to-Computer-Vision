# HelmNet: Introduction to Computer Vision

A deep learning image classifier that detects whether a worker is wearing a safety helmet. Four convolutional neural network models are built and compared, from a custom CNN to VGG-16 transfer learning, and the selected model classifies every image in the held-out test set correctly.

Completed as the Introduction to Computer Vision project of the Great Learning postgraduate programme in AI and Machine Learning.

## Business context

Safety helmets protect workers on construction sites and in industrial plants from head injuries caused by falling objects and machinery. Checking compliance by hand is slow and error-prone, particularly across large operations.

SafeGuard Corp wants an automated image analysis system that detects whether workers are wearing helmets, so that safety rules can be enforced consistently, at scale and with less reliance on manual oversight.

## Objective

Build an image classification model that sorts images into two classes and is ready to form part of an automated, real-time monitoring system:

- **With helmet** (label 1): workers wearing a safety helmet
- **Without helmet** (label 0): workers not wearing a safety helmet

## Dataset

| File | Description |
| --- | --- |
| `images.npy` | 631 RGB images, 200 x 200 pixels, stored as a NumPy array of shape `(631, 200, 200, 3)` |
| `Labels.csv` | One label per image in a single `Label` column: 1 = with helmet, 0 = without helmet |

- 311 images show workers with helmets and 320 show workers without, so the classes are close to balanced.
- The images cover construction sites, factories and other industrial settings, with varied lighting, camera angles and worker postures.
- Workers appear standing, using tools and moving, to reflect real working conditions.

## Approach

1. **Data overview and exploratory analysis**: load the images and labels, view random samples from each class and check the class balance.
2. **Preprocessing**: convert the images to greyscale, scale pixel values to the 0 to 1 range, and make a stratified 70/15/15 split into training (441), validation (95) and test (95) sets.
3. **Model building**: train four models and evaluate each on accuracy, recall, precision and F1 score, with confusion matrices and sample predictions.
4. **Model selection**: compare the models on the validation set, then evaluate the chosen model once on the test set.
5. **Error analysis and recommendations**: review misclassified images and set out insights for the business.

## Models and results

Validation set performance (95 images):

| Model | Accuracy | Precision | Recall | F1 score |
| --- | --- | --- | --- | --- |
| Model 1: Simple CNN | 0.979 | 0.979 | 0.979 | 0.979 |
| Model 2: VGG-16 (base) | 0.979 | 0.980 | 0.979 | 0.979 |
| **Model 3: VGG-16 + feed-forward network** | **0.989** | **0.990** | **0.989** | **0.989** |
| Model 4: VGG-16 + feed-forward network + data augmentation | 0.758 | 0.929 | 0.553 | 0.693 |

**Final model: Model 3 (VGG-16 base with a feed-forward classification head).** On the held-out test set of 95 images it scored 1.00 on accuracy, recall, precision and F1.

## Key insights

- Transfer learning with VGG-16 gave the best result. The pre-trained network supplies strong general image features, which matters with only 631 images.
- A deeper feed-forward classification head with dropout improved on the plain VGG-16 base.
- Data augmentation reduced performance here. Model 4 missed many helmet wearers (recall of 0.55), most likely because the transformations were too aggressive for such a small dataset.
- Error analysis on the simple CNN showed mistakes linked to head shape and haircut being read as a helmet, and to sun glare hiding a helmet.

## Recommendations

- Start with pilot deployments in controlled settings, such as sites with fixed cameras, to test how the model generalises to real conditions and edge cases.
- Connect the model to an edge or cloud monitoring pipeline that raises an alert when a worker is detected without a helmet.
- Collect more labelled images, especially difficult cases such as blur, occlusion and poor lighting, to make the model more robust.
- Monitor the deployed model for bias and failure cases, and retrain it as new data arrives.

## Repository contents

| File | Description |
| --- | --- |
| `JyothiK_HelmNet.ipynb` | Project notebook with code, outputs, observations and recommendations |
| `images.npy` | Image dataset |
| `Labels.csv` | Image labels |

## Running the notebook

The notebook was written for Google Colab with a T4 GPU runtime (Runtime > Change runtime type > T4 GPU) and uses TensorFlow 2.19.

It loads the data from Google Drive as `images_proj.npy` and `Labels_proj.csv`. To run it with the files in this repository, change the two paths in the data-loading cell to point to `images.npy` and `Labels.csv`.

## Tools

Python, TensorFlow and Keras, VGG-16, NumPy, pandas, scikit-learn, OpenCV, Matplotlib and Seaborn.
