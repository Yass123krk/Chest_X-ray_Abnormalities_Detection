# Chest X-ray Abnormalities Detection

---

## Project Overview

### 1. Introduction
The **VinBigData Chest X-ray Abnormalities Detection** project focuses on building an AI-based solution for detecting thoracic abnormalities from chest X-ray images. This project utilizes the **VinBigData dataset**, which includes a wide range of annotated chest conditions, aiming to improve diagnostic accuracy in the medical field.

This project is based on the [VinBigData Chest X-ray Abnormalities Detection Challenge](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/overview) hosted on Kaggle. The challenge invited over 1,200 teams to develop models that could classify and localize 14 thoracic abnormalities using 18,000 expert-annotated X-ray images.

By automating the detection process, this project helps reduce the workload on radiologists and speeds up diagnostics in healthcare systems, especially in environments with a high volume of cases. AI’s consistency in analyzing medical images ensures accurate and reliable predictions, helping improve patient outcomes through early detection.

<p align="center">
  <img src="./Images/Figure1.png" alt="Thoracic Abnormalities Detection Overview" />
</p>

### 2. Key Goals
- **Develop a robust AI model** capable of detecting 14 specific thoracic conditions, including aortic enlargement, atelectasis, and cardiomegaly.
- **Alleviate the workload on radiologists** by providing an AI-assisted tool that can quickly and accurately identify abnormalities in chest X-rays.
- **Ensure consistent and accurate diagnostic predictions**, improving the reliability of healthcare diagnostics and reducing variability in radiologic interpretation.

----

## Repository Structure

The repository is organized as follows:
```bash
Chest_X-ray_Abnormalities_Detection/
│
├── Images/                                      # Contains figures used in the README and project report
│   ├── Figure1.png                              # Pie chart of abnormality distribution
│   ├── ...                                      # Annotations per class
├── Proc_data/                                   # Processed datasets (original sizes)
│   ├── train/                                   # Training data images
│   ├── test/                                    # Test data images
│   ├── Original_Image_Dimensions.csv            # CSV file with original image dimensions
│   ├── Test_Image_Dimensions.csv                # CSV file with test image dimensions
│   ├── train.csv                                # Training dataset metadata
│   └── sample_submission.csv                    # Sample submission file
│
├── proc_data_512/                               # Resized dataset (512x512)
│   ├── train/                                   # Resized training data
│   ├── test/                                    # Resized test data
│   ├── Original_Image_Dimensions.csv            # CSV file with original image dimensions for resized data
│   └── Test_Image_Dimensions.csv                # CSV file with test image dimensions for resized data
│
├── results/                                     # Model results and logs
│   ├── debug/                                   # Debugging data, logs, and final model checkpoints
│   │   ├── inference/                           # Inference results
│   │   ├── AP40.png                             # Average precision curve at IoU 0.40
│   │   ├── loss.png                             # Training loss plot
│   │   ├── flags.yaml                           # Model training configuration
│   │   └── metrics.json                         # Performance metrics for debugging
│   ├── det/                                     # Final detection results
│   │   ├── vinbigdata_0_aug0.jpg                # Example detection results for image 0
│   │   ├── vinbigdata_1_aug0.jpg                # Example detection results for image 1
│   └── test_512x512/                            # Test set results
│       └── submission.csv                       # CSV file for submission
│
├── v20/                                         # Model training metrics for version 20
│   └── metrics.json                             # JSON file containing metrics for this model version
│
├── DataResize.ipynb                             # Jupyter notebook for resizing the dataset images
├── DataVisualisation.ipynb                      # Notebook for visualizing and analyzing the dataset
├── ModelTraining.ipynb                          # Notebook for training the model
├── ModelTesting.ipynb                           # Notebook for testing and evaluating the trained model
├── Report.pdf                                   # Full project report detailing methodology and results
└── requirements.txt                             # Project dependencies and libraries
```
---

## System Requirements

To run this project efficiently, a system with adequate computational resources is necessary due to the large dataset and intensive model training involved. Below are the recommended hardware and software specifications:

### 1. Hardware Requirements
- **Minimum RAM**: 8 GB (16 GB recommended)
- **Processor**: Quad-core CPU or higher (recommended for parallel processing during model training)
- **Storage**: At least 50 GB of free disk space for datasets, models, and results
- **GPU (optional but highly recommended)**: A CUDA-compatible GPU (e.g., NVIDIA) is suggested for faster training of the deep learning models, particularly when using TensorFlow or PyTorch.

### 2. Software Requirements
- **Operating System**: Windows, macOS, or Linux
- **Python Version**: Python 3.8 or higher is required due to dependencies on newer libraries.
- **CUDA Toolkit (optional)**: Required for leveraging GPU-based training (only if using a compatible NVIDIA GPU)

### 3. Key Libraries and Tools
This project relies on several open-source libraries and tools to preprocess the data, train the model, and evaluate its performance.

#### 3.1. Programming Languages
- **Python 3.8+**: The primary programming language used for data preprocessing, model training, and evaluation.

#### 3.2. Machine Learning Frameworks
- **PyTorch**: The main deep learning framework used to train the object detection model.
- **Detectron2**: A Facebook AI Research (FAIR) library built on PyTorch, used for implementing object detection tasks.
  
#### 3.3. Data Handling and Visualization Libraries
- **NumPy**: For handling multi-dimensional arrays and performing mathematical operations.
- **Pandas**: Used for data manipulation, handling CSV files, and processing dataset metadata.
- **OpenCV**: Utilized for image manipulation, resizing, and transformations.
- **Matplotlib** and **Seaborn**: Libraries for data visualization and graphical representation of model performance metrics.

#### 3.4. Additional Libraries
- **Albumentations**: A fast and flexible image augmentation library for advanced image transformations, which is essential for augmenting the dataset to increase diversity.
  
To install all the required libraries, run the following command:

```bash
pip install -r requirements.txt

---

## Dataset

### 1. VinBigData Chest X-ray Dataset
The dataset used for this project is the **VinBigData Chest X-ray Dataset**, consisting of 18,000 chest X-ray images. These images are annotated with 14 different thoracic abnormalities:

- Aortic enlargement
- Cardiomegaly
- Pleural thickening
- Pulmonary fibrosis
- Nodule/Mass
- Lung opacity
- Pleural effusion
- Other lesion
- Infiltration
- ILD (Interstitial Lung Disease)
- Calcification
- Consolidation
- Atelectasis
- Pneumothorax

This dataset provides a comprehensive foundation for training models to detect various abnormalities, which significantly aids in automating diagnostic processes.

<p align="center">
  <img src="./Images/Figure2.png" alt="Distribution of Thoracic Abnormalities in the Dataset" />
</p>


### 2. Data Preprocessing
The preprocessing of this dataset is pivotal to ensuring its compatibility with machine learning models and optimizing the computational load during training and analysis. This stage encompasses several essential steps:

### 2.1. Conversion to PNG
To facilitate easier manipulation and compatibility with a broader range of image processing tools, the X-ray images were converted from the original **DICOM format** to **PNG format**. This conversion not only reduces file sizes significantly (from approximately 200 GB to around 0.5 GB) but also simplifies the application of image processing techniques, which is crucial for large-scale model training.


### 2.2. Image Resizing
The images were resized to two standard resolutions: **256x256** and **512x512** pixels. Resizing serves several purposes:
- **Reduced computational resource requirements**: Smaller images require less memory and processing power, speeding up model training and reducing hardware constraints.
- **Faster training times**: Training on smaller images accelerates the overall process.
- **Decreased risk of overfitting**: Smaller images limit the number of learnable parameters, preventing the model from overfitting to the data.
- **Consistency**: Using standardized image sizes ensures uniformity, which is essential for batch processing in most deep learning frameworks.

<p align="center">
  <img src="./Images/Figure5.png" alt="Image Resizing for Preprocessing" />
</p>

### 2.3. Monochromatism Correction
Some images in the dataset exhibited **monochromatism**, which could skew the results and lead to misinterpretations by computer vision models. A specific preprocessing step was applied to correct these monochrome images, ensuring consistency in visual data representation. This correction ensures that all images have similar chromatic properties, allowing the model to learn features effectively without being confused by variations in intensity and contrast.

<p align="center">
  <img src="./Images/Figure6.png" alt="Monochromatism Correction Process" />
</p>

### 3. Data Split
The dataset is divided into three subsets to allow for training, validation, and testing:

- **Training Set**: The majority of the data is used for training the model, learning the patterns associated with the different abnormalities.
- **Validation Set**: A smaller subset is used during model training to adjust hyperparameters and monitor performance.
- **Testing Set**: This final subset is held back to assess the model’s performance on unseen data.

<p align="center">
  <img src="./Images/Figure3.png" alt="Bounding Box Area Distribution in the Dataset" />
</p>




