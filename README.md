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
## Dataset

### 1. VinBigData Chest X-ray Dataset
The dataset used for this project is the **VinBigData Chest X-ray Dataset**, consisting of 18,000 chest X-ray images. These images are annotated with 14 different thoracic abnormalities, such as:

- Aortic enlargement
- Atelectasis
- Cardiomegaly
- Pleural effusion
- Pulmonary fibrosis

This dataset serves as a comprehensive foundation for training models to detect these abnormalities, improving diagnostic accuracy through automation.

<p align="center">
  <img src="./Images/Figure1.png" alt="Distribution of Thoracic Abnormalities in the Dataset" />
</p>

### 2. DICOM Format and Image Conversion
The original chest X-ray images are stored in **DICOM (Digital Imaging and Communications in Medicine)** format, which is the standard for storing medical imaging data. DICOM images contain metadata about the image, patient, and study, which is essential in a clinical setting but less relevant for machine learning model training.

To simplify the processing pipeline, the DICOM images are converted to **PNG** format. PNG images are smaller and easier to handle while retaining the necessary quality for accurate anomaly detection. The conversion allows for faster data loading and efficient use of GPU resources during model training.

<p align="center">
  <img src="./Images/Figure5.png" alt="DICOM to PNG Conversion Example" />
</p>

<p align="center">
  <img src="./Images/Figure6.png" alt="Comparison of Original vs. Resized X-ray Images" />
</p>

### 3. Data Split
The dataset is split into three subsets to allow for training, validation, and testing:

- **Training Set**: The majority of the data is used to train the model, learning patterns associated with various abnormalities.
- **Validation Set**: A smaller portion is reserved for fine-tuning hyperparameters and assessing model performance during training.
- **Testing Set**: The final subset is used for evaluating the model's ability to generalize to new, unseen data.

<p align="center">
  <img src="./Images/Figure3.png" alt="Bounding Box Area Distribution in the Dataset" />
</p>




