# AML Cancer Classification using Multimodal Machine Learning

## Overview
This project presents a multimodal machine learning approach for Acute Myeloid Leukemia (AML) subtype classification by combining:

- Microscopy blood smear image features
- Clinical laboratory and patient data

The system integrates deep learning–based image feature extraction with machine learning classification models to improve diagnostic accuracy and support automated AML subtype detection.

---

## Objectives
- Develop an automated AML classification system
- Extract meaningful image features using pretrained CNN models
- Analyze clinical data for subtype prediction
- Combine image and clinical features using multimodal learning
- Compare multiple machine learning models and evaluate performance

---

## Dataset

### Source
The dataset was obtained from:

- The Cancer Imaging Archive (TCIA) Blood Cancer Dataset

### Citation
Hehr, M., Sadafi, A., Matek, C., Lienemann, P., Pohlkamp, C., Haferlach, T., Spiekermann, K., & Marr, C. (2023). *A morphological dataset of white blood cells from patients with four different genetic AML entities and non-malignant controls (AML-Cytomorphology_MLL_Helmholtz)* (Version 1) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/6PPE-4020

### Dataset Details
- ~81,000 microscopy blood smear images
- 189 patients
- Multiple AML subtypes and control samples
- Clinical laboratory measurements and demographic data

---

## Project Structure

### 01_clinicalBranch.ipynb
- Clinical data preprocessing
- Missing value handling
- Feature analysis
- Correlation analysis
- Clinical model training

### 02_featureExtraction.ipynb
- Image preprocessing
- CNN feature extraction using pretrained models
- Feature extraction using EfficientNetB0, ResNet50, and MobileNetV2

### 03_imageBranchComparison.ipynb
- Patient-level feature aggregation
- Comparison of image-based classification models
- Random Forest
- XGBoost
- MLP

### 04_multiModalComparison.ipynb
- Fusion of image and clinical features
- Multimodal feature comparison
- Cross-validation evaluation

### 05_finalModel.ipynb
- Trains the selected final model on the full dataset
- Saves the final trained model and preprocessing objects
- Prepares the model for prediction and deployment

### 06_prediction.ipynb
- Prediction pipeline for new patient samples
- End-to-end inference workflow

---

## Methodology

### Image Processing
- Microscopy images resized and normalized
- Deep feature extraction using pretrained CNN architectures
- Mean aggregation of image embeddings at patient level

### Clinical Data Processing
- Missing value imputation
- Feature scaling and preprocessing
- Statistical and correlation analysis

### Machine Learning Models
The following models were evaluated:
- Random Forest
- XGBoost
- Multi-Layer Perceptron (MLP)

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Weighted F1-score
- Confusion Matrix
- Stratified Cross-Validation

---

## Technologies Used
- Python
- TensorFlow / Keras
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Matplotlib
- Streamlit

---

## Results
The multimodal approach achieved improved classification performance compared to single-modality approaches by leveraging both morphological image features and clinical information.

---

## Application
A Streamlit-based web application was developed to allow prediction of AML subtypes using uploaded microscopy images and clinical patient data.

The application also includes an AI-powered chatbot assistant to provide user guidance, project information, and prediction support through an interactive interface.

---

## Future Improvements
- Fine-tuning CNN architectures
- Data augmentation for image branch
- Larger and more balanced datasets
- Explainable AI integration
- Deployment as a clinical decision-support tool

---

## Authors
- Yathindu Jayawardhane
- Ravindi Gunasekara
- Sindupa Ekanayake
- Sudara Jayalath