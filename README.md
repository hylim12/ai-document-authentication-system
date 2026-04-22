# AI-Powered Document Authentication for Anti-Money Laundering (AML) Systems

**Author:** Eldeena Lim Huey Yinn  
**Supervisor:** Prof. Ts. Dr. Tee Connie  
**Institution:** Faculty of Information Science and Technology, Multimedia University (MMU)  
**Academic Year:** 2025/2026  

---

# 1. Introduction

The increasing sophistication of document forgery techniques presents a significant challenge to Anti-Money Laundering (AML) compliance processes. Traditional manual verification methods are often insufficient due to human limitations such as fatigue, subjectivity, and inconsistency.

This project proposes an automated document authentication system that integrates **Computer Vision (CV)**, **Named Entity Recognition (NER)**, and **Machine Learning (ML)** to detect forged identity documents. The system aims to enhance detection accuracy, improve operational efficiency, and provide explainable outputs suitable for compliance auditing.

---

# 2. Objectives

The primary objectives of this system are as follows:

- To extract forensic features from identity documents using computer vision techniques  
- To perform semantic validation using Named Entity Recognition (NER)  
- To detect anomalies indicative of document forgery  
- To classify documents as *Authentic* or *Forged* using supervised machine learning  
- To provide interpretable results through visual and quantitative analysis  

---

# 3. System Architecture

The overall system pipeline is illustrated below:

dataset_builder.py -> train.csv / val.csv / test.csv -> ml_model_training.ipynb (Model Training + Evaluation) -> trained_model.pkl -> predict_one.py (Inference)


The system is designed to ensure consistency between training and inference pipelines, thereby maintaining robustness and reproducibility.

---

# 4. Methodology

## 4.1 Feature Extraction (Computer Vision)

The system performs multi-level feature extraction, including:

- **Character-Level Features:** height, width, aspect ratio, ink density  
- **Structural Features:** alignment, spacing, layout consistency  
- **Background Features:** intensity distribution and texture anomalies  

Image preprocessing techniques such as CLAHE (Contrast Limited Adaptive Histogram Equalization) and Otsu’s binarization are applied to enhance feature quality.

---

## 4.2 Optical Character Recognition and NER

Text is extracted using PaddleOCR, followed by rule-based Named Entity Recognition to identify key document fields such as:

- Name  
- Date of Birth  
- Document Number  
- Nationality  

Logical validation is performed to ensure consistency (e.g., chronological order of dates).

---

## 4.3 Forensic Anomaly Detection

Statistical anomaly detection is applied using Z-score analysis to identify irregularities in:

- Character geometry  
- Ink intensity  
- Edge gradients  
- Background consistency  

Detected anomalies are aggregated and used to compute a document-level risk score.

---

## 4.4 Machine Learning Classification

A Random Forest classifier is employed to perform binary classification:

- Class 0: Authentic  
- Class 1: Forged  

The model is trained using features derived from CV, NER, and anomaly detection processes.

---

# 5. Evaluation Methodology

All evaluation procedures are conducted within the Jupyter Notebook (`ml_model_training.ipynb`) to ensure transparency and reproducibility.

---

## 5.1 Machine Learning Metrics

The following performance metrics are used:

- Accuracy  
- Precision  
- Recall (Detection Rate)  
- F1-score  
- Specificity  
- False Positive Rate (FPR)  
- Receiver Operating Characteristic – Area Under Curve (ROC-AUC)  

---

## 5.2 NER Evaluation Metrics

To assess the effectiveness of information extraction:

- Field Extraction Completeness (Recall)  
- Precision  
- F1-score  
- Missing Field Analysis  

---

## 5.3 CV + AML Hybrid Metrics (Proposed)

This project introduces domain-specific evaluation metrics:

- **OCR Confidence Mean** (Text Extraction Reliability)  
- **Risk Score** (Forgery likelihood indicator)  
- **Risk Consistency Score** (alignment between anomalies and risk)  
- **Field Completeness Rate**  

These metrics provide a more comprehensive evaluation aligned with AML requirements.

---

# 6. Implementation Details

## 6.1 File Descriptions

### `feature_engineering.py`
Core module responsible for:
- Image preprocessing  
- Feature extraction  
- OCR and NER processing  
- Anomaly detection  

---

### `dataset_builder.py`
Generates structured datasets by:
- Processing image folders  
- Extracting features  
- Assigning ground truth labels  
- Exporting CSV files  

---

### `ml_model_training.ipynb`
Primary experimental environment:
- Model training  
- Performance evaluation  
- Metric computation  
- Result visualization  

---

### `predict_one.py`
Deployment script for:
- Single-document inference  
- Prediction output  
- Visualization generation  

---

# 7. Usage Instructions

## 7.1 Dataset Generation

python dataset_builder.py
This produces:

dataset_outputs/
    train.csv
    val.csv
    test.csv

## 7.2 Model Training and Evaluation

Open and execute:
- ml_model_training.ipynb

Outputs include:
- Trained model (.pkl)
- Performance metrics
- Evaluation visualizations

## 7.3 Inference
python predict_one.py

Outputs:
- Classification result
- Confidence score
- Annotated visualization

# 8. Data Integrity and Validity

To ensure methodological correctness:

Features that directly encode labels (e.g., Detection_Label) are excluded
Training and inference pipelines use consistent feature sets
Evaluation is performed on unseen test data

These measures prevent data leakage and ensure reliable performance estimation.

# 9. Contributions

This project contributes the following:

A hybrid CV–NER–ML framework for document authentication
Novel AML-oriented evaluation metrics
An explainable system with visual anomaly localization
A scalable pipeline adaptable to multi-country datasets

# 10. Limitations and Future Work

Future enhancements may include:

Deep learning architectures (e.g., CNNs, Vision Transformers)
Expansion to additional document types
Integration with cloud-based AML systems (e.g., AWS deployment)
Real-time processing capabilities

# 11.  Conclusion

This project demonstrates the feasibility of integrating computer vision, semantic analysis, and machine learning to address document forgery detection within AML systems. The proposed approach provides a robust, interpretable, and scalable solution suitable for real-world compliance applications.