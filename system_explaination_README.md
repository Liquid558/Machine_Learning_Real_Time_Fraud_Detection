# Credit Card Fraud Detection System

This is a machine learning-based system for detecting fraudulent credit card transactions. The system is trained on the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), and can be used to analyze new transactions and predict whether they are **legitimate** or **fraudulent** based on learned patterns.

---

## Project Highlights

- Uses anonymized credit card transaction data
- Implements multiple ML algorithms (Random Forest, Logistic Regression, Decision Tree)
- Selects and saves the best-performing model automatically
- Supports prediction of single or batch transactions
- Provides confidence scores and risk levels for decisions
- Includes visualization of model performance and feature importance

---

## How It Works

1. The system is trained using a dataset of 284,807 transactions labeled as `legitimate` or `fraudulent`.
2. Feature scaling is applied where necessary (e.g., for logistic regression).
3. Several machine learning models are trained and evaluated using accuracy, precision, recall, and F1-score.
4. The best model is selected and saved along with a scaler and metadata.
5. A separate testing interface allows you to generate and test synthetic or real transaction data against the trained model.

---

## Project Structure

```
├── training.py              # Model training and evaluation
├── fraud_detector.py      # Fraud detection class with prediction logic
├── creditcard.csv           # Dataset (not included by default)
├── model.pkl                # Saved best-performing model
├── scaler.pkl               # Feature scaler (if used)
├── model_info.pkl           # Metadata about the training session
├── training_results.png     # Visualizations of model performance
└── README.md                # This file
```

---

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---

## How to Use

### Step 1: Prepare the Dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in your project folder.

### Step 2: Train the Model

Run:

```bash
python training.py
```

This will:

- Train 3 models
- Evaluate and compare them
- Save the best model and scaler

### Step 3: Test the Model

In a Jupyter Notebook or Python script:

```python
from detection_system import FraudDetectionSystem

fraud_detector = FraudDetectionSystem()
fraud_detector.load_model()
fraud_detector.batch_test_transactions(num_transactions=10)
```

This will:

- Load your trained model
- Run predictions on 10 test transactions
- Print a table with predictions, fraud confidence, and labels

---

## Sample Output

```
Batch Test Results:
 id     amount  prediction  fraud_confidence
  1  9999.0000        FRAUD             95.73
  2    15.0000   LEGITIMATE              3.42
  3   321.1287   LEGITIMATE              7.56
```

---

## Visualizations

- **Model Performance Comparison** (Accuracy, Precision, Recall, F1)
- **Confusion Matrix** of the best model
- **Top 10 Feature Importances** (for tree-based models)
- **Class Distribution Bar Chart**

These are automatically generated and saved as `training_results.png`.

---

## Future Improvements

- Support for real-time streaming data (e.g., via Kafka or REST API)
- Integration with a web dashboard or admin panel
- Model retraining with more recent or larger datasets
- Feature engineering for improved accuracy

---

## License

This project is for **educational and research purposes** only. Not suitable for production use without additional security, validation, and data handling compliance (e.g., GDPR, PCI DSS).

---
