# Credit Card Fraud Detection Project

## Project Overview

This is a simple undergraduate machine learning project for detecting credit card fraud using Python and scikit-learn.

## Files Structure

```
project_folder/
├── creditcard.csv           # Dataset from Kaggle
├── training.py          # Training script
├── test_model.py           # Testing script
├── model.pkl               # Trained model (generated after training)
├── scaler.pkl              # Feature scaler (generated after training)
├── model_info.pkl          # Model information (generated after training)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Requirements

### Python Packages

```
pandas
numpy
scikit-learn
matplotlib
seaborn
pickle (built-in)
```

### Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## How to Use

### Step 1: Setup

1. Download the dataset from Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud
2. Place `creditcard.csv` in your project folder
3. Install required packages

### Step 2: Training

```bash
python training.py
```

This will:

- Load and preprocess the data
- Train multiple models (Random Forest, Logistic Regression, Decision Tree)
- Compare their performance
- Save the best model as `model.pkl`
- Generate training visualizations

### Step 3: Testing

```bash
python test_model.py
```

This will:

- Load the trained model
- Make predictions on test data
- Evaluate performance
- Generate testing visualizations
- Save predictions to CSV

## Model Comparison

The project compares three algorithms:

1. **Random Forest** - Good for complex patterns
2. **Logistic Regression** - Simple and interpretable
3. **Decision Tree** - Easy to understand

## Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: How many predicted frauds are actually frauds
- **Recall**: How many actual frauds are detected
- **F1-Score**: Balance between precision and recall

## Key Features

### Data Preprocessing

- Handles missing values
- Scales features for algorithms that need it
- Maintains class balance in train-test split

### Model Selection

- Automatically selects best model based on F1-score
- F1-score is ideal for imbalanced datasets like fraud detection

### Visualization

- Model performance comparison
- Confusion matrices
- Feature importance (for tree-based models)
- Probability distributions
- Threshold analysis

## Output Files

After running the scripts, you'll get:

- `model.pkl` - Trained model
- `scaler.pkl` - Feature scaler
- `model_info.pkl` - Model metadata
- `training_results.png` - Training visualizations
- `testing_results.png` - Testing visualizations

## Understanding the Results

- **High Precision**: Few false fraud alerts
- **High Recall**: Catches most actual frauds
- **High F1-Score**: Good balance for fraud detection

## Tips for Improvement

1. **Feature Engineering**: Create new features from existing ones
2. **Hyperparameter Tuning**: Optimize model parameters
3. **Ensemble Methods**: Combine multiple models
4. **Class Balancing**: Use techniques like SMOTE for imbalanced data
5. **Cross-Validation**: More robust model evaluation

## Dataset Information

The dataset contains:

- 284,807 transactions
- 492 frauds (0.17% of all transactions)
- 30 features (28 PCA-transformed, Amount, Time)
- Binary target (0=Normal, 1=Fraud)

## Project Objectives Achieved

✅ **Objective 1**: Review and preprocess credit card transaction datasets
✅ **Objective 2**: Apply machine learning algorithms for fraud detection
✅ **Objective 3**: Compare different algorithms' effectiveness

## Common Issues and Solutions

1. **Memory Error**: Reduce dataset size or use sampling
2. **Imbalanced Data**: Focus on precision, recall, and F1-score rather than accuracy
3. **Slow Training**: Start with smaller datasets or simpler models
4. **Poor Performance**: Check data quality and feature scaling

## Next Steps

1. Implement more advanced algorithms (XGBoost, Neural Networks)
2. Add real-time prediction capabilities
3. Create a web interface for fraud detection
4. Implement model monitoring and retraining
