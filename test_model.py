# Credit Card Fraud Detection - Testing Script

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_model():
    """Load the trained model, scaler, and model info"""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        return model, scaler, model_info
    except FileNotFoundError:
        print("Error: Model files not found. Please run the training script first.")
        return None, None, None

def predict_fraud(model, scaler, model_info, X_test):
    """Make predictions using the trained model"""
    if model_info['use_scaling']:
        X_test_processed = scaler.transform(X_test)
    else:
        X_test_processed = X_test
    
    predictions = model.predict(X_test_processed)
    probabilities = model.predict_proba(X_test_processed)[:, 1]  # Probability of fraud
    
    return predictions, probabilities

def evaluate_model(y_true, y_pred):
    """Evaluate model performance"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def plot_results(y_true, y_pred, y_prob, model_name):
    """Plot evaluation results"""
    plt.figure(figsize=(15, 10))
    
    # Confusion Matrix
    plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Classification Report Heatmap
    plt.subplot(2, 3, 2)
    report = classification_report(y_true, y_pred, output_dict=True)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='Blues')
    plt.title('Classification Report')
    
    # Probability Distribution
    plt.subplot(2, 3, 3)
    fraud_probs = y_prob[y_true == 1]
    normal_probs = y_prob[y_true == 0]
    
    plt.hist(normal_probs, bins=50, alpha=0.7, label='Normal', density=True)
    plt.hist(fraud_probs, bins=50, alpha=0.7, label='Fraud', density=True)
    plt.xlabel('Fraud Probability')
    plt.ylabel('Density')
    plt.title('Probability Distribution')
    plt.legend()
    
    # Threshold Analysis
    plt.subplot(2, 3, 4)
    thresholds = np.arange(0, 1.01, 0.01)
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_prob >= threshold).astype(int)
        if len(np.unique(y_pred_thresh)) > 1:
            precisions.append(precision_score(y_true, y_pred_thresh))
            recalls.append(recall_score(y_true, y_pred_thresh))
            f1_scores.append(f1_score(y_true, y_pred_thresh))
        else:
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)
    
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1_scores, label='F1-Score')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold Analysis')
    plt.legend()
    
    # Actual vs Predicted
    plt.subplot(2, 3, 5)
    plt.scatter(range(len(y_true)), y_true, alpha=0.5, label='Actual', s=1)
    plt.scatter(range(len(y_pred)), y_pred, alpha=0.5, label='Predicted', s=1)
    plt.xlabel('Sample Index')
    plt.ylabel('Class')
    plt.title('Actual vs Predicted')
    plt.legend()
    
    # Model Performance Summary
    plt.subplot(2, 3, 6)
    metrics = evaluate_model(y_true, y_pred)
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = plt.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title(f'Model Performance - {model_name}')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('testing_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main testing function"""
    print("Credit Card Fraud Detection - Testing")
    print("="*40)
    
    # Load the trained model
    model, scaler, model_info = load_model()
    if model is None:
        return
    
    print(f"Loaded model: {model_info['best_model_name']}")
    print(f"Uses scaling: {model_info['use_scaling']}")
    
    # Load test data (you can modify this to use different test data)
    print("\nLoading test data...")
    try:
        df = pd.read_csv('creditcard.csv')
        
        # Use the same train-test split as training for consistency
        from sklearn.model_selection import train_test_split
        X = df.drop('Class', axis=1)
        y = df['Class']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"Test data shape: {X_test.shape}")
        print(f"Test fraud cases: {y_test.sum()}")
        
    except FileNotFoundError:
        print("Error: creditcard.csv not found. Please ensure the dataset is in the same directory.")
        return
    
    # Make predictions
    print("\nMaking predictions...")
    predictions, probabilities = predict_fraud(model, scaler, model_info, X_test)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    metrics = evaluate_model(y_test, predictions)
    
    print("\nTest Results:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print("-" * 40)
    print(classification_report(y_test, predictions))
    
    # Show some example predictions
    print("\nSample Predictions:")
    print("-" * 30)
    sample_indices = np.random.choice(len(X_test), 10, replace=False)
    for i, idx in enumerate(sample_indices):
        actual = y_test.iloc[idx]
        predicted = predictions[idx]
        probability = probabilities[idx]
        status = "✓" if actual == predicted else "✗"
        print(f"Sample {i+1}: Actual={actual}, Predicted={predicted}, Probability={probability:.3f} {status}")
    
    # Find high-risk transactions
    high_risk_threshold = 0.8
    high_risk_indices = np.where(probabilities >= high_risk_threshold)[0]
    print(f"\nHigh-risk transactions (probability >= {high_risk_threshold}): {len(high_risk_indices)}")
    
    if len(high_risk_indices) > 0:
        print("Top 5 high-risk transactions:")
        top_risk_indices = high_risk_indices[np.argsort(probabilities[high_risk_indices])[-5:]]
        for i, idx in enumerate(top_risk_indices):
            actual = y_test.iloc[idx]
            probability = probabilities[idx]
            print(f"  {i+1}. Index {idx}: Actual={actual}, Probability={probability:.3f}")
    
    # Plot results
    print("\nGenerating visualization...")
    plot_results(y_test, predictions, probabilities, model_info['best_model_name'])
    print("Testing visualization saved as 'testing_results.png'")
    
    # Save predictions
    results_df = pd.DataFrame({
        'actual': y_test.reset_index(drop=True),
        'predicted': predictions,
        'probability': probabilities
    })
    results_df.to_csv('test_predictions.csv', index=False)
    print("Predictions saved to 'test_predictions.csv'")

if __name__ == "__main__":
    main()