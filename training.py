# Credit Card Fraud Detection - Training Script

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('creditcard.csv')

# Basic dataset exploration
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Fraud cases: {df['Class'].value_counts()}")

# Data preprocessing
print("\nPreprocessing data...")

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features (important for some algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Train and evaluate models
results = {}
trained_models = {}

print("\nTraining models...")
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Use scaled data for Logistic Regression, original for tree-based models
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    trained_models[name] = model
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

# Compare models
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

comparison_df = pd.DataFrame(results).T
print(comparison_df.round(4))

# Find best model based on F1-score (good for imbalanced datasets)
best_model_name = max(results, key=lambda x: results[x]['f1_score'])
best_model = trained_models[best_model_name]

print(f"\nBest model: {best_model_name}")
print(f"F1-Score: {results[best_model_name]['f1_score']:.4f}")

# Save the best model and scaler
print(f"\nSaving best model ({best_model_name}) and scaler...")
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save model info
model_info = {
    'best_model_name': best_model_name,
    'use_scaling': best_model_name == 'Logistic Regression',
    'results': results
}

with open('model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print("Model training completed!")
print("Files saved: model.pkl, scaler.pkl, model_info.pkl")

# Visualize results
plt.figure(figsize=(12, 8))

# Plot 1: Model comparison
plt.subplot(2, 2, 1)
metrics = ['accuracy', 'precision', 'recall', 'f1_score']
model_names = list(results.keys())
x = np.arange(len(model_names))
width = 0.2

for i, metric in enumerate(metrics):
    values = [results[model][metric] for model in model_names]
    plt.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x + width*1.5, model_names, rotation=45)
plt.legend()
plt.tight_layout()

# Plot 2: Confusion matrix for best model
plt.subplot(2, 2, 2)
if best_model_name == 'Logistic Regression':
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# Plot 3: Class distribution
plt.subplot(2, 2, 3)
df['Class'].value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class (0=Normal, 1=Fraud)')
plt.ylabel('Count')
plt.xticks(rotation=0)

# Plot 4: Feature importance (for tree-based models)
plt.subplot(2, 2, 4)
if best_model_name in ['Random Forest', 'Decision Tree']:
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title(f'Top 10 Feature Importance - {best_model_name}')
else:
    plt.text(0.5, 0.5, 'Feature importance not available\nfor Logistic Regression', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Feature Importance')

plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nTraining visualization saved as 'training_results.png'")