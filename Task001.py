# task_1_bank_marketing.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, f1_score
import shap
import matplotlib.pyplot as plt

np.random.seed(42)

# 1. Generate Synthetic Bank Marketing Dataset
n_samples = 1000
data = pd.DataFrame({
    'age': np.random.randint(18, 70, size=n_samples),
    'job': np.random.choice(['management', 'technician', 'blue-collar', 'admin.', 'services'], size=n_samples),
    'marital': np.random.choice(['married', 'single', 'divorced'], size=n_samples),
    'education': np.random.choice(['primary', 'secondary', 'tertiary', 'unknown'], size=n_samples),
    'balance': np.random.normal(1500, 3000, size=n_samples).astype(int),
    'duration': np.random.exponential(scale=250, size=n_samples).astype(int),
    'campaign': np.random.randint(1, 10, size=n_samples),
    'y': np.random.choice(['yes', 'no'], size=n_samples, p=[0.15, 0.85])
})

print("Dataset loaded. Shape:", data.shape)

# 2. Encode Categorical Features
le_dict = {}
categorical_cols = ['job', 'marital', 'education']
df = data.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

df['y'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Split features and target
X = df.drop(columns=['y'])
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train Classification Models
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 4. Evaluate Models
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# 5. Explain Predictions with SHAP
print("\nCalculating SHAP values...")
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
print("SHAP analysis complete. You can visualize using shap.summary_plot(shap_values, X_test)")