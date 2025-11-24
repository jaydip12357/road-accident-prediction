# ========================================
# COMPLETE ROAD ACCIDENT PREDICTION PIPELINE
# Ultra-fast version with Logistic Regression
# ========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import mlflow
import mlflow.sklearn
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🚀 ROAD ACCIDENT SEVERITY PREDICTION")
print("=" * 60)

# ===== LOAD AND FILTER DATA =====
print("\n📁 Loading data...")
df = pd.read_csv('content/road.csv')
print(f"✅ Initial data: {df.shape[0]:,} rows")

# Take only recent 50% of data
if 'collision_year' in df.columns:
    median_year = df['collision_year'].median()
    df = df[df['collision_year'] >= median_year]
    print(f"✅ Using data from {int(median_year)} onwards: {df.shape[0]:,} rows")
elif 'date' in df.columns:
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    median_date = df['date_parsed'].median()
    df = df[df['date_parsed'] >= median_date]
    print(f"✅ Using recent 50% of data: {df.shape[0]:,} rows")

# Sample if still too large
if df.shape[0] > 200000:
    df = df.sample(n=200000, random_state=42)
    print(f"✅ Sampled to: {df.shape[0]:,} rows")

# ===== FEATURE ENGINEERING =====
print("\n🔧 Feature engineering...")
df_model = df.copy()

if 'time' in df_model.columns:
    df_model['hour'] = pd.to_datetime(df_model['time'], format='%H:%M', errors='coerce').dt.hour
    df_model['hour'] = df_model['hour'].fillna(df_model['hour'].median())

if 'date' in df_model.columns:
    df_model['date'] = pd.to_datetime(df_model['date'], errors='coerce')
    df_model['month'] = df_model['date'].dt.month

# Select features
selected_features = [
    'speed_limit', 'number_of_vehicles', 'number_of_casualties',
    'hour', 'light_conditions', 'weather_conditions',
    'road_surface_conditions', 'urban_or_rural_area'
]
selected_features = [col for col in selected_features if col in df_model.columns]
df_model = df_model[selected_features + ['collision_severity']].dropna(subset=['collision_severity'])

# ===== PREPROCESSING =====
# Handle missing values
numerical_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()
for col in numerical_cols:
    if df_model[col].isnull().sum() > 0:
        df_model[col].fillna(df_model[col].median(), inplace=True)

categorical_cols = df_model.select_dtypes(include=['object']).columns.tolist()
if 'collision_severity' in categorical_cols:
    categorical_cols.remove('collision_severity')
for col in categorical_cols:
    if df_model[col].isnull().sum() > 0:
        df_model[col].fillna(df_model[col].mode()[0], inplace=True)

# Encode target
target_mapping = {'Slight': 0, 'Serious': 1, 'Fatal': 2}
df_model['target'] = df_model['collision_severity'].map(target_mapping)
if df_model['target'].isnull().any():
    le_target = LabelEncoder()
    df_model['target'] = le_target.fit_transform(df_model['collision_severity'])

print(f"✅ Target distribution:\n{df_model['target'].value_counts().sort_index()}")

# Encode categoricals
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le

# Prepare X and y
X = df_model.drop(['collision_severity', 'target'], axis=1)
y = df_model['target']
print(f"\n📊 Final dataset: {X.shape[0]:,} samples, {X.shape[1]} features")

# ===== TRAIN-TEST SPLIT =====
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
print(f"✅ Train: {X_train.shape[0]:,}, Val: {X_val.shape[0]:,}, Test: {X_test.shape[0]:,}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ===== TRAIN MODEL =====
print("\n🤖 Training Logistic Regression...")
mlflow.set_experiment("road-accident-severity")

with mlflow.start_run(run_name="logistic_regression_v1"):
    params = {
        'C': 1.0, 'max_iter': 200, 'class_weight': 'balanced',
        'random_state': 42, 'solver': 'lbfgs'
    }
    mlflow.log_params(params)

    lr_model = LogisticRegression(**params, n_jobs=-1)
    lr_model.fit(X_train_scaled, y_train)

    y_test_pred = lr_model.predict(X_test_scaled)

    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')

    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_f1_macro", test_f1)
    mlflow.sklearn.log_model(lr_model, "model")

    print(f"\n✅ Test Accuracy: {test_acc:.4f}")
    print(f"✅ Test F1 Score: {test_f1:.4f}")

# ===== EVALUATION =====
print("\n📈 Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Slight', 'Serious', 'Fatal']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Slight', 'Serious', 'Fatal'],
            yticklabels=['Slight', 'Serious', 'Fatal'])
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# Feature coefficients
avg_coef = np.abs(lr_model.coef_).mean(axis=0)
feature_imp = pd.DataFrame({
    'feature': X.columns,
    'coefficient': avg_coef
}).sort_values('coefficient', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_imp, x='coefficient', y='feature', palette='viridis')
plt.title('Feature Coefficients', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n📊 Feature Importance:")
print(feature_imp.to_string(index=False))

# ===== SAVE MODELS =====
print("\n💾 Saving models...")
with open('models2/accident_severity_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
with open('models2/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models2/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
with open('models2/feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
with open('models2/categorical_cols.pkl', 'wb') as f:
    pickle.dump(categorical_cols, f)
print("✅ All models saved!")

# ===== TEST PREDICTION =====
def predict_severity(input_data):
    input_df = pd.DataFrame([input_data])
    for col in categorical_cols:
        if col in input_df.columns and col in label_encoders:
            try:
                input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
            except:
                input_df[col] = 0
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[X.columns]
    input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    input_scaled = scaler.transform(input_df)
    pred = lr_model.predict(input_scaled)[0]
    probs = lr_model.predict_proba(input_scaled)[0]
    return {0: 'Slight', 1: 'Serious', 2: 'Fatal'}[pred], probs

print("\n" + "=" * 60)
print("TEST PREDICTIONS")
print("=" * 60)

# Test 1: Moderate risk
test1 = {
    'speed_limit': 30, 'number_of_vehicles': 2, 'number_of_casualties': 1,
    'hour': 17, 'light_conditions': 1, 'weather_conditions': 1,
    'road_surface_conditions': 1, 'urban_or_rural_area': 1
}
pred1, probs1 = predict_severity(test1)
print(f"\n🎯 Test 1 - Moderate Risk:")
print(f"   Prediction: {pred1}")
print(f"   Probabilities: Slight={probs1[0]:.1%}, Serious={probs1[1]:.1%}, Fatal={probs1[2]:.1%}")

# Test 2: High risk
test2 = {
    'speed_limit': 70, 'number_of_vehicles': 3, 'number_of_casualties': 2,
    'hour': 23, 'light_conditions': 4, 'weather_conditions': 2,
    'road_surface_conditions': 2, 'urban_or_rural_area': 0
}
pred2, probs2 = predict_severity(test2)
print(f"\n🎯 Test 2 - High Risk:")
print(f"   Prediction: {pred2}")
print(f"   Probabilities: Slight={probs2[0]:.1%}, Serious={probs2[1]:.1%}, Fatal={probs2[2]:.1%}")

# Test 3: Low risk
test3 = {
    'speed_limit': 20, 'number_of_vehicles': 1, 'number_of_casualties': 1,
    'hour': 10, 'light_conditions': 1, 'weather_conditions': 1,
    'road_surface_conditions': 1, 'urban_or_rural_area': 1
}
pred3, probs3 = predict_severity(test3)
print(f"\n🎯 Test 3 - Low Risk:")
print(f"   Prediction: {pred3}")
print(f"   Probabilities: Slight={probs3[0]:.1%}, Serious={probs3[1]:.1%}, Fatal={probs3[2]:.1%}")