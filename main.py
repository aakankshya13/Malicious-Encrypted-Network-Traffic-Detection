import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# ================================
# Load NSL-KDD Dataset
# ================================

train_data = pd.read_csv("KDDTrain+.txt", header=None)
test_data = pd.read_csv("KDDTest+.txt", header=None)

data = pd.concat([train_data, test_data])

# ================================
# Split features and labels
# ================================

X = data.iloc[:, :-2]
y = data.iloc[:, -2]

# ================================
# Binary Classification
# ================================

y = y.apply(lambda x: 0 if x == 'normal' else 1)

print("\nClass distribution:\n")
print(y.value_counts())

# ================================
# Efficient Encoding
# ================================

for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

X = X.astype('float32')

# ================================
# Train-Test Split
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ================================
# Model
# ================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# ================================
# Prediction
# ================================

predictions = model.predict(X_test)

# 🔥 IMPORTANT (NEW)
proba = model.predict_proba(X_test)

# ================================
# Evaluation
# ================================

accuracy = accuracy_score(y_test, predictions)

print("\nModel trained successfully!")
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, predictions, zero_division=0))

# ================================
# OUTPUT TABLES
# ================================

accuracy_table = pd.DataFrame({
    "Metric": ["Accuracy"],
    "Value (%)": [round(accuracy * 100, 2)]
})

print("\n===== Accuracy Table =====\n")
print(accuracy_table)

report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()

report_df = report_df.loc[['0', '1'], ['precision', 'recall', 'f1-score']]
report_df.index = ["Normal (0)", "Attack (1)"]

print("\n===== Precision, Recall, F1 Table =====\n")
print(report_df)

# ================================
# Behaviour Analysis (IMPROVED)
# ================================

print("\nEnhanced Behaviour Analysis (Confidence-Based):\n")

for i in range(min(50, len(X_test))):

    if predictions[i] == 0:
        print(f"Traffic {i}: Normal → allowed\n")

    else:
        confidence = proba[i][1]

        print(f"Traffic {i}: Suspicious detected")
        print(f"  → Model confidence: {confidence:.2f}")

        if confidence > 0.7:
            print("  → High confidence → Confirmed malicious\n")
        else:
            print("  → Low confidence → Likely benign\n")

# ================================
# FINAL COUNT AFTER BEHAVIOUR
# ================================

final_benign = 0
final_malicious = 0

for i in range(len(X_test)):

    if predictions[i] == 0:
        final_benign += 1

    else:
        confidence = proba[i][1]

        if confidence > 0.7:
            final_malicious += 1
        else:
            final_benign += 1

print("\n===== FINAL RESULT AFTER BEHAVIOUR ANALYSIS =====\n")
print("Benign (Normal):", final_benign)
print("Malicious:", final_malicious)
print("Total:", final_benign + final_malicious)

# ================================
# Confusion Matrix
# ================================

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ================================
# Feature Importance
# ================================

importance = model.feature_importances_

feature_importance = pd.Series(importance, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10,5))
feature_importance[:10].plot(kind='bar')

plt.title("Top Important Features")
plt.xlabel("Features")
plt.ylabel("Importance Score")

plt.show()

# ================================
# Prediction Summary
# ================================

prediction_counts = pd.Series(predictions).value_counts()

print("\n===== Overall Prediction Summary =====\n")
print("Normal (0):", prediction_counts.get(0, 0))
print("Attack (1):", prediction_counts.get(1, 0))
print("Total Predictions:", len(predictions))