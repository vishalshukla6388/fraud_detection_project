import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve
import shap

# Dummy data generation for demonstration
np.random.seed(42)
X = pd.DataFrame(np.random.randn(1000, 5), columns=['V1', 'V2', 'V3', 'V4', 'Amount'])
y = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, preds))

# SHAP Explanations
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
print("SHAP values calculated successfully!")