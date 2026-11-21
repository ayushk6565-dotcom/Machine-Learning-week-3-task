import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv("Credit card fraud/creditcard.csv")
df = pd.DataFrame(data)
df = df.drop_duplicates()
X = df.drop("Class",axis=1)
y = df["Class"]
print("Feature:",X.shape)
print("Target:",y.shape)

scaler = StandardScaler()
X[["Time","Amount"]]= scaler.fit_transform(X[["Time","Amount"]])
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,train_size=0.8,stratify=y)
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train,y_train)
print("Model training complete")

prediction = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:,1]
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

print("Accuracy:", accuracy_score(y_test, prediction))

print("\nClassification Report:")
print(classification_report(y_test, prediction))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, prediction))