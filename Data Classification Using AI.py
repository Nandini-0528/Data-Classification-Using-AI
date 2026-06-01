# Data Classification Using AI

# Import Libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
print("Loading Iris Dataset...\n")
iris = load_iris()

# Create DataFrame
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add Target Column
data['target'] = iris.target

# Display First 5 Rows
print("First 5 Rows of Dataset:\n")
print(data.head())

# Understand Dataset
print("\nDataset Information:\n")
print(data.info())
print("\nDataset Description:\n")
print(data.describe())

# Split Features and Labels
X = iris.data          # Features
y = iris.target        # Labels

# Split Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("\nTraining Data Size :", len(X_train))
print("Testing Data Size  :", len(X_test))

# Apply Classification Algorithm
print("\nTraining Decision Tree Classifier...\n")
model = DecisionTreeClassifier()

# Train Model
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Check Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy :", accuracy * 100, "%")

# Display Results
print("\nPredicted Values:\n")
print(y_pred)
print("\nActual Values:\n")
print(y_test)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# End