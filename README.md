# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and Explore the Dataset
2. Preprocess the Data
3. Split the Dataset
4. Train the Logistic Regression Model
5. Evaluate the Model
 

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('food_items (1).csv')

# Inspect the dataset
print("Name:Harshal Richu s")
print("Reg. No:25008177")
print("Dataset Overview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# Separate features and target
X_raw = df.iloc[:, :-1]
y_raw = df.iloc[:, -1]

# Scaling the input features
scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=123
)

# Define Logistic Regression model
l2_model = LogisticRegression(
    random_state=123,
    penalty='l2',              # L2 Regularization
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000
)

# Train the model
l2_model.fit(X_train, y_train)

# Predictions
y_pred = l2_model.predict(X_test)

# Evaluation
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Optional: Visualize Confusion Matrix
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


```

## Output:
<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/8ae5b84a-4840-4ee9-a84c-3c78043b6816" />
<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/ca84dea5-6f32-4cf9-aeae-58a066c09091" />
<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/a4c7c44e-42bf-4c62-a1ba-4ce367b0e7f4" />
<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/422995e7-4e4a-476e-9f82-7c6f2726d419" />

<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/f236c177-86af-4419-bf2e-b61cd4719e69" />








## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
