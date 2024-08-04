# Importing necessary libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

# Load data
df = pd.read_csv("D:\\Akshat\\Python_ML_Projects\\Income Predictor\\data2.csv")

# Workclass mapping
workclass_map = {
    'Federal-gov': 'Federal Government',
    'Local-gov': 'Local Government',
    'Private': 'Private Sector',
    'Self-emp-inc': 'Self-Employed (Incorporated)',
    'Self-emp-not-inc': 'Self-Employed (Not Incorporated)',
    'State-gov': 'State Government',
    'Without-pay': 'Without Pay'
}

# Occupation mapping
occupation_map = {
    'Adm-clerical': 'Administrative/Clerical',
    'Armed-Forces': 'Armed Forces',
    'Craft-repair': 'Craft/Repair',
    'Exec-managerial': 'Executive/Managerial',
    'Farming-fishing': 'Farming/Fishing',
    'Handlers-cleaners': 'Handlers/Cleaners',
    'Machine-op-inspct': 'Machine Operators/Inspectors',
    'Other-service': 'Other Service',
    'Priv-house-serv': 'Private Household Service',
    'Prof-specialty': 'Professional Specialty',
    'Protective-serv': 'Protective Services',
    'Sales': 'Sales',
    'Tech-support': 'Technical Support',
    'Transport-moving': 'Transportation/Moving'
}

# Education level mapping
education_map = {
    1: 'Preschool',
    2: '1st to 4th Grade',
    3: '5th to 6th Grade',
    4: '7th to 8th Grade',
    5: '9th Grade',
    6: '10th Grade',
    7: '11th Grade',
    8: '12th Grade',
    9: 'High School Graduate',
    10: 'Some College',
    11: 'Vocational School',
    12: 'Academic Associate Degree',
    13: 'Bachelor\'s Degree',
    14: 'Master\'s Degree',
    15: 'Professional School',
    16: 'Doctorate'
}

# Encode categorical features with more user-friendly names
df['workclass'] = df['workclass'].map(workclass_map)
df['occupation'] = df['occupation'].map(occupation_map)
df['education.num'] = df['education.num'].map(education_map)

# Encode categorical variables
label_encoders = {}
for column in ['workclass', 'occupation', 'sex', 'education.num']:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Encode target variable
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

# Preprocess data
# Splitting data into features and labels
columns_to_keep = [
    'income', 'education.num', 'age', 'hours.per.week', 'sex', 'workclass', 'occupation'
]
df = df.loc[:, columns_to_keep]

X = df.drop('income', axis=1)
y = df['income']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train the model
xgb_clf = XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=100)
xgb_clf.fit(X_train, y_train)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Save the model and the scaler
model_path = 'xgb_model.pkl'
scaler_path = 'scaler.pkl'
# Save the label encoders
label_encoder_path = 'label_encoders.pkl'
joblib.dump(label_encoders, label_encoder_path)
joblib.dump(xgb_clf, model_path)
joblib.dump(scaler, scaler_path)

# Check if the files are created
if os.path.exists(model_path) and os.path.exists(scaler_path):
    print(f"Model saved to {model_path} and scaler saved to {scaler_path}")
else:
    print("Error: Files not saved.")

# Evaluate the model
y_pred = xgb_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Save the model and the scaler
model_path = 'xgb_model.pkl'
scaler_path = 'scaler.pkl'
# Save the label encoders
label_encoder_path = 'label_encoders.pkl'

joblib.dump(label_encoders, label_encoder_path)
joblib.dump(xgb_clf, model_path)
joblib.dump(scaler, scaler_path)