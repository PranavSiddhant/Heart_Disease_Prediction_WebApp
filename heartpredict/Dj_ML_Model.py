import sklearn
import numpy as np
import pandas as pd

import seaborn as sns
import os
from sklearn.metrics import accuracy_score


from sklearn.model_selection import train_test_split


# Load dataset
df= pd.read_csv('heart_statlog_cleveland_hungary_final.csv')

# renaming features to proper name
df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope','target']

# Split into training data and test data
X = df.drop(['target'],axis=1)
y = df['target']

# Create training and testing vars, It’s usually around 80/20 or 70/30.
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Now we’ll fit the model on the training data
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Make predictions on validation dataset
predictions = model.predict(X_test)


# Pickle model
pd.to_pickle(model,r'new_model.pickle')

# Unpickle model
model = pd.read_pickle(r'new_model.pickle')

# Take input from user
Patient_ID = int(input("Enter Patient_ID: "))
Patient_Age = int(input("Enter Patient_Age: "))
Patient_Gender = int(input("Enter Patient_Gender: "))
Patient_Blood_Pressure = int(input("Enter Patient_Blood_Pressure: "))
Patient_Heartrate = int(input("Enter Patient_Heartrate: "))
result = model.predict([[Patient_ID,Patient_Age,Patient_Gender,Patient_Blood_Pressure,Patient_Heartrate]])  # input must be 2D array
print(result)