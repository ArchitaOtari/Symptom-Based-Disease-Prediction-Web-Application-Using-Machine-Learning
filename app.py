import streamlit as st

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB





# Load dataset
data = pd.read_csv('trainmodel.csv')

# Separate features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Identify categorical features
categorical_features = X.select_dtypes(include=['object']).columns

# Create a ColumnTransformer for one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'
)

# Fit and transform the features
X_transformed = preprocessor.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and combine models
# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_output = knn.predict_proba(X_train)

# Train Random Forest Using KNN Output
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_input = np.hstack((X_train, knn_output))
rf.fit(rf_input, y_train)
rf_output = rf.predict_proba(rf_input)

# Train Decision Tree Using Random Forest Output
dt = DecisionTreeClassifier(random_state=42)
dt_input = np.hstack((X_train, rf_output))
dt.fit(dt_input, y_train)
dt_output = dt.predict_proba(dt_input)

# Train Naive Bayes Using Decision Tree Output
nb = GaussianNB()
nb_input = np.hstack((X_train, dt_output))
nb.fit(nb_input, y_train)

# Function for final prediction
def predict_disease(input_data):
    input_data = scaler.transform(preprocessor.transform(input_data))
    knn_test_output = knn.predict_proba(input_data)
    rf_test_input = np.hstack((input_data, knn_test_output))
    rf_test_output = rf.predict_proba(rf_test_input)
    dt_test_input = np.hstack((input_data, rf_test_output))
    dt_test_output = dt.predict_proba(dt_test_input)
    nb_test_input = np.hstack((input_data, dt_test_output))
    final_prediction = nb.predict(nb_test_input)
    return final_prediction[0]

# Streamlit UI
st.markdown(
    """
    <style>
        body { 
        background: linear-gradient(135deg, #1e1e1e 30%, #3c64c7); 
        font-family: 'Poppins', sans-serif;
        font-size: 13px;
        color: #333; }

        .reportview-container {
             background: #f0f2f6;
                }
        .sidebar .sidebar-content { 
            background: #ffffff;
              }
        .stButton button { 
        background-color: #4CAF50;
        color: white; 
        border: none;
        padding: 10px 20px;
        text-align: center; 
        text-decoration: none; 
        display: inline-block;
        font-size: 16px; 
        margin: 4px 2px; 
        cursor: pointer; 
        border-radius: 12px; }

        .stButton button:hover 
        { background-color: #45a049;
          }
    </style>
    """, unsafe_allow_html=True
)

st.title("Symptom-Based Disease Prediction")
st.write("Select the symptoms from the options below:")

# Assuming the symptom names are the feature names in your dataset
symptoms = X.columns
selected_symptoms = st.multiselect('Symptoms', symptoms)

# Predict button
if st.button('Predict'):
    if selected_symptoms:
        # Create input for prediction
        input_data = np.zeros(len(symptoms))
        symptom_indices = [symptoms.get_loc(symptom) for symptom in selected_symptoms]
        input_data[symptom_indices] = 1
        input_data = input_data.reshape(1, -1)

        # Make predictions
        final_prediction = predict_disease(input_data)
        st.write(f"The predicted disease is: **{final_prediction}**")
    else:
        st.write("Please select at least one symptom.")
