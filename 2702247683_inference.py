# Nama : Cresenshia Hillary Benida
# NIM : 2702247683
# Dataset A (Loan)

import streamlit as st
import joblib
import numpy as np

# Load the machine learning model
model = joblib.load('xgb_class.pkl')

# Encoding untuk input categorical features
def encode_input(data):
    # Binary encoding
    gender_map = {'male': 1, 'female': 0}
    previous_defaults_map = {'Yes': 1, 'No': 0}
    data['person_gender'] = gender_map.get(data['person_gender'], 0)
    data['previous_loan_defaults_on_file'] = previous_defaults_map.get(data['previous_loan_defaults_on_file'], 0)
    
    # Label encoding
    education_map = {
        'High School': 0,
        'Associate': 1,
        'Bachelor': 2,
        'Master': 3,
        'Doctorate': 4
    }
    data['person_education'] = education_map.get(data['person_education'], 0)
    
    # One-hot encoding
    home_ownership_categories = ['MORTGAGE', 'RENT', 'OWN', 'OTHER']
    loan_intent_categories = ['PERSONAL', 'VENTURE', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'DEBTCONSOLIDATION']
    for cat in home_ownership_categories:
        data[f'person_home_ownership_{cat}'] = 1 if data['person_home_ownership'] == cat else 0
    for cat in loan_intent_categories:
        data[f'loan_intent_{cat}'] = 1 if data['loan_intent'] == cat else 0
    del data['person_home_ownership']
    del data['loan_intent']
    
    return data

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

def main():
    st.title('Loan Problem Machine Learning Model Deployment')

    # Input user per fitur
    person_age = st.number_input('Age', min_value=0, value=25)
    person_gender = st.selectbox('Gender', ['male', 'female'])
    person_education = st.selectbox('Education Level', ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'])
    person_income = st.number_input('Income', min_value=0, value=30000)
    person_emp_exp = st.number_input('Employment Experience (years)', min_value=0, value=1)
    person_home_ownership = st.selectbox('Home Ownership', ['MORTGAGE', 'RENT', 'OWN', 'OTHER'])
    loan_amnt = st.number_input('Loan Amount', min_value=0, value=15000)
    loan_intent = st.selectbox('Loan Intent', ['PERSONAL', 'VENTURE', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'DEBTCONSOLIDATION'])
    loan_int_rate = st.number_input('Loan Interest Rate (%)', value=10.0)
    loan_percent_income = st.number_input('Loan Percent of Income', value=0.1)
    cb_person_cred_hist_length = st.number_input('Credit History Length (years)', min_value=0, value=3)
    credit_score = st.number_input('Credit Score', min_value=0, value=600)
    previous_loan_defaults_on_file = st.selectbox('Previous Loan Defaults', ['Yes', 'No'])

    if st.button('Make Prediction'):
        # Input dari user 
        features = {
            'person_age': person_age,
            'person_gender': person_gender,
            'person_education': person_education,
            'person_income': person_income,
            'person_emp_exp': person_emp_exp,
            'person_home_ownership': person_home_ownership,
            'loan_amnt': loan_amnt,
            'loan_intent': loan_intent,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_percent_income,
            'cb_person_cred_hist_length': cb_person_cred_hist_length,
            'credit_score': credit_score,
            'previous_loan_defaults_on_file': previous_loan_defaults_on_file
        }

        # Encode fitur kategorikal dan one-hot
        encoded_features = encode_input(features)

        # Urutkan fitur sesuai urutan yang dipakai oleh model
        feature_order = [
            'person_age',
            'person_gender',
            'person_education',
            'person_income',
            'person_emp_exp',
            'loan_amnt',
            'loan_int_rate',
            'loan_percent_income',
            'cb_person_cred_hist_length',
            'credit_score',
            'previous_loan_defaults_on_file',
            'person_home_ownership_MORTGAGE',
            'person_home_ownership_OTHER',
            'person_home_ownership_OWN',
            'person_home_ownership_RENT',
            'loan_intent_DEBTCONSOLIDATION',
            'loan_intent_EDUCATION',
            'loan_intent_HOMEIMPROVEMENT',
            'loan_intent_MEDICAL',
            'loan_intent_PERSONAL',
            'loan_intent_VENTURE'
        ]

        # Encode nilai features dari user sebelum dimasukan ke dalam model
        features = [encoded_features[feat] for feat in feature_order]

        # Prediksi
        result = make_prediction(features)

        st.success(f'The prediction of loan status is: {"Approved" if result == 1 else "Rejected"}')

if __name__ == '__main__':
    main()

