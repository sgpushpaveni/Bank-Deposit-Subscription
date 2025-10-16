import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os

import requests

def build_preprocessor():
    # Ordinal mapping for education
    education_order = ['unknown', 'primary', 'secondary', 'tertiary']
    ordinal_cols = ['education']
    nominal_cols = ['job', 'marital', 'contact', 'month', 'poutcome']
    binary_cols = ['default', 'housing', 'loan']
    numeric_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous', 'day']

    ordinal_pipe = Pipeline([
        ('ord', OrdinalEncoder(categories=[education_order], handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    nominal_pipe = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    binary_pipe = Pipeline([
        ('onehot', OneHotEncoder(drop='if_binary'))
    ])
    numeric_pipe = Pipeline([
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('ord', ordinal_pipe, ordinal_cols),
        ('nom', nominal_pipe, nominal_cols),
        ('bin', binary_pipe, binary_cols),
        ('num', numeric_pipe, numeric_cols)
    ])
    return preprocessor


st.title("Bank Term Deposit Subscription Prediction")
st.write("Enter client details to predict subscription likelihood.")

# Load model and preprocessor
data_file_path =  r"../data"
model_path =  r"../notebooks/models"
st.write("Current working directory:", os.getcwd())


model = joblib.load("XGBoost.joblib")
preprocessor = joblib.load("preprocessor.joblib")
#preprocessor = build_preprocessor()

#id,age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome

# --- Option Menu ---
with st.sidebar:
    opt_selected = option_menu(
        menu_title="Data type",
        options=["File Upload", "Single Data", "API"],
        icons=["list", "gear","gear" ],
        menu_icon="cast",
        default_index=0,
    )
if opt_selected == "File Upload":
    test_data_path = data_file_path + r"/test.csv"

    uploaded_file = st.file_uploader("Choose a test data file", type=["csv"])
    input_df = pd.DataFrame()

    if uploaded_file is not None:
        # Read the uploaded file into a pandas DataFrame.
        try:
            input_df = pd.read_csv(uploaded_file)
            st.write("### Uploaded DataFrame")
            # Display the DataFrame.
            st.dataframe(input_df)
            #input_df = pd.read_csv(test_data_path)
            input_df['balance_x_duration'] = input_df['balance'] * input_df['duration']
            input_df['age_x_education'] = input_df['age'] * input_df['education'].map({'unknown':0, 'primary':1, 'secondary':2, 'tertiary':3})

            #st.write(input_df)
            # Preprocess
            X_processed = preprocessor.transform(input_df)
            #st.write(X_processed)
            prediction = model.predict(X_processed)
            probability = model.predict_proba(X_processed)
            pred_df = pd.DataFrame(prediction, columns=["pred"])
            prob_df = pd.DataFrame(probability, columns=["Failure Rate", "Success Rate"])
            #st.dataframe(pred_df)
            #st.dataframe(prob_df)
            df_combined = pd.concat([pred_df,prob_df["Success Rate"], input_df], axis=1)
            out_df = df_combined.sort_values(by="Success Rate", ascending=False)
            st.dataframe(out_df)
        except Exception as e:
            st.error(f"Error reading the file: {e}")

elif opt_selected == "Single Data":
    # Collect inputs
    age = st.number_input("Age", 18, 99, 35)
    balance = st.number_input("Balance", -10000, 100000, 0)
    duration = st.number_input("Last Contact Duration (seconds)", 0, 5000, 100)
    education = st.selectbox("Education", ["unknown", "primary", "secondary", "tertiary"])
    job = st.selectbox("Job", ["admin.", "technician", "blue-collar", "student", "management", "entrepreneur", "self-employed", "unknown"])
    marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
    default = st.selectbox("Default?", ["yes", "no"])
    housing = st.selectbox("Housing Loan?", ["yes", "no"])
    loan = st.selectbox("Personal Loan?", ["yes", "no"])
    contact = st.selectbox("Contact", ["cellular", "telephone", "unknown"])
    month = st.selectbox("Last Contact Month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
    day = st.number_input("Last Contact Day", 1, 31, 1)
    campaign = st.number_input("Number of Contacts During Campaign", 1, 50, 1)
    pdays = st.number_input("Days Since Last Contact", -1, 999, -1)
    previous = st.number_input("Number of Previous Contacts", 0, 10, 0)
    poutcome = st.selectbox("Previous Outcome", ["unknown", "other", "failure", "success"])

    input_dict = dict(
        age=age, balance=balance, duration=duration, education=education, job=job, marital=marital,
        default=default, housing=housing, loan=loan, contact=contact, month=month, day=day,
        campaign=campaign, pdays=pdays, previous=previous, poutcome=poutcome
    )
    input_df = pd.DataFrame([input_dict])
    input_df['balance_x_duration'] = input_df['balance'] * input_df['duration']
    input_df['age_x_education'] = input_df['age'] * input_df['education'].map({'unknown':0, 'primary':1, 'secondary':2, 'tertiary':3})

    st.write(input_df)
    # Preprocess
    X_processed = preprocessor.transform(input_df)
    st.write(X_processed)
    prediction = model.predict(X_processed)[0]
    probability = model.predict_proba(X_processed)[0, 1]

    st.write(f"**Prediction:** {'Subscribed' if prediction == 1 else 'Not Subscribed'}")
    st.write(f"**Probability:** {probability:.2f}")

elif opt_selected == "API":
    # Collect inputs
    age = st.number_input("Age", 18, 99, 35)
    balance = st.number_input("Balance", -10000, 100000, 0)
    duration = st.number_input("Last Contact Duration (seconds)", 0, 5000, 100)
    education = st.selectbox("Education", ["unknown", "primary", "secondary", "tertiary"])
    job = st.selectbox("Job", ["admin.", "technician", "blue-collar", "student", "management", "entrepreneur", "self-employed", "unknown"])
    marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
    default = st.selectbox("Default?", ["yes", "no"])
    housing = st.selectbox("Housing Loan?", ["yes", "no"])
    loan = st.selectbox("Personal Loan?", ["yes", "no"])
    contact = st.selectbox("Contact", ["cellular", "telephone", "unknown"])
    month = st.selectbox("Last Contact Month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
    day = st.number_input("Last Contact Day", 1, 31, 1)
    campaign = st.number_input("Number of Contacts During Campaign", 1, 50, 1)
    pdays = st.number_input("Days Since Last Contact", -1, 999, -1)
    previous = st.number_input("Number of Previous Contacts", 0, 10, 0)
    poutcome = st.selectbox("Previous Outcome", ["unknown", "other", "failure", "success"])

    st.button("Predict")

    

    #api_url = "http://127.0.0.1:8000/predict"  
    #api_url = "http://bank-deposit-subscription-production.up.railway.app:8000/predict"
    api_url = "https://bank-term-api-1022655012071.us-central1.run.app/predict"
    data = {
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "balance": balance, "housing": housing, "loan": loan,
        "contact": contact, "day": day, "month": month, "duration": duration,
        "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome
    }


    response = requests.post(api_url, json=data)
    if response.status_code == 200:
        result = response.json()
        st.write(f"**Prediction:** {'Subscribed' if result['prediction']==1 else 'Not Subscribed'}")
        st.write(f"**Probability:** {result['probability']:.2f}")
    else:
        st.error("API Error: " + str(response.status_code))

#     {
#   "age": 0,
#   "job": "string",
#   "marital": "string",
#   "education": "string",
#   "default": "string",
#   "balance": 0,
#   "housing": "string",
#   "loan": "string",
#   "contact": "string",
#   "day": 0,
#   "month": "string",
#   "duration": 0,
#   "campaign": 0,
#   "pdays": 0,
#   "previous": 0,
#   "poutcome": "string"
# }