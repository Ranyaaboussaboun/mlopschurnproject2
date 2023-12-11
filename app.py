import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


data = pd.read_csv('final_data.csv')

with open('column_names.pkl', 'rb') as col_names_file:
    column_names = pickle.load(col_names_file)

column_names['features'] = [col for col in column_names['features'] if col not in ['customerID', 'Unnamed: 0']]

with open('model.pkl', 'rb') as model_file:
    saved_model = pickle.load(model_file)

st.subheader("Enter Customer Information:")
user_input = {}
for feature in column_names['features']:  # Use the feature columns from preprocessing
    user_input[feature] = st.text_input(f"Enter {feature}:", "")

if st.button("Predict"):
    try:
        user_input_df = pd.DataFrame([user_input])

        numerical_imputer = SimpleImputer(strategy='most_frequent')
        user_input_df[column_names['numerical']] = numerical_imputer.fit_transform(user_input_df[column_names['numerical']])

        encoder = OneHotEncoder(drop='first', sparse=False)
        user_input_encoded = pd.DataFrame(encoder.fit_transform(user_input_df[column_names['categorical']]))
        user_input_encoded.columns = encoder.get_feature_names_out(column_names['categorical'])

        user_input_processed = user_input_encoded.align(pd.DataFrame(columns=column_names['all_encoded']), join='outer', axis=1, fill_value=0)[0]

        if isinstance(saved_model, LogisticRegression):
            # Make predictions using the loaded model
            prediction = saved_model.predict(user_input_processed)

            # Display predictions
            st.subheader("Prediction:")
            st.text(prediction[0])  
        else:
            st.error("Error: Loaded model is not a Logistic Regression model.")

    except Exception as e:
        st.error(f"Error: {str(e)}")
