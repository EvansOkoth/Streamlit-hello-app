import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit

#Load a saved model
loaded_pickle_model = pickle.load(open("C:/Users/user/Desktop/Just try/Hevy_random_classifier_model_1.pkl", "rb"))

#creating a function for prediction
def Customer_churn_prediction(input_data):

    # Changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_pickle_model.predict(input_data_reshaped)
    prediction

    if (prediction[0] == 0):
        return 'The customer will not churn'
    else:
        return 'The customer will churn'

def main():

    #Giving a title

    streamlit.title('Customer Churn Web App')

    #getting the imput data from the user

    CreditScore = streamlit.text_input('Your credit score')
    Geography = streamlit.text_input('Your Geographical area')
    Gender = streamlit.text_input('Your Gender')
    Age = streamlit.text_input('Your Age')
    Tenure = streamlit.text_input('Tenure')
    Balance = streamlit.text_input('Balance')
    NumOfProducts = streamlit.text_input('Number of products')
    HasCrCard = streamlit.text_input('Do you have a card')
    IsActiveMember = streamlit.text_input('Are you an active member')
    EstimatedSalary = streamlit.text_input('Estimate your salary')

    #code for prediction

    predictions = ''

    #creating a button for prediction

    if streamlit.button('Customer Churn Result'):
        predictions = Customer_churn_prediction([CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary])
    streamlit.success(predictions)

if __name__ == '__main__':
    main()

