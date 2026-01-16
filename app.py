import streamlit as st
import pickle

model = pickle.load(open('C:/Users/Arefin/Downloads/ML Model Deployment/bullying_model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('C:/Users/Arefin/Downloads/ML Model Deployment/tfidf_vectorizer.pkl', 'rb'))


st.title("Vulnarable Content Detection Model")
st.write("Enter a text to check if it contains bullying content.")

user_input = st.text_area("Enter the text")

if st.button("Predict"):
    if user_input:
       
        text_vectorized = tfidf_vectorizer.transform([user_input])

        prediction = model.predict(text_vectorized)
  
        if prediction[0]:
            st.success("Prediction: Bullying")
        else:
            st.success("Prediction: Not Bullying")
    else:
        st.error("Please enter some text to predict.")

