import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("Fake News Detection using NLP")

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

user_input = st.text_area("Enter the news content:")

if st.button("Check"):
    if user_input:
        vect_input = vectorizer.transform([user_input])
        prediction = model.predict(vect_input)
        if prediction[0] == 1:
            st.success("This is Real News")
        else:
            st.error("This is Fake News")
    else:
        st.warning("Please enter some text.")
