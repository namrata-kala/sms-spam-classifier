import streamlit as st
import pickle 
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

nltk.download("punkt", download_dir=nltk_data_path)
nltk.download("punkt_tab", download_dir=nltk_data_path)
nltk.download("stopwords", download_dir=nltk_data_path)


ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
        
    return  " ".join(y)

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    #1. preprocess
    transform_sms = transform_text(input_sms)
    #2. vectorize
    vector_input = tfidf.transform([transform_sms])
    #3. predict
    result = model.predict(vector_input)[0]
    #4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

