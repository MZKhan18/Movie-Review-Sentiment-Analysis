import streamlit as st
import string
import pickle
from bs4 import BeautifulSoup
import numpy as np

f = open('english_stopwords.txt')
stopwords = f.read()

def preprocessing(text):
    text = text.lower()
    soup = BeautifulSoup(text)
    text = soup.get_text()
    text = text.split()
    y = []
    for word in text:
        if word not in stopwords and word not in string.punctuation:
            y.append(word)
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Movie Reviews Sentiment Analysis")
review = st.text_area("Enter Movie Review (Avoid One Liners)")

review = preprocessing(review)

if st.button("Predict Sentiment"):
    rev_x = tfidf.transform([review])
    ans = model.predict(rev_x)[0]
    if ans ==0:
        st.header('Negative Review')
    else:
        st.header('Postive Review')

st.text("Note : for more accuracy write atleas 20 words in the reviews ")