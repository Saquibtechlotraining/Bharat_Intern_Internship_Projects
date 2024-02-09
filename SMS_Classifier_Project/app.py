import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()  # Converting text to lowercase
    text = nltk.word_tokenize(text)  # Doing tokenize to break down text into small words of list

    y = []
    for i in text:
        if i.isalnum():  # Removing special characters
            y.append(i)

    text = y[:]  # Assign text to y
    y.clear()  # Clear the y

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:  # Removing stopwords and punctuation
            y.append(i)

    text = y[:]  # Again, Assign text to y
    y.clear()  # And clear the y

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)  # Joining of list of data into a single string

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("SMS Spam Classifier")
input_sms = st.text_area('Enter the message')

if st.button('Predict'):


    # preprocess
    transformed_sms = transform_text(input_sms)
    # vectorize
    vector_input = tfidf.transform([transformed_sms])
    # predict
    result = model.predict(vector_input)[0]
    print(result)
    # Display
    if result == 1:
        print(result)
        st.header('Spam')
    else:
        st.header('Not Spam')

# Adding acknowledgment
st.text("Developed by Saquib Ahmad")
