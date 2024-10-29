import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Load the pre-trained vectorizer and model
tfidf = pickle.load(open('Vectorizer.pkl', 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()



# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercase the text
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize the text
    words = [lemmatizer.lemmatize(word, pos='v') for word in tokens if word not in set(stopwords.words('english'))]
    processed_text = ' '.join(words)
    return processed_text


# Set up the Streamlit app title
st.title("SMS Spam Classifier")

# Input text for SMS
input_sms = st.text_area("Enter the message")


# Prediction only when the "Predict" button is clicked
if st.button('Predict'):
    # Preprocess the input text
    processed_text = preprocess_text(input_sms)

    # Vectorize the processed text
    input_vector = tfidf.transform([processed_text])

    # Model prediction
    result = model.predict(input_vector)[0]

    # Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
