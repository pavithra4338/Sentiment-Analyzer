import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download stopwords
nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Initialize preprocessing tools
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Streamlit UI
st.set_page_config(page_title="Movie Review Sentiment Analyzer")

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review and find out whether it's Positive or Negative!")

review = st.text_area("Enter your movie review:")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")