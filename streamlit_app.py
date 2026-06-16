import streamlit as st
import pickle
import re
import base64
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords
nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text preprocessing setup
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Clean text
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Convert image to base64
def get_base64(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()

bg_image = get_base64("static/cinema-bg.jpg")

# Page settings
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    layout="centered"
)

# Custom CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
    }}

    .main-card {{
        background: rgba(0, 0, 0, 0.80);
        padding: 25px;
        border-radius: 25px;
        margin-top: 15px;
        text-align: center;
    }}

    .title {{
        color: red;
        font-size: 36px;
        font-weight: bold;
    }}

    .subtitle {{
        color: lightgray;
        font-size: 18px;
        margin-bottom: 15px;
    }}

    .footer {{
        color: lightgray;
        text-align: center;
        margin-top: 30px;
    }}
    
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 1rem;
    }}
    
    header {{
         visibility: hidden;
    }}
    
    footer {{
        visibility: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Card Start
st.markdown(
    """
    <div class="main-card">
        <div class="title">🎬 Movie Review Sentiment Analyzer</div>
        <div class="subtitle">
            Discover what people feel about movies using Machine Learning
        </div>
    """,
    unsafe_allow_html=True
)

# Input
review = st.text_area(
    "",
    height=120,
    placeholder="Enter your movie review here..."
)

# Analyze button
if st.button("Analyze Sentiment", use_container_width=True):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:
        cleaned = clean_text(review)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")

# Footer
st.markdown(
    """
        <div class="footer">
            Powered by Flask • Naive Bayes • IMDB Dataset
        </div>
    </div>
    """,
    unsafe_allow_html=True
)