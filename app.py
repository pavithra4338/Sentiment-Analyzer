from flask import Flask, request, jsonify, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

app = Flask(__name__)

model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["review"]

    cleaned = clean_text(data)

    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        result = "Positive Review"
    else:
        result = "Negative Review"

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)