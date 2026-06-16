# 🎬 Movie Review Sentiment Analyzer

A Machine Learning web application that analyzes movie reviews and predicts whether the sentiment is **Positive** or **Negative** using Natural Language Processing (NLP) techniques.

## 🚀 Features

* Predicts whether a movie review is Positive or Negative
* Text preprocessing using NLTK
* Feature extraction using CountVectorizer
* Sentiment classification using Multinomial Naive Bayes
* Interactive web interface using Flask
* Real-time sentiment prediction

## 🛠 Technologies Used

* Python
* Flask
* Scikit-learn
* NLTK
* Pandas
* NumPy
* HTML
* CSS
* JavaScript

## 📂 Dataset

**IMDB Movie Reviews Dataset**

Source: Hugging Face Datasets

## 🤖 Machine Learning Model

**Multinomial Naive Bayes Classifier**

### Workflow

1. Load the IMDB dataset.
2. Preprocess the reviews.
3. Remove stopwords and apply stemming.
4. Convert text using CountVectorizer.
5. Train the Naive Bayes model.
6. Save the trained model using Pickle.
7. Integrate the model with Flask.

## ▶️ How to Run

```bash
git clone https://github.com/pavithra4338/Sentiment-Analyzer.git
cd Sentiment-Analyzer
pip install -r requirements.txt
python app.py
```

Then open:

```
http://127.0.0.1:5000
```

## 🌐 Live Demo

Try the deployed application here:

https://sentiment-analyzer-f5abnfzqfxy9jfacspzmjc.streamlit.app/

Deployment Platform: Streamlit Community Cloud

## 👩‍💻 Author

**Pavithra Veesam**

GitHub: https://github.com/pavithra4338
