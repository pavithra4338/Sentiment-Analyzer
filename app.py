from flask import Flask, request, jsonify, send_file
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# Route for homepage
@app.route("/")
def home():
    return send_file("index.html")


# Route for prediction
@app.route("/predict", methods=["POST"])
def predict():

    data = request.json["review"]

    vector = vectorizer.transform([data])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        result = "Positive 😊"
    else:
        result = "Negative 😞"

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)