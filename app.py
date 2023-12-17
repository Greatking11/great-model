from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

app = Flask(__name__)

# Load the pre-trained model
model_filename = 'rhetorical_model.joblib'
classifier = joblib.load(model_filename)

# Load the vectorizer
vectorizer_filename = 'tfidf_vectorizer.joblib'
vectorizer = joblib.load(vectorizer_filename)

@app.route('/predict', methods=['POST'])
def predict():
    # Receive input data from the request
    data = request.get_json()

    # Extract the text input from the request
    input_text = data['input']

    # Vectorize the text
    input_vec = vectorizer.transform([input_text])

    # Make predictions
    prediction = classifier.predict(input_vec)

    # Return the prediction as a string
    return jsonify({'prediction': str(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
