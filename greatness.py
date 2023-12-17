import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
data = pd.read_csv('rhetorical_questions.csv', encoding='latin1')

# Print column names
print("Column Names:", data.columns)

# Remove leading and trailing spaces from column names
data.columns = data.columns.str.strip()

# Separate features and labels
X = data['Question']
y = data['Label']

# Vectorize the text
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)

# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model performance
accuracy = classifier.score(X_test, y_test)
print('Accuracy:', accuracy)

# Save the model and vectorizer
model_filename = 'rhetorical_model.joblib'
vectorizer_filename = 'tfidf_vectorizer.joblib'

joblib.dump(classifier, model_filename)
joblib.dump(vectorizer, vectorizer_filename)

print(f"Trained model saved to {model_filename}")
print(f"Vectorizer saved to {vectorizer_filename}")
