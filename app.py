from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
with open('logistic_regression_model.pkl', 'rb') as f:
    theta = pickle.load(f)

# Normalize the features (important for Gradient Descent)
def normalize(X):
    return (X - X.mean()) / X.std()

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Prediction function
def predict(X, theta):
    X = np.c_[np.ones(X.shape[0]), X]  # Add intercept term
    X = np.array(X)
    probability = sigmoid(X.dot(theta))  # Get the probability predictions
    return probability

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_admission():
    # Get form input data
    GRE_score = float(request.form['GRE_score'])
    TOEFL_score = float(request.form['TOEFL_score'])
    university_rating = float(request.form['university_rating'])
    SOP = float(request.form['SOP'])
    LOR = float(request.form['LOR'])
    CGPA = float(request.form['CGPA'])
    research = int(request.form['research'])  # Assuming research is either 0 or 1

    # Create the input feature vector
    features = np.array([GRE_score, TOEFL_score, university_rating, SOP, LOR, CGPA, research])

    # Normalize the features (same as during model training)
    features = normalize(features)

    # Make the prediction
    probability = predict(features.reshape(1, -1), theta)

    # Determine if the admission chance is greater than 0.5
    prediction = "Admission Probablity Detected" if probability >= 0.5 else "No Admission Probability"
    
    return render_template('result.html', prediction=prediction, probability=probability[0])

if __name__ == "__main__":
    app.run(debug=True)
