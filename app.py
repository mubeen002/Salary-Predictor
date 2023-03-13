import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request

# Load the data from the CSV file
data = pd.read_csv("Salary_Data.csv")

# Split the data into features (X) and target (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression object and fit the model
logreg = LogisticRegression(max_iter=5000)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
accuracy = logreg.score(X_test, y_test)

# Create a Flask app
app = Flask(__name__, template_folder=os.getcwd())

# Define the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input from the web form
    input_data = [float(x) for x in request.form.values()]

    # Make a prediction for the input data
    score = logreg.predict([input_data])[0]

    # Render the prediction and score to the results page
    return render_template('results.html', score=score)

# Run the app
if __name__ == '__main__':
    app.run(debug=False)
