from flask import Flask, render_template, request
import pandas as pd
import joblib
import logging

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model')

# Define the feature names
feature_names = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from the form
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        region = int(request.form['region'])

        # Create a DataFrame with the user inputs
        input_data = pd.DataFrame({'age': [age],
                                   'sex': [sex],
                                   'bmi': [bmi],
                                   'children': [children],
                                   'smoker': [smoker],
                                   'region': [region]})
        print(input_data)

        # Make the prediction
        prediction = model.predict(input_data)
        output = round(prediction[0], 2)

        return render_template('home.html', prediction_text='Estimated Expenses: {}'.format(output))
    except Exception as e:
        # Log any exceptions that occur during prediction
        app.logger.error(f"An error occurred during prediction: {e}")
        return render_template('home.html', prediction_text='Error occurred during prediction')


if __name__ == '__main__':
    app.run(debug=True)
