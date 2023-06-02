from flask import Flask, render_template, request
import pandas as pd

from model_train import train_data,input_transform

model = None

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/train', methods=['POST'])
def train_model():
    global model
    # Get the uploaded file
    file = request.files['file']

    # Read the file as a pandas DataFrame
    data = pd.read_csv(file)

    # Perform preprocessing and train the model (same steps as before)
    model = train_data(data)

    return render_template('prediction.html')


@app.route('/predict', methods=['POST'])
def predict_loan_approval():
    # Get user input from the form

    input_data = {
        'Gender': request.form['gender'],
        'Married': request.form['married'],
        'Dependents': request.form['dependents'],
        'Education': request.form['education'],
        'Self_Employed': request.form['self_employed'],
        'ApplicantIncome': float(request.form['applicant_income']),
        'CoapplicantIncome': float(request.form['coapplicant_income']),
        'LoanAmount': float(request.form['loan_amount']),
        'Loan_Amount_Term': float(request.form['loan_amount_term']),
        'Credit_History': float(request.form['credit_history']),
        'Property_Area': request.form['property_area']
    }

    # Perform preprocessing on the input data (same steps as before)
    input = input_transform(input_data)

    # Use the trained model to make predictions
    prediction = model.predict(input)[0]

    # Prepare the prediction result message
    result_message = 'Approved' if prediction == 1 else 'Not Approved'

    return render_template('prediction.html', prediction=result_message)

if __name__ == '__main__':
    app.run(debug=True)
