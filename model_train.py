import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler

from sklearn.linear_model import LogisticRegression


def train_data(data):
    # Dropping Loan_ID column
    data.drop(['Loan_ID'], axis=1, inplace=True)

    # filling null values

    data.Gender.fillna(data['Gender'].mode()[0], inplace=True)

    data.Married.fillna(data['Married'].mode()[0], inplace=True)

    data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)

    data.Self_Employed.fillna(data['Self_Employed'].mode()[0], inplace=True)

    data['LoanAmount'].fillna(data['LoanAmount'].mean(), inplace=True)  # Mean

    data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean(), inplace=True)

    data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)  # Mode

    X = data.drop(['Loan_Status'], axis=1)
    Y = data['Loan_Status']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)

    # Encoding categorical data
    # Encoding the Independent Variable

    columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed']
    labelencoder_X = LabelEncoder()
    for i, col in enumerate(columns):
        X_train[col] = labelencoder_X.fit_transform(X_train.iloc[:, i])

    X_train["Property_Area"] = labelencoder_X.fit_transform(X_train.iloc[:, 10])

    # Encoding the Dependent Variable
    labelencoder_y = LabelEncoder()
    Y_train = labelencoder_y.fit_transform(Y_train)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_train, Y_train)
    return model

def input_transform(input):
    input = pd.DataFrame(input,index=[0],columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area'])
    labelencoder_X = LabelEncoder()
    columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed']
    for i, col in enumerate(columns):
        input[col] = labelencoder_X.fit_transform(input.iloc[:, i])

    input["Property_Area"] = labelencoder_X.fit_transform(input.iloc[:, 10])

    return input
