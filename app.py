# -*- coding: utf-8 -*-

import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('pickles/model.pkl', 'rb'))
le_gender = pickle.load(open('pickles/le_gender.pkl', 'rb'))
le_married = pickle.load(open('pickles/le_married.pkl', 'rb'))
le_dependents = pickle.load(open('pickles/le_dependents.pkl', 'rb'))
le_education = pickle.load(open('pickles/le_education.pkl', 'rb'))
le_self_employed = pickle.load(open('pickles/le_self_employed.pkl', 'rb'))
le_property_area = pickle.load(open('pickles/le_property_area.pkl', 'rb'))
le_loan_status = pickle.load(open('pickles/le_loan_status.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    test_data = []
    features = [x for x in request.form.values()]
    test_data.append(le_gender.transform([features[0]])[0])
    test_data.append(le_married.transform([features[1]])[0])
    test_data.append(le_dependents.transform([features[2]])[0])
    test_data.append(le_education.transform([features[3]])[0])
    test_data.append(le_self_employed.transform([features[4]])[0])
    test_data.append(le_property_area.transform([features[5]])[0])
    test_data.append(float(features[6]))
    test_data.append(float(features[7]))
    test_data.append(float(features[8]))

    prediction = model.predict(np.array(test_data).reshape(1, -1))
    output = le_loan_status.inverse_transform(prediction)[0]
    if output == "Y":
        status = "Approved"
    else:
        status = "Not Approved"

    return render_template('index.html', prediction_text='Your Loan Status is: {}'.format(status))


if __name__ == "__main__":
    app.run(debug=True)
