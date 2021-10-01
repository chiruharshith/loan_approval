# -*- coding: utf-8 -*-
# Importing the libraries
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Importing the dataset
df = pd.read_csv('data/loan_train.csv')
df.drop('Loan_ID', axis=1, inplace=True)

cat_data = []
num_data = []

for i, c in enumerate(df.dtypes):
    if c == object:
        cat_data.append(df.iloc[:, i])
    else:
        num_data.append(df.iloc[:, i])

cat_data = pd.DataFrame(cat_data).transpose()
cat_data = cat_data.apply(lambda x: x.fillna(x.value_counts().index[0]))

num_data = pd.DataFrame(num_data).transpose()
num_data.fillna(method='bfill', inplace=True)

le_gender = LabelEncoder()
cat_data["Gender"] = le_gender.fit_transform(cat_data["Gender"])

le_married = LabelEncoder()
cat_data["Married"] = le_married.fit_transform(cat_data["Married"])

le_dependents = LabelEncoder()
cat_data["Dependents"] = le_dependents.fit_transform(cat_data["Dependents"])

le_education = LabelEncoder()
cat_data["Education"] = le_education.fit_transform(cat_data["Education"])

le_self_employed = LabelEncoder()
cat_data["Self_Employed"] = le_self_employed.fit_transform(cat_data["Self_Employed"])

le_property_area = LabelEncoder()
cat_data["Property_Area"] = le_property_area.fit_transform(cat_data["Property_Area"])

le_loan_status = LabelEncoder()
cat_data["Loan_Status"] = le_loan_status.fit_transform(cat_data["Loan_Status"])

df = pd.concat([cat_data, num_data], axis=1)
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X, y)

# Saving model to disk
pickle.dump(decision_tree, open('pickles/model.pkl', 'wb'))
pickle.dump(le_gender, open('pickles/le_gender.pkl', 'wb'))
pickle.dump(le_married, open('pickles/le_married.pkl', 'wb'))
pickle.dump(le_dependents, open('pickles/le_dependents.pkl', 'wb'))
pickle.dump(le_education, open('pickles/le_education.pkl', 'wb'))
pickle.dump(le_self_employed, open('pickles/le_self_employed.pkl', 'wb'))
pickle.dump(le_property_area, open('pickles/le_property_area.pkl', 'wb'))
pickle.dump(le_loan_status, open('pickles/le_loan_status.pkl', 'wb'))

print("Model Trained and Saved Successfully")
