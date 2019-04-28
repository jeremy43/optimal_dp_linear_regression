#####################Import the packages 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


####################IMPORT THE DATABASE

columns = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num','marital-status', 'occupation', 'relationship', 'race', 'gender','capital-gain', 'capital-loss', 'hours-per-week', 'native-country','income']


train = pd.read_csv('adult_data.txt', sep=",\s", header=None, names = columns, engine = 'python')
test = pd.read_csv('adult_test.txt', sep=",\s", header=None, names = columns, engine = 'python')
test['income'].replace(regex=True,inplace=True,to_replace=r'\.',value=r'')


adult = pd.concat([test,train])
adult.reset_index(inplace = True, drop = True)

# Setting all the categorical columns to type category
for col in set(adult.columns) - set(adult.describe().columns):
    adult[col] = adult[col].astype('category')


for i,j in zip(adult.columns,(adult.values.astype(str) == '?').sum(axis = 0)):
    if j > 0:
        print(str(i) + ': ' + str(j) + ' records')


# Create one hot encoding of the categorical columns in the data frame.
def oneHotCatVars(df, df_cols):
    df_1 = adult_data = df.drop(columns=df_cols, axis=1)
    df_2 = pd.get_dummies(df[df_cols])

    return (pd.concat([df_1, df_2], axis=1, join='inner'))





majority_class = adult.workclass.value_counts().index[0]


adult.loc[(adult.workclass.values == '?'),'workclass'] = majority_class



test_data = adult[(adult.occupation.values == '?')].copy()
test_label = test_data.occupation

majority_class = adult.occupation.value_counts().index[0]


adult.loc[(adult.occupation.values == '?'),'occupation'] = majority_class
print(adult.occupation.unique())

majority_class = adult['native-country'].value_counts().index[0]
adult.loc[(adult['native-country'].values == '?'),'native-country'] = majority_class
"""
adult['workclass'] = adult['workclass'].cat.remove_categories('?')
adult['occupation'] = adult['occupation'].cat.remove_categories('?')
adult['native-country'] = adult['native-country'].cat.remove_categories('?')

"""
########### Preparing data for Training and testing

# Data Prep
adult_data = adult.drop(columns = ['income'])
adult_label = adult.income


adult_cat_1hot = pd.get_dummies(adult_data.select_dtypes('category'))
adult_non_cat = adult_data.select_dtypes(exclude = 'category')

adult_data_1hot = pd.concat([adult_non_cat, adult_cat_1hot], axis=1, join='inner')
train_data, test_data, train_label, test_label = train_test_split(adult_data_1hot, adult_label, test_size  = 0.25)

# Normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fitting only on training data
scaler.fit(train_data)
train_data = scaler.transform(train_data)

# Applying same transformation to test data
test_data = scaler.transform(test_data)

import pickle


log_reg = LogisticRegression(penalty = 'l2', dual = False, tol = 1e-4, fit_intercept = True,
                            solver = 'liblinear')
dataset = {}
dataset['train_data'] = train_data
dataset['test_data'] = test_data
dataset['train_label'] = train_label
dataset['test_label'] = test_label
log_reg.fit(train_data, train_label)
log_reg_pred = log_reg.predict(test_data)

file_Name = "adult.data"
# open the file for writing
fileObject = open(file_Name, 'wb')

# this writes the object a to the
# file named 'testfile'
pickle.dump(dataset, fileObject)
print(log_reg.score(test_data,test_label))
