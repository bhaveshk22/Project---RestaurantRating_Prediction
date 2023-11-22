# you can create model here and save it using pickle.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
import pickle

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('zomato_df.csv')

df.drop('Unnamed: 0', axis=1, inplace=True)

#splitting the dataset
x = df.drop('rate',axis=1)
y = df['rate']

x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=.3, random_state=10)


# #creating model
# etr = ExtraTreesRegressor(n_estimators=150)
# etr.fit(x_train,y_train)

# yhat = etr.predict(x_test)


# #saving our model to disk using pickle
# pickle.dump(etr, open('etr_model.pkl', 'wb'))
# model = pickle.load(open('etr_model.pkl','rb'))

# print(yhat)







# or you can use the model saved in jupyter notebook
model = pickle.load(open('model.pkl','rb'))

print(model.score(x_test,y_test))