import pandas as pd
import numpy as np
import os

df=pd.read_csv('result.csv')

X=[]
y=[]
temp=list(df['temp'])
power=list(df['power'])
accesses=list(df['accesses'])
for i in range(0,df.shape[0],32):
    X.append(temp[i:i+32]+power[i:i+32]+accesses[i:i+32])
    y.append(df['prediction'][i:i+32])
    

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin = LinearRegression()

lin.fit(X_train, y_train)

# Predicting the Test set results
y_pred = lin.predict(X_test)

print("Leniar Regression Done")

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 3)
X_poly = poly.fit_transform(X_train)
poly.fit(X_poly, y_train)
lin2 = LinearRegression()
lin2.fit(X_poly, y_train)

# Predicting the Test set results
y_pred2 = lin2.predict(poly.fit_transform(X_test))

print("Polynomial Regression Done")

# # Fitting SVR to the dataset
# from sklearn.svm import SVR
# svr = SVR(kernel = 'rbf')
# svr.fit(X_train, y_train)

# # Predicting the Test set results
# y_pred3 = svr.predict(X_test)

# print("SVR Done")

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state = 0)
dtr.fit(X_train, y_train)

# Predicting the Test set results
y_pred4 = dtr.predict(X_test)

print("Decision Tree Regression Done")

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 10, random_state = 0)
rfr.fit(X_train, y_train)

# Predicting the Test set results
y_pred5 = rfr.predict(X_test)

print("Random Forest Regression Done")

# # Fitting AdaBoost to the Training set
# from sklearn.ensemble import AdaBoostRegressor
# ada = AdaBoostRegressor()
# ada.fit(X_train, y_train)

# # Predicting the Test set results
# y_pred9 = ada.predict(X_test)

# print("AdaBoost Done")

# # Fitting GradientBoosting to the Training set
# from sklearn.ensemble import GradientBoostingRegressor
# gbr = GradientBoostingRegressor()
# gbr.fit(X_train, y_train)

# # Predicting the Test set results
# y_pred10 = gbr.predict(X_test)

# print("GradientBoosting Done")

# Fitting BaggingRegressor to the Training set
from sklearn.ensemble import BaggingRegressor
br = BaggingRegressor()
br.fit(X_train, y_train)

# Predicting the Test set results
y_pred11 = br.predict(X_test)

print("BaggingRegressor Done")

# Fitting ExtraTreesRegressor to the Training set
from sklearn.ensemble import ExtraTreesRegressor
etr = ExtraTreesRegressor()
etr.fit(X_train, y_train)

# Predicting the Test set results
y_pred12 = etr.predict(X_test)

print("ExtraTreesRegressor Done")

# Fitting Neural Network to the Training set
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor()
mlp.fit(X_train, y_train)

# Predicting the Test set results
y_pred13 = mlp.predict(X_test)

print("MLPRegressor Done")

# Calculating the accuracy of the model
from sklearn.metrics import r2_score
print("Linear Regression: ",r2_score(y_test,y_pred))
print("Polynomial Regression: ",r2_score(y_test,y_pred2))
# print("SVR: ",r2_score(y_test,y_pred3))
print("Decision Tree Regression: ",r2_score(y_test,y_pred4))
print("Random Forest Regression: ",r2_score(y_test,y_pred5))
# print("AdaBoost: ",r2_score(y_test,y_pred9))
# print("GradientBoosting: ",r2_score(y_test,y_pred10))
print("BaggingRegressor: ",r2_score(y_test,y_pred11))
print("ExtraTreesRegressor: ",r2_score(y_test,y_pred12))
print("MLPRegressor: ",r2_score(y_test,y_pred13))

# Saving the model
os.mkdir('models2')
import pickle
pickle.dump(lin, open('models2/lin.pkl','wb'))
pickle.dump(lin2, open('models2/lin2.pkl','wb'))
# pickle.dump(svr, open('models2/svr.pkl','wb'))
pickle.dump(dtr, open('models2/dtr.pkl','wb'))
pickle.dump(rfr, open('models2/rfr.pkl','wb'))
# pickle.dump(ada, open('models2/ada.pkl','wb'))
# pickle.dump(gbr, open('models2/gbr.pkl','wb'))
pickle.dump(br, open('models2/br.pkl','wb'))
pickle.dump(etr, open('models2/etr.pkl','wb'))
pickle.dump(mlp, open('models2/mlp.pkl','wb'))
