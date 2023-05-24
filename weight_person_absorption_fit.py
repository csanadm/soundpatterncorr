import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

##################################################
################## WEIGHT VS AD ##################
##################################################

df = pd.read_excel("weight_person_absorption.xlsx", sheet_name="Munka1")
X = df["weight (kg)"].values.reshape(-1, 1)
y = df["min aD (dB)"].values

regressor = LinearRegression()
regressor.fit(X, y)

plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color='red')
plt.xlabel('weight (kg)')
plt.ylabel('min aD (dB)')
plt.title('Linear Regression Fit')
plt.text(0.7, 0.7, f"Slope = {regressor.coef_[0]:.2f} dB/kg\nIntercept = {regressor.intercept_:.2f} dB\n$R^2$ = {regressor.score(X,y):.2f}", transform=plt.gca().transAxes)
plt.savefig("weight_person_absorption_fit.png")


##################################################
################## MEAN PD VS AD #################
##################################################

df = pd.read_excel("weight_person_absorption.xlsx", sheet_name="Munka2")
X = df["mean aD (dB)"].values.reshape(-1, 1)
y = df["mean pD"].values

regressor = LinearRegression()
regressor.fit(X, y)

plt.figure()
plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color='red')
plt.xlabel('mean aD (dB)')
plt.ylabel('mean pD')
plt.title('Linear Regression Fit')
plt.text(0.2, 0.7, f"Slope = {regressor.coef_[0]:.2f} 1/dB\nIntercept = {regressor.intercept_:.2f}\n$R^2$ = {regressor.score(X,y):.2f}", transform=plt.gca().transAxes)
plt.savefig("weight_person_absorption_fit_pd.png")