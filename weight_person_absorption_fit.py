import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_excel("weight_person_absorption.xlsx", sheet_name="Munka2")
X = df["mean aD (dB)"].values.reshape(-1, 1)
y = df["mean pD"].values

regressor = LinearRegression()
regressor.fit(X, y)

plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color='red')
plt.xlabel('mean aD (dB)')
plt.ylabel('mean pD')
plt.title('Linear Regression Fit')
plt.text(0.2, 0.7, f"Slope = {regressor.coef_[0]:.2f}\nIntercept = {regressor.intercept_:.2f}\n$R^2$ = {regressor.score(X,y):.2f}", transform=plt.gca().transAxes)
plt.savefig("weight_person_absorption_fit.png")