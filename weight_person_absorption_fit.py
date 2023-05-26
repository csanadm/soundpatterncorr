import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

##################################################
############### WEIGHT VS AD SOUND ###############
##################################################

df = pd.read_excel("weight_person_absorption.xlsx", sheet_name="Munka1")
X = df["weight (kg)"].values.reshape(-1, 1)
y = df["min aD (dB)"].values

regressor = LinearRegression()
regressor.fit(X, y)

plt.figure()
plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color='red')
plt.xlabel('weight (kg)')
plt.ylabel('min aD (dB)')
plt.title('Linear Regression Fit, min aD vs weight')
plt.text(0.7, 0.7, f"Slope = {regressor.coef_[0]:.2f} dB/kg\nIntercept = {regressor.intercept_:.2f} dB\n$R^2$ = {regressor.score(X,y):.2f}", transform=plt.gca().transAxes)
plt.savefig("weight_person_absorption_fit.png")

regressor = LinearRegression()
regressor.fit(X, y**2)

plt.figure()
plt.scatter(X, y**2)
plt.plot(X, regressor.predict(X), color='red')
plt.xlabel('weight (kg)')
plt.ylabel(r'min aD$^2$ (dB$^2$)')
plt.title(r'Linear Regression Fit, min aD$^2$ vs weight')
plt.text(0.2, 0.7, f"Slope = {regressor.coef_[0]:.2f} dB$^2$/kg\nIntercept = {regressor.intercept_:.2f} dB$^2$\n$R^2$ = {regressor.score(X,y**2):.2f}", transform=plt.gca().transAxes)
plt.savefig("weight_person_absorption_fit_square.png")


##################################################
############## MEAN AD VS AD BRAIN ###############
##################################################

df = pd.read_excel("weight_person_absorption.xlsx", sheet_name="Munka2")
X = df["mean aD (dB)"].values.reshape(-1, 1)
y = df["mean aD (dB) brain"].values

regressor = LinearRegression()
regressor.fit(X, y)

plt.figure()
plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color='red')
plt.xlabel('mean aD (dB) sound')
plt.ylabel('mean aD (dB) brain')
plt.title('Linear Regression Fit, mean aD sound vs brain')
plt.text(0.2, 0.7, f"Slope = {regressor.coef_[0]:.2f}\nIntercept = {regressor.intercept_:.2f} dB\n$R^2$ = {regressor.score(X,y):.2f}", transform=plt.gca().transAxes)

plt.savefig("weight_person_absorption_fit_brain.png")

X = df["mean aD (dB) high freq"].values.reshape(-1, 1)
y = df["mean aD (dB) gamma"].values

regressor = LinearRegression()
regressor.fit(X, y)

plt.figure()
plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color='red')
plt.xlabel('mean aD (dB) high freq')
plt.ylabel('mean aD (dB) gamma')
plt.title('Linear Regression Fit, mean aD high freq vs gamma')
plt.text(0.2, 0.7, f"Slope = {regressor.coef_[0]:.2f}\nIntercept = {regressor.intercept_:.2f} dB\n$R^2$ = {regressor.score(X,y):.2f}", transform=plt.gca().transAxes)
plt.savefig("weight_person_absorption_fit_highfreq.png")