import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV , RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_excel("weight_person_absorption.xlsx", sheet_name="Munka1")

X = df["weight (kg)"].values.reshape(-1, 1)
y = df["min aD (dB)"].values
i = np.arange(X.shape[0])
i_train, i_test, X_train, X_test, y_train, y_test = train_test_split(i, X, y, test_size=0.15, random_state=42)

param_grid = {'C': [1, 10, 100], 'gamma': [0.01, 0.1, 1.0], 'epsilon': [0.01, 0.1, 0.5], 'degree': [2], 'coef0': [1.0]}
param_grid = {'C': [10], 'gamma': [0.1], 'epsilon': [0.1], 'degree': [2], 'coef0': [1.0]}

grid_rbf = GridSearchCV(SVR(kernel='rbf'), param_grid=param_grid)
grid_lin = GridSearchCV(SVR(kernel='linear'), param_grid=param_grid)
grid_pol = GridSearchCV(SVR(kernel='poly'), param_grid=param_grid)
grid_sig = GridSearchCV(SVR(kernel='sigmoid'), param_grid=param_grid)

grid_rbf.fit(X_train, y_train)
grid_lin.fit(X_train, y_train)
grid_pol.fit(X_train, y_train)
grid_sig.fit(X_train, y_train)

y_rbf_pred_train = grid_rbf.predict(X_train)
y_lin_pred_train = grid_lin.predict(X_train)
y_pol_pred_train = grid_pol.predict(X_train)
y_sig_pred_train = grid_sig.predict(X_train)

y_rbf_pred_test = grid_rbf.predict(X_test)
y_lin_pred_test = grid_lin.predict(X_test)
y_pol_pred_test = grid_pol.predict(X_test)
y_sig_pred_test = grid_sig.predict(X_test)

print("Best parameters for RBF kernel:", grid_rbf.best_params_)
print("Best parameters for linear kernel:", grid_lin.best_params_)
print("Best parameters for polynomial kernel:", grid_pol.best_params_)
print("Best parameters for sigmoid kernel:", grid_sig.best_params_)

print("Predicting power for test sample")

mse_rbf = mean_squared_error(y_train, y_rbf_pred_train)
mse_lin = mean_squared_error(y_train, y_lin_pred_train)
mse_pol = mean_squared_error(y_train, y_pol_pred_train)
mse_sig = mean_squared_error(y_train, y_sig_pred_train)

print("  MSE for RBF kernel:", mse_rbf)
print("  MSE for linear kernel:", mse_lin)
print("  MSE for polynomial kernel:", mse_pol)
print("  MSE for sigmoid kernel:", mse_sig)

mae_rbf = mean_absolute_error(y_train, y_rbf_pred_train)
mae_lin = mean_absolute_error(y_train, y_lin_pred_train)
mae_pol = mean_absolute_error(y_train, y_pol_pred_train)
mae_sig = mean_absolute_error(y_train, y_sig_pred_train)

print("  MAE for RBF kernel:", mae_rbf)
print("  MAE for linear kernel:", mae_lin)
print("  MAE for polynomial kernel:", mae_pol)
print("  MAE for sigmoid kernel:", mae_sig)

r2_rbf = r2_score(y_train, y_rbf_pred_train)
r2_lin = r2_score(y_train, y_lin_pred_train)
r2_pol = r2_score(y_train, y_pol_pred_train)
r2_sig = r2_score(y_train, y_sig_pred_train)

print("  R^2 score for RBF kernel:", r2_rbf)
print("  R^2 score for linear kernel:", r2_lin)
print("  R^2 score for polynomial kernel:", r2_pol)
print("  R^2 score for sigmoid kernel:", r2_sig)

plt.figure()
plt.title("ML SVR quality check, various kernels")
plt.ylabel("min aD (dB)")
plt.xlabel("Measurement ")
plt.plot(i_train, y_train,    marker='P', color='C0', mfc='none', linestyle='none', label='data for training')
plt.plot(i_test,  y_test,     marker='P', color='C0',             linestyle='none', label='data for testing ')
plt.plot(i_train, y_rbf_pred_train, marker='o', color='C1', mfc='none', linestyle='none', label='RBF training')
plt.plot(i_test,  y_rbf_pred_test,  marker='o', color='C1',             linestyle='none', label='RBF testing')
#plt.plot(i_train, y_lin_pred_train, marker='x', color='C2', mfc='none', linestyle='none', label='Lin')
#plt.plot(i_test,  y_lin_pred_test,  marker='x', color='C2',             linestyle='none', label='Lin')
plt.plot(i_train, y_pol_pred_train, marker='s', color='C3', mfc='none', linestyle='none', label='Poly training')
plt.plot(i_test,  y_pol_pred_test,  marker='s', color='C3',             linestyle='none', label='Poly testing')
#plt.plot(i_train, y_sig_pred_train, marker='d', color='C4', mfc='none', linestyle='none', label='Sigm')
#plt.plot(i_test,  y_sig_pred_test,  marker='d', color='C4',             linestyle='none', label='Sigm')
plt.legend(loc='upper left')
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0-box.width*0.03,box.y0,box.width*0.97,box.height*1.05])
ax.legend(loc='center right', bbox_to_anchor=(1.20, 0.5))
plt.savefig("svr_scikit_weight_abs.png")