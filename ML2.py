import pandas as pd
import numpy as np
import math
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("C:\\Users\\G.Ruchitha\\OneDrive\\Documents\\Experiment-1.csv")
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
df = scalar.fit_transform(df)
df = pd.DataFrame(df,columns=['Rain Fall (mm)','Fertilizer(urea) (kg/acre)','Temperature (°C)','Nitrogen (N)','Phosphorus (P)','Potassium (K)','Yeild (Q/acre)'])
print("NORMALIZED DATA")

print(df.head())

#declaring variables
independent = df.iloc[:,0:6]
print('independent variables:')

print(independent.head())
target = df['Yeild (Q/acre)']
print('target variable:')

print(target.head())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(independent, target, test_size=0.2, random_state=50)
from sklearn.ensemble import RandomForestRegressor 
model = RandomForestRegressor(n_estimators= 50,min_samples_split=3,min_samples_leaf=3)  
model.fit(X_train, y_train)  
y_pred= model.predict(X_test) 


# Support Vector Regression (SVR)
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_model.fit(X_train, y_train)

# Radial Basis Function Neural Network (RBFNN)
rbfnn_model = MLPRegressor(hidden_layer_sizes=(10,), activation='logistic', solver='lbfgs', max_iter=1000)
rbfnn_model.fit(X_train, y_train)

# Back Propagation Neural Network (BPNN)
bp_model = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=1000)
bp_model.fit(X_train, y_train)

# Predictions
y_pred_svr = svr_model.predict(X_test)
y_pred_rbfnn = rbfnn_model.predict(X_test)
y_pred_bp = bp_model.predict(X_test)

 #Calculate R2 score
r2_svr = r2_score(y_test, y_pred_svr)
r2_rbfnn = r2_score(y_test, y_pred_rbfnn)
r2_bp = r2_score(y_test, y_pred_bp)

# Calculate MSE
mse_svr = mean_squared_error(y_test, y_pred_svr)
mse_rbfnn = mean_squared_error(y_test, y_pred_rbfnn)
mse_bp = mean_squared_error(y_test, y_pred_bp)

# Calculate RMSE
rmse_svr = np.sqrt(mse_svr)
rmse_rbfnn = np.sqrt(mse_rbfnn)
rmse_bp = np.sqrt(mse_bp)

# Print the evaluation metrics
print("SVR: R2 =", r2_svr, "MSE =", mse_svr, "RMSE =", rmse_svr)
print("RBFNN: R2 =", r2_rbfnn, "MSE =",mse_rbfnn, "RMSE=", rmse_rbfnn)
print("BP: R2 =", r2_bp, "MSE =",mse_bp, "RMSE=", rmse_bp)



