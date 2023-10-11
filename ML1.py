import pandas as pd
import math

#importing data from csv file

df = pd.read_csv("C:\\Users\\G.Ruchitha\\OneDrive\\Documents\\Experiment-1.csv")


#Normalization

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


#training model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(independent, target, test_size= 0.2,random_state=50)  
from sklearn.ensemble import RandomForestRegressor 
model = RandomForestRegressor(n_estimators= 50,min_samples_split=3,min_samples_leaf=3)  
model.fit(X_train, y_train)  
y_pred= model.predict(X_test) 


#accuracy of the model

accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

mae = mean_absolute_error(y_test,y_pred)
print("Mean Absolute Error",mae)

rmse = math.sqrt(mse)
print("Root Mean Square Error",rmse)

r2 = r2_score(y_test,y_pred )
print("R squared" ,r2)