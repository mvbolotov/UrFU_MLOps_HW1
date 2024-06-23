from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle

X_train_scaled = pd.read_csv('X_train_scaled.csv')
y_train = pd.read_csv('y_train.csv')

model = LinearRegression()

model.fit(X_train_scaled, y_train)

pickle.dump(model, open('model.pkl', 'wb'))