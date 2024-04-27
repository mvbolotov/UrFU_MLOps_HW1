from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle

X_train_scaled = pd.read_csv('/home/maksim/PycharmProjects/MLOps/HW1/train/X_train_scaled.csv')
y_train = pd.read_csv('/home/maksim/PycharmProjects/MLOps/HW1/train/y_train.csv')

model = LinearRegression()

model.fit(X_train_scaled, y_train)

pickle.dump(model, open('/home/maksim/PycharmProjects/MLOps/HW1/model.pkl', 'wb'))