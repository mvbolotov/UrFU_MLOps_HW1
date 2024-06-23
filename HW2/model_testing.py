from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import pickle

model = pickle.load(open('model.pkl', 'rb'))
X_test_scaled = pd.read_csv('X_test_scaled.csv')
y_pred = model.predict(X_test_scaled)

#Проверим результат работы модели на тестовых данных
y_test = pd.read_csv('y_test.csv')

print('r2:', r2_score(y_test, y_pred))

print('mse:', mean_squared_error(y_test, y_pred))
