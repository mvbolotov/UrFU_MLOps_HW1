from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path

X_train = pd.read_csv('/home/maksim/PycharmProjects/MLOps/HW2/train/X_train.csv')
X_test = pd.read_csv('/home/maksim/PycharmProjects/MLOps/HW2/test/X_test.csv')


scaler = StandardScaler()

# масштабируем признаки обучающей выборки
X_train_scaled = scaler.fit_transform(X_train)
df_X_train_scaled = pd.DataFrame(X_train_scaled)

# преобразуем тестовые данные с использованием среднего и СКО, рассчитанных на обучающей выборке
# так тестовые данные не повлияют на обучение модели, и мы избежим утечки данных
X_test_scaled = scaler.transform(X_test)
df_X_test_scaled = pd.DataFrame(X_test_scaled)

filepath_X_train_scaled = Path('/home/maksim/PycharmProjects/MLOps/HW2/train/X_train_scaled.csv')
filepath_X_train_scaled.parent.mkdir(parents=True, exist_ok=True)
df_X_train_scaled.to_csv(filepath_X_train_scaled, index=False)

filepath_X_test_scaled = Path('/home/maksim/PycharmProjects/MLOps/HW2/test/X_test_scaled.csv')
filepath_X_test_scaled.parent.mkdir(parents=True, exist_ok=True)
df_X_test_scaled.to_csv(filepath_X_test_scaled, index=False)