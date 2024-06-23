from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path


# Создание искусственного датасета с 6 признаками, небольшим шумом, и одной целевой переменной

features, target = make_regression(n_samples=100,
                                   n_features=6,
                                   noise=1,
                                   n_targets=1,
                                   random_state=42)

# Разделение на train и test

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

df_X_train = pd.DataFrame(X_train)
df_y_train = pd.DataFrame(y_train)
df_X_test = pd.DataFrame(X_test)
df_y_test = pd.DataFrame(y_test)

filepath_X_train = Path('X_train.csv')
filepath_X_train.parent.mkdir(parents=True, exist_ok=True)
df_X_train.to_csv(filepath_X_train, index=False)

filepath_y_train = Path('y_train.csv')
filepath_y_train.parent.mkdir(parents=True, exist_ok=True)
df_y_train.to_csv(filepath_y_train, index=False)

filepath_X_test = Path('X_test.csv')
filepath_X_test.parent.mkdir(parents=True, exist_ok=True)
df_X_test.to_csv(filepath_X_test, index=False)

filepath_y_test = Path('y_test.csv')
filepath_y_test.parent.mkdir(parents=True, exist_ok=True)
df_y_test.to_csv(filepath_y_test, index=False)

