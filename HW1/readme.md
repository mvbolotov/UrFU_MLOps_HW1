В данной работе представлен конвейер для автоматизации работы с моделью машинного обучения. 
Отдельные этапы конвейера машинного обучения находятся в разных python–скриптах, которые потом соединяются 
с помощью bash-скрипта.

Этапы:
1. [data_creation.py](data_creation.py) Создание набора данных с помощью функции `make_regression`
2. [model_preprocessing.py](model_preprocessing.py) Предобработка данных с помощью стандартизации `StandardScaler`
3. [model_preparation.py](model_preparation.py) Создание модели `LinearRegression` и обучение на тренировочной выборке
4. [model_testing.py](model_testing.py) Тестирование модели на тестовых данных и анализ результатов с помощью метрик машинного обучения `mean_squared_error`, `r2_score`
5. [pipeline.sh](pipeline.sh) Создание bash скрипта, который "склеивает" все этапы и последовательно их реализует