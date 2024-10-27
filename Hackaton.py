import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Загружаем данные, пропуская первую строку
file_path = 'tables/Train.xlsx'
data = pd.read_excel(file_path, skiprows=1)

# Устанавливаем заголовки из второй строки и удаляем её
data.columns = data.iloc[0]
data = data.drop(0)

# Заменяем некорректные значения на NaN
data.iloc[:, 5:] = data.iloc[:, 5:].replace({'68.20.2': None})  # Замените на нужные значения

# Преобразуем данные в числовой формат, игнорируя ошибки
numeric_data = data.iloc[:, 5:].apply(pd.to_numeric, errors='coerce')

# Убедимся, что имена столбцов являются строками
numeric_data.columns = numeric_data.columns.astype(str)

# Отделяем признаки (X) от целевой переменной (y)
X = numeric_data.iloc[:, :-2]  # Все столбцы, кроме последних двух
y = numeric_data.iloc[:, -2:]    # Последние два столбца — это целевая переменная

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаём и обучаем модель линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Делаем предсказания на тестовой выборке
predictions = model.predict(X_test)

# Выводим предсказанные значения объёма и платы для 33-й даты
print("Предсказанные значения объёма и платы для 33-й даты:", predictions)