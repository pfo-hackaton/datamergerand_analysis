import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# 1. Загружаем данные из Excel файла, пропуская первую строку
file_path = 'tables/Train.xlsx'
data = pd.read_excel(file_path, skiprows=1)

# 2. Устанавливаем заголовки из второй строки и удаляем её
data.columns = data.iloc[0]
data = data.drop(0)

# 3. Заменяем неправильные значения на NaN (пустые значения)
data.iloc[:, 5:] = data.iloc[:, 5:].replace({'68.20.2': None})

# 4. Преобразуем данные в числовой формат, чтобы с ними было легче работать
numeric_data = data.iloc[:, 5:].apply(pd.to_numeric, errors='coerce')

# 5. Убедимся, что имена столбцов — это строки
numeric_data.columns = numeric_data.columns.astype(str)

# 6. Отделяем признаки (X) от целевой переменной (Y)
X = numeric_data.iloc[:, :-2]  # Все столбцы, кроме последних двух
y = numeric_data.iloc[:, -2:]   # Последние два столбца — это целевая переменная

# 7. Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Создаём модель линейной регрессии и обучаем её
model = LinearRegression()
model.fit(X_train, y_train)

# 9. Предсказываем значения на тестовой выборке
predictions = model.predict(X_test)

# 10. Выводим предсказанные значения объёма и платы для 33-й даты
print("Предсказанные значения объёма и платы для 33-й даты:", predictions)

# 11. Сортируем объемы поставок по данным
sorted_data = numeric_data.sort_values(by=numeric_data.columns[-2], ascending=False)
print("Сортированные данные по объему поставок:n", sorted_data)

# 12. Объединяем данные с региональными таблицами по ID
region_data = pd.read_excel('tables/RegionData.xlsx')  # Загружаем региональные данные
merged_data = pd.merge(sorted_data, region_data, on='ID', how='left')
print("Данные после объединения с региональными таблицами:n", merged_data)

# 13. Убираем компании с объемами поставок 0, если отсутствует елс
filtered_data = merged_data[(merged_data[numeric_data.columns[-2]] > 0) | (merged_data['елс'].notnull())]
print("Данные после фильтрации:n", filtered_data)

# 14. Проверка динамики роста/падения объема продаж
def classify_growth(data):
    """Функция для определения роста или падения объемов продаж."""
    if data[-1] > data[-2]:
        return 'Рост'
    elif data[-1] < data[-2]:
        return 'Падение'
    else:
        return 'Стабильно'

# 15. Применяем классификацию к каждому региону
filtered_data['Динамика'] = filtered_data.iloc[:, -2:].apply(classify_growth, axis=1)
print("Данные с классификацией динамики:n", filtered_data)

# 16. Дополнительный анализ для компаний с падением объемов продаж
declining_sales = filtered_data[filtered_data['Динамика'] == 'Падение']
if not declining_sales.empty:
    print("Компании с падением объемов продаж:")
    for index, row in declining_sales.iterrows():
        print(f"Компания: {row['Компания']}, Объем: {row[numeric_data.columns[-2]]}")

# 17. Визуализация данных: График объемов продаж по компаниям
plt.figure(figsize=(10, 6))
plt.barh(filtered_data['Компания'], filtered_data[numeric_data.columns[-2]], color='skyblue')
plt.xlabel('Объем продаж')
plt.title('Объем продаж по компаниям')
plt.grid(axis='x')
plt.show()

# 18. Сохранение результатов в новый Excel файл
output_file_path = 'tables/Processed_Data.xlsx'
filtered_data.to_excel(output_file_path, index=False)
print(f"Результаты сохранены в файл: {output_file_path}")
