import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from keras.optimizers import Adam

app = Flask(__name__)

# Загрузка данных
df = pd.read_csv("movie_rental_dataset.csv")

# Предобработка данных
numerical_cols = ['amount', 'length', 'rental_period', 'release_year']
categorical_cols = ['customer_id', 'city', 'district', 'category', 'title', 'last_name', 'name', 'address']

print("Изначальное количество строк в DataFrame:", len(df))

# Преобразование целевой переменной
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')  # Преобразуем рейтинг в числовой формат
if df['rating'].isnull().any():
    print("Заменяем NaN значения в rating на среднее значение.")
    df['rating'].fillna(df['rating'].mean(), inplace=True)

# Преобразование числовых столбцов
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Преобразуем в числовой формат
    if df[col].isnull().any():
        print(f"Заменяем NaN значения в столбце {col} на среднее значение.")
        df[col].fillna(df[col].mean(), inplace=True)  # Заменяем NaN на среднее значение

print("Количество строк после обработки числовых столбцов:", len(df))

# Масштабирование числовых данных
if len(df) > 0:
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
else:
    raise ValueError("После обработки числовых данных DataFrame пуст. Проверьте входные данные.")

# Кодирование категориальных признаков
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Преобразование в строку для кодирования
    label_encoders[col] = le

print("Количество строк после обработки категориальных столбцов:", len(df))

# Проверка итогового DataFrame
if df.empty:
    raise ValueError("После всех этапов предобработки DataFrame пуст. Проверьте входной датасет.")

print("Предобработка завершена. Итоговые данные:")
print(df.head())

# Разделение данных
X = df[categorical_cols + numerical_cols]
y = df['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели
inputs = []
embeddings = []

for col in categorical_cols:
    input_cat = Input(shape=(1,))
    embedding = Embedding(input_dim=df[col].nunique(), output_dim=10)(input_cat)
    embedding = Flatten()(embedding)
    inputs.append(input_cat)
    embeddings.append(embedding)

input_num = Input(shape=(len(numerical_cols),))
inputs.append(input_num)
embeddings.append(input_num)

merged = Concatenate()(embeddings)
dense = Dense(128, activation='relu')(merged)
dense = Dense(64, activation='relu')(dense)
output = Dense(1, activation='linear')(dense)

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Подготовка данных для обучения
X_train_cat = [X_train[col] for col in categorical_cols]
X_train_num = X_train[numerical_cols]
X_test_cat = [X_test[col] for col in categorical_cols]
X_test_num = X_test[numerical_cols]

model.fit(X_train_cat + [X_train_num], y_train, epochs=10, batch_size=32, validation_split=0.2)

# Flask endpoints
@app.route('/')
def index():
    # Получение уникальных значений для выпадающих списков
    dropdown_data = {
        'last_name': df['last_name'].unique(),
        'city': df['city'].unique(),
        'category': df['category'].unique(),
        'rental_period': sorted(df['rental_period'].unique()),
        'amount': sorted(df['amount'].unique())
    }
    return render_template('index.html', dropdown_data=dropdown_data)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.form  # Получение данных из формы

    try:
        # Обработка категориальных данных
        input_cat = [np.array([label_encoders[col].transform([str(data[col])])[0]]) for col in categorical_cols]

        # Обработка числовых данных
        input_num = np.array([[float(data[col]) for col in numerical_cols]])

        # Предсказание
        prediction = model.predict(input_cat + [input_num])
        predicted_rating = prediction[0][0]

        # Выбор топ-5 фильмов по рейтингу
        recommendations = df.sort_values(by='rating', ascending=False)['title'].unique()[:5].tolist()

        return render_template('result.html', predicted_rating=predicted_rating, recommendations=recommendations)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
