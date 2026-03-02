import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
from pathlib import Path
import os
from datetime import timedelta
from train_model import train

def download_data():
    # Загружаем ваш локальный файл
    df = pd.read_csv('./ndtv_data_final.csv', delimiter=',')
    # Удаляем ненужный первый столбец (индекс)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df.to_csv("phones.csv", index=False)
    print("Загружено записей: ", df.shape)
    return df

def clear_data():
    df = pd.read_csv("phones.csv")
    
    # Определяем типы колонок для вашего датасета
    cat_columns = ['Brand', 'Model', 'Touchscreen', 'Operating system', 
                   'Wi-Fi', 'Bluetooth', 'GPS', '3G', '4G/ LTE']
    
    num_columns = ['Battery capacity (mAh)', 'Screen size (inches)', 
                   'Resolution x', 'Resolution y', 'Processor', 'RAM (MB)', 
                   'Internal storage (GB)', 'Rear camera', 'Front camera', 
                   'Number of SIMs', 'Price']
    
    # Проверяем наличие колонок
    available_cat = [col for col in cat_columns if col in df.columns]
    available_num = [col for col in num_columns if col in df.columns]
    
    print(f"Категориальные колонки: {available_cat}")
    print(f"Числовые колонки: {available_num}")
    
    # Очистка данных от выбросов
    
    # 1. Удаляем записи с отрицательной ценой или слишком низкой ценой
    question_price = df[df['Price'] < 1000]
    if len(question_price) > 0:
        print(f"Удаляем {len(question_price)} записей с ценой < 1000")
        df = df.drop(question_price.index)
    
    # 2. Удаляем слишком дорогие телефоны (выбросы)
    question_price_high = df[df['Price'] > 200000]
    if len(question_price_high) > 0:
        print(f"Удаляем {len(question_price_high)} записей с ценой > 200000")
        df = df.drop(question_price_high.index)
    
    # 3. Проверяем батарею (реалистичные значения)
    if 'Battery capacity (mAh)' in df.columns:
        question_battery = df[df['Battery capacity (mAh)'] < 500]
        if len(question_battery) > 0:
            print(f"Удаляем {len(question_battery)} записей с батареей < 500 mAh")
            df = df.drop(question_battery.index)
        
        question_battery_high = df[df['Battery capacity (mAh)'] > 10000]
        if len(question_battery_high) > 0:
            print(f"Удаляем {len(question_battery_high)} записей с батареей > 10000 mAh")
            df = df.drop(question_battery_high.index)
    
    # 4. Проверяем RAM (реалистичные значения)
    if 'RAM (MB)' in df.columns:
        question_ram = df[df['RAM (MB)'] < 256]
        if len(question_ram) > 0:
            print(f"Удаляем {len(question_ram)} записей с RAM < 256 MB")
            df = df.drop(question_ram.index)
        
        question_ram_high = df[df['RAM (MB)'] > 16000]
        if len(question_ram_high) > 0:
            print(f"Удаляем {len(question_ram_high)} записей с RAM > 16000 MB")
            df = df.drop(question_ram_high.index)
    
    # 5. Проверяем внутреннюю память
    if 'Internal storage (GB)' in df.columns:
        question_storage = df[df['Internal storage (GB)'] < 1]
        if len(question_storage) > 0:
            print(f"Удаляем {len(question_storage)} записей с памятью < 1 GB")
            df = df.drop(question_storage.index)
    
    # 6. Проверяем разрешение экрана
    if 'Resolution x' in df.columns and 'Resolution y' in df.columns:
        question_res_x = df[df['Resolution x'] < 200]
        question_res_y = df[df['Resolution y'] < 200]
        if len(question_res_x) > 0:
            df = df.drop(question_res_x.index)
        if len(question_res_y) > 0:
            df = df.drop(question_res_y.index)
    
    # Сбрасываем индекс после удалений
    df = df.reset_index(drop=True)
    
    # Кодируем категориальные признаки
    # Для бинарных признаков (Yes/No) используем простой OrdinalEncoder
    binary_features = ['Touchscreen', 'Wi-Fi', 'Bluetooth', 'GPS', '3G', '4G/ LTE']
    binary_available = [col for col in binary_features if col in df.columns]
    
    for col in binary_available:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # Для Brand, Model, Operating system используем OneHotEncoder
    # Но сначала проверим количество уникальных значений
    high_card_features = ['Brand', 'Model', 'Operating system']
    high_card_available = [col for col in high_card_features if col in df.columns]
    
    # Используем OneHotEncoder для категориальных с большим количеством значений
    for col in high_card_available:
        if col in df.columns:
            # Проверяем количество уникальных значений
            if df[col].nunique() < 50:  # Если меньше 50, применяем OneHotEncoder
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
            else:  # Если много уникальных значений, используем частотное кодирование
                freq_encoding = df[col].value_counts().to_dict()
                df[f'{col}_freq'] = df[col].map(freq_encoding)
                df = df.drop(columns=[col])
    
    # Сохраняем очищенные данные
    df.to_csv('df_clear.csv', index=False)
    print("Очистка завершена. Итоговый размер:", df.shape)
    return True

# Настройка DAG
dag_cars = DAG(
    dag_id="phone_price_prediction",
    start_date=datetime(2025, 2, 3),
    concurrency=4,
    schedule_interval=timedelta(minutes=30),
    max_active_runs=1,
    catchup=False,
)

download_task = PythonOperator(
    python_callable=download_data, 
    task_id="download_phones", 
    dag=dag_cars
)

clear_task = PythonOperator(
    python_callable=clear_data, 
    task_id="clear_phones", 
    dag=dag_cars
)

train_task = PythonOperator(
    python_callable=train, 
    task_id="train_model", 
    dag=dag_cars
)

download_task >> clear_task >> train_task