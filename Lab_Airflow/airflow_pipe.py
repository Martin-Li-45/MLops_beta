import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer # т.н. преобразователь колонок
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
    df = pd.read_csv('/home/meshkov/airflow/dags/ndtv_data_final.csv', delimiter = ',')
    df.to_csv("/home/meshkov/airflow/dags/phones.csv", index = False)
    print("df: ", df.shape)
    return df

def clear_data():
    # Загружаем данные
    df = pd.read_csv("/home/meshkov/airflow/dags/phones.csv", index_col=0)
    
    # Определяем типы колонок
    num_columns = ['Battery capacity (mAh)', 'Screen size (inches)', 'Resolution x', 'Resolution y', 
                   'RAM (MB)', 'Internal storage (GB)', 'Rear camera', 'Front camera', 
                   'Number of SIMs', 'Price']
    binary_columns = ['Touchscreen', 'Wi-Fi', 'Bluetooth', 'GPS', '3G', '4G/ LTE']
    
    # Колонки, которые мы полностью удаляем (они категориальные / идентификаторы)
    columns_to_drop = ['Name', 'Model', 'Brand', 'Processor', 'Operating system']
    
    # Удаляем их сразу после загрузки
    df = df.drop(columns=columns_to_drop, errors='ignore')

    df['PPI'] = np.sqrt(df['Resolution x']**2 + df['Resolution y']**2) / df['Screen size (inches)']
    df['Camera_sum'] = df['Rear camera'] + df['Front camera']
    df['RAM_GB'] = df['RAM (MB)'] / 1024
    df['Storage_per_RAM'] = df['Internal storage (GB)'] / df['RAM_GB'].clip(0.5)
    df['Battery_per_inch'] = df['Battery capacity (mAh)'] / df['Screen size (inches)']
    # Преобразуем Yes/No → 1/0 (до фильтрации)
    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, "yes": 1, "no": 0}).fillna(0).astype(int)
    
    # ────────────────────────────────────────────────────────────────
    # Фильтрация выбросов
    # ────────────────────────────────────────────────────────────────
    
    df = df[df["Battery capacity (mAh)"].between(500, 10000)]
    df = df[df["Screen size (inches)"].between(3.0, 7.5)]
    df = df[df["Resolution x"].between(240, 4000)]
    df = df[df["Resolution y"].between(320, 4000)]
    df = df[df["RAM (MB)"].between(256, 16384)]
    df = df[df["Internal storage (GB)"].between(1, 1024)]
    df = df[df["Rear camera"].between(0, 200)]
    df = df[df["Front camera"].between(0, 100)]
    df = df[df["Number of SIMs"] <= 3]
    df = df[df["Price"].between(1000, 500000)]
    
    # Сбрасываем индекс
    df = df.reset_index(drop=True)
    
    # Сохраняем очищенный датасет (теперь только числовые + бинарные столбцы)
    df.to_csv('/home/meshkov/airflow/dags/df_clear.csv', index=False)
    print("Очищенный датасет сохранён в df_clear.csv")
    print(f"Размер после очистки: {df.shape}")
    print("Оставшиеся столбцы:", df.columns.tolist())
    
    return True

dag_phones = DAG(
    dag_id="train_pipe",
    start_date=datetime(2025, 2, 3),
    max_active_tasks=4,
    schedule=timedelta(minutes=5),
    #schedule="@hourly",
    max_active_runs=1,
    catchup=False,
)
download_task = PythonOperator(python_callable=download_data, task_id = "download_phones", dag = dag_phones)
clear_task = PythonOperator(python_callable=clear_data, task_id = "clear_phones", dag = dag_phones)
train_task = PythonOperator(python_callable=train, task_id = "train_phones", dag = dag_phones)
download_task >> clear_task >> train_task


