import pandas as pd
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from train_model import train

def download_data():
    df = pd.read_csv('/home/meshkov/airflow/dags/ndtv_data_final.csv', delimiter=',')
    df.to_csv("/home/meshkov/airflow/dags/phones.csv", index=False)
    print("Данные загружены, размер:", df.shape)
    return df

def clear_data():
    """Очистка и базовая предобработка датасета смартфонов"""
    # Загружаем данные
    df = pd.read_csv("/home/meshkov/airflow/dags/phones.csv", index_col=0)
    
    # ───────────────────────────────────────────────
    # 1. Удаляем почти бесполезные для модели колонки
    # ───────────────────────────────────────────────
    cols_to_drop = ['Name', 'Model']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # ───────────────────────────────────────────────
    # 2. Мягкая фильтрация аномалий (не удаляем всё подряд)
    # ───────────────────────────────────────────────
    # Батарея
    df = df[df["Battery capacity (mAh)"].between(800, 8500)]
    
    # Экран
    df = df[df["Screen size (inches)"].between(3.0, 7.8)]
    
    # Разрешение
    df = df[df["Resolution x"].between(320, 3840)]
    df = df[df["Resolution y"].between(480, 4320)]
    
    # Оперативка
    df = df[df["RAM (MB)"].between(512, 24576)]          # до 24 ГБ
    
    # Встроенная память
    df = df[df["Internal storage (GB)"].between(1, 2048)] # до 2 ТБ
    
    # Камеры
    df = df[df["Rear camera"].between(0, 200)]
    df = df[df["Front camera"].between(0, 100)]
    
    # SIM-карты
    df = df[df["Number of SIMs"].between(1, 3)]
    
    # Цена (в рублях, судя по датасету)
    df = df[df["Price"].between(3000, 450000)]
    
    # ───────────────────────────────────────────────
    # 3. Приводим бинарные признаки к 0/1
    # ───────────────────────────────────────────────
    binary_cols = ['Touchscreen', 'Wi-Fi', 'Bluetooth', 'GPS', '3G', '4G/ LTE']
    
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).fillna(0).astype(int)
    
    # ───────────────────────────────────────────────
    # 4. Базовая обработка пропусков (если есть)
    # ───────────────────────────────────────────────
    # Для числовых — медиана, для категориальных — самая частая
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    
    # ───────────────────────────────────────────────
    # 5. Сбрасываем индекс и сохраняем
    # ───────────────────────────────────────────────
    df = df.reset_index(drop=True)
    df.to_csv('/home/meshkov/airflow/dags/df_clear.csv', index=False)
    
    print("Очищенный датасет сохранён в df_clear.csv")
    print(f"Размер после очистки: {df.shape}")
    print(f"Пропусков осталось: {df.isna().sum().sum()}")
    
    return True
    
dag_phones = DAG(
    dag_id="train_pipe",
    start_date=datetime(2025, 2, 3),
    schedule=timedelta(minutes=5),
    max_active_runs=1,
    catchup=False,
)

download_task = PythonOperator(python_callable=download_data, task_id="download_phones", dag=dag_phones)
clear_task = PythonOperator(python_callable=clear_data, task_id="clear_phones", dag=dag_phones)
train_task = PythonOperator(python_callable=train, task_id="train_phones", dag=dag_phones)

download_task >> clear_task >> train_task

