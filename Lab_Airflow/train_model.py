import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import joblib
from datetime import datetime

def create_features(df):
    """Добавляем полезные признаки"""
    df = df.copy()
    # Качество экрана
    df['PPI'] = np.sqrt(df['Resolution x']**2 + df['Resolution y']**2) / df['Screen size (inches)']
    df['Total_pixels'] = df['Resolution x'] * df['Resolution y']
    
    # Камеры и память
    df['Camera_total'] = df['Rear camera'] + df['Front camera']
    df['RAM_GB'] = df['RAM (MB)'] / 1024.0
    df['Battery_per_inch'] = df['Battery capacity (mAh)'] / df['Screen size (inches)']
    df['Storage_per_RAM'] = df['Internal storage (GB)'] / df['RAM_GB']
    
    return df


def train():
    print("=== Запуск обучения улучшенной модели ===")
    
    # Загружаем данные
    df = pd.read_csv("/home/meshkov/airflow/dags/df_clear.csv")
    
    # УДАЛЯЕМ бесполезные колонки
    df = df.drop(columns=['Name', 'Model'], errors='ignore')
    
    # Feature Engineering
    df = create_features(df)
    
    # Разделяем признаки
    cat_features = ['Brand', 'Processor', 'Operating system']
    num_features = [col for col in df.columns if col not in ['Price'] + cat_features]
    
    X = df.drop(columns=['Price'])
    y = df['Price']
    
    # Train / Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Preprocessor
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ])
    
    # Полный пайплайн
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', SGDRegressor(random_state=42, max_iter=2000, tol=1e-3))
    ])
    
    # Преобразование целевой переменной (очень важно для цен!)
    power_trans = PowerTransformer(method='yeo-johnson')
    y_train_scaled = power_trans.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    
    # Параметры для поиска (уменьшил сетку для скорости)
    param_grid = {
        'model__alpha': [0.0001, 0.001, 0.01],
        'model__l1_ratio': [0.05, 0.1, 0.15, 0.2],
        'model__penalty': ['l1', 'l2', 'elasticnet'],
        'model__loss': ['squared_error', 'huber']
    }
    
    # MLflow
    mlflow.set_experiment("phones_price_prediction_v2")
    
    with mlflow.start_run(run_name=f"improved_run_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        
        print("Запуск GridSearchCV...")
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=3, 
            n_jobs=-1, 
            scoring='neg_root_mean_squared_error',
            verbose=1
        )
        
        grid_search.fit(X_train, y_train_scaled)
        
        best_model = grid_search.best_estimator_
        
        # Предсказание
        y_pred_scaled = best_model.predict(X_val)
        y_pred = power_trans.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        # Метрики
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        # Логирование в MLflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        for param, value in grid_search.best_params_.items():
            clean_name = param.replace('model__', '')
            mlflow.log_param(clean_name, value)
        
        mlflow.log_param("n_features", X_train.shape[1])
        
        # Сохраняем модель
        mlflow.sklearn.log_model(best_model, "model")
        
        # Локальное сохранение
        joblib.dump(best_model, "/home/meshkov/airflow/dags/lr_phones.pkl")
        joblib.dump(power_trans, "/home/meshkov/airflow/dags/power_trans.pkl")
        
        print(f"\n✅ Модель успешно обучена!")
        print(f"R² = {r2:.4f}")
        print(f"RMSE = {rmse:.0f} руб.")
        print(f"MAE  = {mae:.0f} руб.")
        
    return best_model


if __name__ == "__main__":
    train()
