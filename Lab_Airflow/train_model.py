from os import name
from sklearn.preprocessing import StandardScaler, PowerTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import joblib


def scale_frame(frame):
    """
    Масштабирование признаков и целевой переменной
    """
    df = frame.copy()
    
    # Целевая переменная - Price
    X = df.drop(columns=['Price'])
    y = df['Price']
    
    # Удаляем строки с пропущенными значениями
    X = X.fillna(0)
    
    # Масштабирование
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    
    X_scale = scaler.fit_transform(X.values)
    Y_scale = power_trans.fit_transform(y.values.reshape(-1, 1))
    
    return X_scale, Y_scale, power_trans, scaler

def eval_metrics(actual, pred):
    """
    Расчет метрик качества модели
    """
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train():
    """
    Основная функция обучения
    """
    # Загружаем очищенные данные
    df = pd.read_csv("./df_clear.csv")
    
    # Масштабируем данные
    X, Y, power_trans, scaler = scale_frame(df)
    
    # Разделяем на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )
    
    # Параметры для GridSearchCV
    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'l1_ratio': [0.001, 0.05, 0.1, 0.2, 0.5],
        "penalty": ["l1", "l2", "elasticnet"],
        "loss": ['squared_error', 'huber', 'epsilon_insensitive'],
        "fit_intercept": [True, False],
        "max_iter": [1000, 2000],
        "tol": [1e-3, 1e-4]
    }
    
    # Настройка MLflow
    mlflow.set_experiment("phone_price_prediction")
    
    with mlflow.start_run():
        # Создаем и обучаем модель
        lr = SGDRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv=3, n_jobs=4, scoring='r2', verbose=1)
        clf.fit(X_train, y_train.reshape(-1))
        
        # Лучшая модель
        best = clf.best_estimator_
        
        # Предсказания на валидационной выборке
        y_pred = best.predict(X_val)
        
        # Обратное преобразование цен
        y_val_original = power_trans.inverse_transform(y_val.reshape(-1, 1))
        y_pred_original = power_trans.inverse_transform(y_pred.reshape(-1, 1))
        
        # Расчет метрик
        rmse, mae, r2 = eval_metrics(y_val_original, y_pred_original)
        
        # Логирование параметров лучшей модели
        mlflow.log_param("alpha", best.alpha)
        mlflow.log_param("l1_ratio", getattr(best, 'l1_ratio', None))
        mlflow.log_param("penalty", best.penalty)
        mlflow.log_param("loss", best.loss)
        mlflow.log_param("fit_intercept", best.fit_intercept)
        mlflow.log_param("max_iter", best.max_iter)
        mlflow.log_param("tol", best.tol)
        
        # Логирование метрик
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("best_score", clf.best_score_)
        
        # Логирование дополнительной информации
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_samples", X.shape[0])
        
        # Сохраняем модель и скейлеры
        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)
        
        # Сохраняем модель локально
        with open("phone_price_model.pkl", "wb") as file:
            joblib.dump({
                'model': best,
                'scaler': scaler,
                'power_transformer': power_trans
            }, file)
        
        print(f"Модель обучена. Метрики: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")
        print(f"Лучшие параметры: {clf.best_params_}")
        
        # Анализ важности признаков (для линейной модели)
        if hasattr(best, 'coef_'):
            feature_names = df.drop(columns=['Price']).columns
            coef_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': best.coef_
            })
            coef_df['abs_coef'] = np.abs(coef_df['coefficient'])
            coef_df = coef_df.sort_values('abs_coef', ascending=False)
            
            print("\nТоп-10 важных признаков:")
            print(coef_df.head(10))
            
            # Сохраняем важность признаков
            coef_df.to_csv('feature_importance.csv', index=False)

if __name__ == "__main__":
    train()