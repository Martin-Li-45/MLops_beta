import pandas as pd
from os import name
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import joblib


def scale_frame(frame):
    df = frame.copy()
    X, y = df.drop(columns=['Price']), df['Price']
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scale = scaler.fit_transform(X.values)
    Y_scale = power_trans.fit_transform(y.values.reshape(-1, 1)).ravel()  # .ravel() для совместимости
    return X_scale, Y_scale, power_trans


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train():
    df = pd.read_csv("/home/meshkov/airflow/dags/df_clear.csv")
    
    X, Y, power_trans = scale_frame(df)
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )

    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'l1_ratio': [0.001, 0.05, 0.01, 0.2],
        "penalty": ["l1", "l2", "elasticnet"],
        "loss": ['squared_error', 'huber', 'epsilon_insensitive'],
        "fit_intercept": [False, True],
    }

    mlflow.set_experiment("linear model phones")
    
    with mlflow.start_run():
        lr = SGDRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv=3, n_jobs=4)
        clf.fit(X_train, y_train)
        
        best = clf.best_estimator_
        
        y_pred_scaled = best.predict(X_val)
        
        # Обратное преобразование
        y_price_pred = power_trans.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).ravel()
        
        y_true = power_trans.inverse_transform(
            y_val.reshape(-1, 1)
        ).ravel()
        
        # Убираем NaN / inf
        valid_mask = np.isfinite(y_price_pred) & np.isfinite(y_true)
        
        if not valid_mask.all():
            print(f"WARNING: removed {(~valid_mask).sum()} invalid (NaN/Inf) samples")
        
        if valid_mask.sum() == 0:
            print("ERROR: no valid samples left after inverse transform")
            # можно залогировать что-то и выйти, но пока просто продолжим с пустыми метриками
            rmse, mae, r2 = np.nan, np.nan, np.nan
        else:
            y_price_pred_valid = y_price_pred[valid_mask]
            y_true_valid = y_true[valid_mask]
            rmse, mae, r2 = eval_metrics(y_true_valid, y_price_pred_valid)

        # Логируем параметры
        mlflow.log_param("alpha", best.alpha)
        mlflow.log_param("l1_ratio", best.l1_ratio)
        mlflow.log_param("penalty", best.penalty)
        mlflow.log_param("eta0", best.eta0)
        mlflow.log_param("loss", best.loss)
        mlflow.log_param("fit_intercept", best.fit_intercept)
        mlflow.log_param("epsilon", best.epsilon)

        # Логируем метрики (даже если nan)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("valid_samples_ratio", valid_mask.mean() if valid_mask.any() else 0)

        # Логируем модель
        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)
        
        with open("lr_phones.pkl", "wb") as file:
            joblib.dump(best, file)


if __name__ == "__main__":
    train()
