def train():
    # Загружаем очищенный датасет с телефонами
    df = pd.read_csv("/home/meshkov/airflow/dags/df_clear.csv")
    
    X, Y, power_trans = scale_frame(df)
    X_train, X_val, y_train, y_val = train_test_split(X, Y,
                                                       test_size=0.3,
                                                       random_state=42)

    params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
              'l1_ratio': [0.001, 0.05, 0.01, 0.2],
              "penalty": ["l1", "l2", "elasticnet"],
              "loss": ['squared_error', 'huber', 'epsilon_insensitive'],
              "fit_intercept": [False, True],
              }

    mlflow.set_experiment("linear model phones")
    with mlflow.start_run():
        lr = SGDRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv=3, n_jobs=4)
        clf.fit(X_train, y_train.reshape(-1))
        best = clf.best_estimator_
        
        y_pred_scaled = best.predict(X_val)
        
        # Обратное преобразование
        y_price_pred = power_trans.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_true = power_trans.inverse_transform(y_val.reshape(-1, 1)).ravel()
        
        # Фильтруем только валидные (не NaN и не inf) значения
        valid_mask = np.isfinite(y_price_pred) & np.isfinite(y_true)
        
        if not valid_mask.all():
            print(f"Removed {(~valid_mask).sum()} invalid (NaN/inf) samples")
        
        if valid_mask.sum() == 0:
            raise ValueError("No valid samples left after inverse transform — check data or transformation")
        
        y_price_pred_valid = y_price_pred[valid_mask]
        y_true_valid = y_true[valid_mask]
        
        (rmse, mae, r2) = eval_metrics(y_true_valid, y_price_pred_valid)

        alpha = best.alpha
        l1_ratio = best.l1_ratio
        penalty = best.penalty
        eta0 = best.eta0
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("eta0", eta0)
        mlflow.log_param("loss", best.loss)
        mlflow.log_param("fit_intercept", best.fit_intercept)
        mlflow.log_param("epsilon", best.epsilon)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # можно также залогировать сколько процентов данных осталось
        mlflow.log_metric("valid_samples_ratio", valid_mask.mean())

        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)
        
        with open("lr_phones.pkl", "wb") as file:
            joblib.dump(best, file)
