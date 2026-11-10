import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import sqlite3
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import time


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

print(f'\n\n\nРегрессия с помощью случайного леса:')
s_m1 = time.perf_counter()
model_rf = RandomForestRegressor(n_estimators=200, random_state=13)
model_rf.fit(X_train, y_train)
print(f'Время обучения Random Forest: {round(time.perf_counter() - s_m1, 2)} секунд')

importances = model_rf.feature_importances_
indices = np.argsort(importances)[::-1]
ar_f = []
for f, idx in enumerate(indices):
    ar_f.append([round(importances[idx], 4), col[idx]])
print('Значимость признаков:')
ar_f.sort(reverse=True)
print(ar_f)

y1 = model_rf.predict(X_test)
print(y1)
print(y_test)
print('\n Метрики качества случайного леса:')
print(f'r2 score: {r2_score(y1, y_test)}')
print(f'MSE: {mean_squared_error(y1, y_test)}')
print(f'RMSE: {np.sqrt(mean_squared_error(y1, y_test))}')
print(f'MAE: {mean_absolute_error(y1, y_test)}')

print('\n\n\nРегрессия с помощью Gradient Boosting:')
s_m2 = time.perf_counter()
model_gb = GradientBoostingRegressor(n_estimators=384, learning_rate=0.01,
                                     max_depth=6, random_state=13, min_samples_leaf=5)
model_gb.fit(X_train, y_train)
print(f'Время обучения Gradient Boosting: {round(time.perf_counter() - s_m2, 2)} секунд')
y_pred_gb = model_gb.predict(X_test)
# Оценка качества
print('\nМетрики качества Gradient Boosting:')
print(f'r2 score: {r2_score(y_test, y_pred_gb)}')
print(f'MSE: {mean_squared_error(y_test, y_pred_gb)}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_gb))}')
print(f'MAE: {mean_absolute_error(y_test, y_pred_gb)}')


print('\n\n\nРегрессия с помощью LightGBM:')
s_m3 = time.perf_counter()
model_lgb = LGBMRegressor(
    n_estimators=2000, learning_rate=0.03, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8,
    random_state=13, n_jobs=-1, verbose=-1
)
model_lgb.fit(X_train, y_train)
print(f'Время обучения LightGBM: {round(time.perf_counter() - s_m3, 2)} секунд')
y_pred_lgb = model_lgb.predict(X_test)
# Оценка качества
print('\nМетрики качества LightGBM:')
print(f'r2 score: {r2_score(y_test, y_pred_lgb)}')
print(f'MSE: {mean_squared_error(y_test, y_pred_lgb)}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lgb))}')
print(f'MAE: {mean_absolute_error(y_test, y_pred_lgb)}')


print('\n\n\nРегрессия с помощью XGBoost:')
s_m4 = time.perf_counter()
model_xgb = XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=8)
model_xgb.fit(X_train, y_train)
print(f'Время обучения XGBoost: {round(time.perf_counter() - s_m4, 2)} секунд')
y_pred_xgb = model_xgb.predict(X_test)
# Оценка качества
print('\nМетрики качества XGBoost:')
print(f'r2 score: {r2_score(y_test, y_pred_xgb)}')
print(f'MSE: {mean_squared_error(y_test, y_pred_xgb)}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_xgb))}')
print(f'MAE: {mean_absolute_error(y_test, y_pred_xgb)}')


print('\n\n\nРегрессия с помощью CatBoost:')
s_m5 = time.perf_counter()
model_cgb = CatBoostRegressor(n_estimators=2000, learning_rate=0.03, max_depth=8)
model_cgb.fit(X_train, y_train)
print(f'Время обучения CatBoost: {round(time.perf_counter() - s_m5, 2)} секунд')
y_pred_cgb = model_cgb.predict(X_test)
# Оценка качества
print('\n Метрики качества CatBoost:')
print(f'r2 score: {r2_score(y_test, y_pred_cgb)}')
print(f'MSE: {mean_squared_error(y_test, y_pred_cgb)}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_cgb))}')
print(f'MAE: {mean_absolute_error(y_test, y_pred_cgb)}')


# Добавить коэффициент корреляции