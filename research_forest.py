import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time


res = pd.read_csv('joined_result.csv')


Y = res['freight_value'].copy()
Y.fillna(Y.median(), inplace=True)
col = ['order_id', 'payment_sequential', 'payment_type',
       'payment_installments', 'payment_value', 'customer_id', 'order_status',
       'order_item_id', 'product_id',
       'seller_id', 'price',
       'product_category_name', 'product_name_lenght',
       'product_description_lenght', 'product_photos_qty', 'product_weight_g',
       'product_length_cm', 'product_height_cm', 'product_width_cm']


X = pd.DataFrame()
for i in col:
    if res[i].dtype.name != 'str':
        X[i] = res[i].copy()
        X[i] = X[i].fillna(X[i].median())
    else:
        X[i] = pd.factorize(res[i])[0]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

text = ['n_estimators \t time \t r2_score \t mean_absolute_error ']
for n in range(10, 510, 10):
    t1 = time.perf_counter()
    model1 = RandomForestRegressor(n_estimators=n, random_state=13)
    model1.fit(X_train, y_train)
    y1 = model1.predict(X_test)
    text.append('\t'.join([str(n), str(r2_score(y1, y_test)), str(time.perf_counter() - t1),
                          str(mean_absolute_error(y1, y_test))]))
    print(f'\n n_estimators: \n {n}, \n r2 score:')
    print(r2_score(model1.predict(X_test), y_test))

try:
    with open('n_research.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(text))
except Exception as e:
    print(e)
    print('\n'.join(text))
#
# text = []
# for n in range(1, 31):
#     model2 = RandomForestRegressor(max_depth=n, random_state=13)
#     model2.fit(X_train, y_train)
#     text.append(' '.join([str(n), str(r2_score(model2.predict(X_test), y_test))]))
#     print(f'\n max_depth: \n {n}, \n r2 score:')
#     print(r2_score(model2.predict(X_test), y_test))
#
# try:
#     with open('max_depth_research.txt', 'w', encoding='utf-8') as f:
#         f.write('\n'.join(text))
# except Exception as e:
#     print(e)
#     print('\n'.join(text))
#
# text = []
# for n in range(2, 31):
#     model3 = RandomForestRegressor(min_samples_split=n, random_state=13)
#     model3.fit(X_train, y_train)
#     text.append(' '.join([str(n), str(r2_score(model3.predict(X_test), y_test))]))
#     print(f'\n min_samples_split: \n {n}, \n r2 score:')
#     print(r2_score(model3.predict(X_test), y_test))
#
# try:
#     with open('min_samples_split_research.txt', 'w', encoding='utf-8') as f:
#         f.write('\n'.join(text))
# except Exception as e:
#     print(e)
#     print('\n'.join(text))