import pandas as pd
import sqlite3


df2 = pd.read_csv('data/olist_orders_dataset.csv')
df3 = pd.read_csv('data/olist_order_items_dataset.csv')
df4 = pd.read_csv('data/olist_products_dataset.csv')

conn = sqlite3.connect(':memory:')

df2.to_sql('orders', conn, index=False, if_exists='replace')
df3.to_sql('items', conn, index=False, if_exists='replace')
df4.to_sql('products', conn, index=False, if_exists='replace')

print(df2.columns)
print(df3.columns)
print(df4.columns)


query = """
    SELECT *
    FROM orders
    INNER JOIN items t3 USING(order_id)
    INNER JOIN products t4 USING(product_id)
"""

res = pd.read_sql_query(query, conn)

conn.close()
print('Колонны таблицы после JOINа')
print(res.columns)

res.to_csv('joined_result.csv', index=False)
print('Статистика по таблице:')
pd.set_option('display.max_columns', None)
print(res.describe())

Y = res['freight_value'].copy()
Y.fillna(Y.median(), inplace=True)

print('\nТипы данных в res:')
print(res.dtypes)
