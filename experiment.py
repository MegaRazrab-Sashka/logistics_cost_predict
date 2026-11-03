w = []
f1 = open('data/olist_customers_dataset.csv', encoding='utf-8').read().split('\n')
n1 = 'customers'
f2 = open('data/olist_geolocation_dataset.csv', encoding='utf-8').read().split('\n')
n2 = 'geolocation'
f3 = open('data/olist_order_items_dataset.csv', encoding='utf-8').read().split('\n')
n3 = 'order_items'
f4 = open('data/olist_order_payments_dataset.csv', encoding='utf-8').read().split('\n')
n4 = 'order_payments'
f5 = open('data/olist_order_reviews_dataset.csv', encoding='utf-8').read().split('\n')
n5 = 'order_reviews'
f6 = open('data/olist_orders_dataset.csv', encoding='utf-8').read().split('\n')
n6 = 'orders'
f7 = open('data/olist_products_dataset.csv', encoding='utf-8').read().split('\n')
n7 = 'products'
f8 = open('data/olist_sellers_dataset.csv', encoding='utf-8').read().split('\n')
n8 = 'sellers'
f9 = open('data/product_category_name_translation.csv', encoding='utf-8').read().split('\n')
n9 = 'product_category'
for x, y in zip([n1, n2, n3, n4, n5, n6, n7, n8, n9], [f1, f2, f3, f4, f5, f6, f7, f8, f9]):
    w.append(x + ': ' + y[0] + '\n')
for i in w:
    print(i)