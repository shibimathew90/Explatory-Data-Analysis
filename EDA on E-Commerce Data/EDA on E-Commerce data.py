#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

import missingno as msno
import pandas_profiling

import gc
import datetime

color = sns.color_palette()


# In[2]:


pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 100)


# In[3]:

# The dataset can be obtained from the Kaggle - https://www.kaggle.com/carrie1/ecommerce-data?source=post_page---------------------------
df = pd.read_csv('C:/Shibi/data.csv', encoding = 'ISO-8859-1')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.rename(index = str, columns = {'InvoiceNo': 'invoice_num', 
                                  'StockCode' : 'stock_code',
                                  'Description' : 'description',
                                  'Quantity' : 'quantity',
                                  'InvoiceDate' : 'invoice_date',
                                  'UnitPrice' : 'unit_price',
                                  'CustomerID' : 'cust_id',
                                  'Country' : 'country'}, inplace = True)


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


df.isnull().sum().sort_values(ascending = False)


# In[10]:


df[df.isnull().any(axis = 1)].head()     # Check the missing row values


# In[11]:


df['invoice_date'] = pd.to_datetime(df['invoice_date'], format = '%m/%d/%Y %H:%M')


# In[12]:


df['description'] = df['description'].str.lower()


# In[13]:


df.head()


# In[14]:


df_new = df.dropna()


# In[15]:


df_new.isnull().sum().sort_values(ascending = False)


# In[16]:


df_new.info()


# In[17]:


df_new['cust_id'] = df_new['cust_id'].astype('int64')


# In[18]:


df_new.head()


# In[19]:


df_new.describe().round(2)


# In[20]:


df_new = df_new[df_new['quantity']>0]


# In[21]:


df_new.describe().round(2)


# In[22]:


df_new['amount_spent'] = df_new['quantity'] * df_new['unit_price']


# In[23]:


df_new = df_new[['invoice_num', 'invoice_date', 'stock_code', 'description', 'quantity', 'unit_price', 'amount_spent',
                'cust_id', 'country']]


# In[24]:


df_new.head()


# In[25]:


df_new.insert(loc=2, column = 'year_month', value = df_new['invoice_date'].map(lambda x : 100 * x.year + x.month))

df_new.insert(loc = 3, column = 'month', value = df_new['invoice_date'].dt.month)
df_new.insert(loc = 4, column = 'day', value = (df_new['invoice_date'].dt.dayofweek) + 1)
# Added +1 to make Monday = 1, .....Sunday = 7

df_new.insert(loc = 5, column = 'hour', value = df_new['invoice_date'].dt.hour)


# In[26]:


df_new.head()


# In[27]:


# Now we will check on how many orders where made by the customer
orders = df_new.groupby(by = ['cust_id', 'country'], as_index=False)['invoice_num'].count()


# In[28]:


orders.head()


# In[29]:


plt.subplots(figsize = (15,6))
plt.plot(orders.cust_id, orders.invoice_num)
plt.xlabel('Customer ID')
plt.ylabel('Number of orders')
plt.title('Number of orders for different Customers')
plt.show()


# In[30]:


print('Top 5 customers with most numbers of orders')
orders.sort_values(by = 'invoice_num', ascending = False).head()


# In[31]:


# Checking on how much money spent by customers

money_spent = df_new.groupby(by = ['cust_id', 'country'], as_index = False)['amount_spent'].sum()


# In[32]:


money_spent.head()


# In[33]:


plt.subplots(figsize = (15, 6))
plt.plot(money_spent.cust_id, money_spent.amount_spent)
plt.xlabel('Customer ID')
plt.ylabel('Total money spent(Dollars)')
plt.title('Total Money spent by customers')
plt.show()


# In[34]:


print('Top 5 customers who spent highest')
money_spent.sort_values(by = 'amount_spent', ascending = False).head()


# In[35]:


df_new.head()


# In[61]:


# How many orders per month

df_new.groupby('invoice_num')['year_month'].unique().value_counts().sort_index().head()


# In[53]:


ax = df_new.groupby('invoice_num')['year_month'].unique().value_counts().sort_index()
ax.plot('bar', color = 'green', figsize = (15, 6))
plt.xlabel('Month')
plt.ylabel('Number of orders')
plt.title('Number of orders per month(1th Dec 2010 - 9th Dec 2011)', fontsize = 12)
plt.xticks(range(len(ax)), ('Dec_10', 'Jan_11', 'Feb_11', 'Mar_11', 'Apr_11', 'May_11', 'Jun_11', 'July_11', 
                    'Aug_11', 'Sep_11', 'Oct_11', 'Nov_11', 'Dec_11'), rotation = 'horizontal')
plt.show()


# In[63]:


# How many orders per day
df_new.groupby('invoice_num')['day'].unique().value_counts().sort_index()


# In[66]:


ax = df_new.groupby('invoice_num')['day'].unique().value_counts().sort_index()
ax.plot('bar', color = 'green', figsize = (15,6))
plt.xlabel('Day')
plt.ylabel('Number of orders')
plt.title('Number of orders per day', fontsize = 12)
plt.xticks(range(len(ax)), ('Mon','Tue', 'Wed', 'Thur', 'Fri', 'Sun'), rotation = 'horizontal')
plt.show()


# In[78]:


# How many orders per hour
df_new.groupby('invoice_num')['hour'].unique().value_counts().iloc[:-1].sort_index()


# In[81]:


ax = df_new.groupby('invoice_num')['hour'].unique().value_counts().iloc[:-1].sort_index()
ax.plot('bar', color = 'green', figsize = (15, 6))
plt.xlabel('Hour')
plt.ylabel('Number of orders')
plt.title('Number of orders per hour', fontsize = 12)
plt.xticks(range(len(ax)),  rotation = 'horizontal')
plt.show()


# In[82]:



df_new.unit_price.describe()


# In[83]:


# Observe that we have items with unit price as 0.( These may be free items)


# In[86]:


plt.subplots(figsize=(15,6))
sns.boxplot(df_new.unit_price)
plt.show()


# In[87]:


df_free = df_new[df['unit_price']==0]


# In[88]:


df_free.head()


# In[89]:


df_free.shape


# In[90]:


df_free['year_month'].value_counts().sort_index()


# In[91]:


ax = df_free['year_month'].value_counts().sort_index()
ax.plot('bar', color = 'green', figsize = (15,6))
plt.xlabel('Month')
plt.ylabel('Frequency')
plt.title('Frequency of freebies for different months')
plt.xticks(range(len(ax)), ('Dec_10', 'Jan_11', 'Feb_11', 'Mar_11', 'Apr_11', 'May_11', 'Jul_11', 'Aug_11',
                            'Sept_11', 'Oct_11', 'Nov_11'), rotation = 'horizontal')
plt.show()


# In[92]:


# As per the above graph, we can observe that every month on average company provides 2-4 freebies to customers except
# in the month of Jun'11


# In[93]:


## Discover patterns for each country


# In[94]:


# Draw how many orders for each country


# In[100]:


df_new.groupby('country')['invoice_num'].count().sort_values(ascending = False)


# In[102]:


ax = df_new.groupby('country')['invoice_num'].count().sort_values()
ax.plot('barh', figsize = (15,8), color = 'green')
plt.xlabel('Number of orders')
plt.ylabel('Country')
plt.title('Number of orders for different countries')
plt.show()


# In[106]:


# How much money spent by each country
df_new.groupby('country')['amount_spent'].sum().sort_values(ascending=False).head()


# In[107]:


ax = df_new.groupby('country')['amount_spent'].sum().sort_values()
ax.plot('barh', figsize = (15,8), color = 'green')
plt.xlabel('Money spent')
plt.ylabel('Country')
plt.title('Money for different countries')
plt.show()


# ## Observations

# ##### 
# 1. The customer with highest number of orders is from United Kingdom. 
# 2. The customer with hightst money spent on purchases is from United Kingdom
# 3. The company receives the highest number of orders and highest money spent in United Kingdom since it's a UK Based company.
# The top 5 countries with highest number of orders are as below:
#  - United Kingdom
#  - Germany
#  - France
#  - ERIE(Ireland)
#  - Spain
#  
#    The top 5 countries with most money spent are as below:
#  - United Kingdom
#  - Netherlands
#  - ERIE(Ireland)
#  - Germany
#  - France
# 4. November 2011 has the highest sales. Also, for Dec'2011 the data is insufficient as the data is only till 9th Dec. Hence we are unable to determine the month with lowest sales.
# 5. There are no transactions on Saturday between 1st Dec 2010 - 9th Dec 2011
# 6. The number of orders received by company increases from Monday to Thrusday and decrease afterwards
# 7. The company tends to give out FREE items for purchases occasionally each month (Except June 2011)
# 8. The company receives the highest number of orders at 12pm which might possibly be because most customer have lunch time between 12:00 pm - 2:00 pm

# In[ ]:




