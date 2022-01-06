#!/usr/bin/env python
# coding: utf-8

# In[1]:


## !pip install yfinance --upgrade --no-cache-dir 


# ### Importing all the necessary libraries which are required to implement

# In[2]:


# LinearRegression is a machine learning library for linear regression
from sklearn.linear_model import LinearRegression

# pandas and numpy are used for data manipulation
import pandas as pd
import numpy as np

# matplotlib and seaborn are used for plotting graphs
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-darkgrid')
import yfinance as yf


# ### Reading the past 10 years of daily Gold ETF price data and store it in DataFrame
# #### Plotting the CLOSE prices of Gold

# In[27]:


# Read data
Df = yf.download('GLD', '2006-01-01', '2022-01-03', auto_adjust=True)

# Only keep close columns
Df = Df[['Close']]

# Drop rows with missing values
Df = Df.dropna()

# Plot the closing price of GLD
Df.Close.plot(figsize=(10, 7),color='r')
plt.ylabel("Gold ETF Prices")
plt.title("Gold ETF Price Series")
plt.show()


# ### Viewing the Dataframe collected from ticker/index

# In[28]:


Df


# ### Some EDA on Data Collected

# In[29]:


#Create histogram with density plot
import seaborn as sns
sns.distplot(Df, hist=True, kde=True,
             bins=20,              
             color = 'blue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2})


# In[30]:


# yearly Box-Plot on Data
single_year = Df
groups = single_year.groupby(pd.Grouper(freq='A'))
years = pd.concat([pd.DataFrame(x[1].values) for x in groups], axis=1)
years = pd.DataFrame(years)
years.boxplot(figsize=(25,15))
plt.show()


# In[31]:


# create a scatter plot
from pandas.plotting import lag_plot
plt.figure(figsize=(20,10))
lag_plot(Df)
plt.show()


# In[32]:


# Creating Autocorrelation plot
pd.plotting.autocorrelation_plot(Df)
plt.show()


# ### Explantory Variable = variable which is manipulated to determind the next day price of commodity
# ### Dependant Valriable = which depends upon value of explanatory variable (Gold Price in this case)
# #### S3 = moving avg of last 3 days
# #### S9 = moving avg of last 15 Days
# #### y where we will store our Gold Price

# In[33]:


# Define explanatory variables
Df['S3'] = Df['Close'].rolling(window=3).mean()
Df['S15'] = Df['Close'].rolling(window=15).mean()
Df['next_day_price'] = Df['Close'].shift(-1)

Df = Df.dropna()
X = Df[['S3', 'S15']]

# Define dependent variable
y = Df['next_day_price']


# In[34]:


Df


# ### Splitting Data for Test & Training. Test Data to be used for Linear Regression Model
# #### - First 80% of the data is used for training and remaining data for testing

# In[35]:


# Split the data into train and test dataset
t = .8
t = int(t*len(Df))

# Train dataset
X_train = X[:t]
y_train = y[:t]

# Test dataset
X_test = X[t:]
y_test = y[t:]


# ### Y = m1 * X1 + m2 * X2 + C
# ### Gold ETF price = m1 * 3 days moving average + m2 * 15 days moving average + c

# In[36]:


# Create a linear regression model
linear = LinearRegression().fit(X_train, y_train)
print("Linear Regression model")
print("Gold ETF Price (y) = %.2f * 3 Days Moving Average (x1) + %.2f * 15 Days Moving Average (x2) + %.2f (constant)" % (linear.coef_[0], linear.coef_[1], linear.intercept_))


# ### Predict the Gold ETF prices

# In[37]:


predicted_price = linear.predict(X_test)
predicted_price = pd.DataFrame(
    predicted_price, index=y_test.index, columns=['price'])
predicted_price.plot(figsize=(20, 14))
y_test.plot()
plt.legend(['predicted_price', 'actual_price'])
plt.ylabel("Gold ETF Price")
plt.show()


# ### Computing the goodness of the fit

# In[38]:


# R square
r2_score = linear.score(X[t:], y[t:])*100
float("{0:.2f}".format(r2_score))


# ### Plotting cumulative returns

# In[39]:


gold = pd.DataFrame()

gold['price'] = Df[t:]['Close']
gold['predicted_price_next_day'] = predicted_price
gold['actual_price_next_day'] = y_test
gold['gold_returns'] = gold['price'].pct_change().shift(-1)

gold['signal'] = np.where(gold.predicted_price_next_day.shift(1) < gold.predicted_price_next_day,1,0)

gold['strategy_returns'] = gold.signal * gold['gold_returns']
((gold['strategy_returns']+1).cumprod()).plot(figsize=(20,14),color='g')
plt.ylabel('Cumulative Returns')
plt.show()


# #### A 'buy' trading signal represented by “1” when the next day’s predicted price is more than the current day predicted price. No position is taken otherwise

# In[40]:


gold


# ### Calculating Sharpe Ratio
# #### Sharpe ratio is the measure of risk-adjusted return of a financial portfolio.

# In[41]:


sharpe = gold['strategy_returns'].mean()/gold['strategy_returns'].std()*(252**0.5)
'Sharpe Ratio %.2f' % (sharpe)


# ### Predict daily moves as per signal created

# In[42]:


# import datetime and get today's date
import datetime as dt
current_date = dt.datetime.now()

# Get the data
data = yf.download('GLD', '2007-01-01', current_date, auto_adjust=True)
data['S3'] = data['Close'].rolling(window=3).mean()
data['S15'] = data['Close'].rolling(window=9).mean()
data = data.dropna()

# Forecast the price
data['predicted_gold_price'] = linear.predict(data[['S3', 'S15']])
data['signal'] = np.where(data.predicted_gold_price.shift(1) < data.predicted_gold_price,"Buy","No Position")

# Print the forecast
data.tail(1)[['signal','predicted_gold_price']].T


# ## Model Training: Random Forest Regressor

# In[43]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# In[44]:


regressor = RandomForestRegressor(n_estimators=100)


# In[45]:


regressor.fit(X_train,y_train)


# ### Model Evaluation

# In[46]:


test_data_prediction = regressor.predict(X_test)


# In[47]:


error_score = metrics.r2_score(y_test, test_data_prediction)
print("R squared error : ", error_score)


# ### Comparing the Actual Values and the Predicted Values

# In[48]:


y_test = list(y_test)


# In[49]:


plt.plot(y_test, color='blue', label = 'Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()


# In[ ]:





# from sklearn import model_selection
# # prepare configuration for cross validation test harness
# seed = 7
# X1 = Df[['S3', 'S15']]
# 
# # Define dependent variable
# Y = Df['next_day_price']
# # prepare models
# models = []
# models.append(('LR', linear))
# models.append(('RFR', regressor))
# 
# # evaluate each model in turn
# results = []
# names = []
# scoring = 'accuracy'
# for name, model in models:
# 	kfold = model_selection.KFold(n_splits=10)
# 	cv_results = model_selection.cross_val_score(model, X1, Y, cv=kfold, scoring=scoring)
# 	results.append(cv_results)
# 	names.append(name)
# 	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
# 	print(msg)
# # boxplot algorithm comparison
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

# In[ ]:





# In[ ]:




