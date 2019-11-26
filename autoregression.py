import pandas as pd
import matplotlib.pylab as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import r2_score

plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use('ggplot')

sales_data = pd.read_csv('retail_sales.csv')
sales_data['date'] = pd.to_datetime(sales_data['date'])
sales_data.set_index('date', inplace=True)

sales_data.plot()
pd.plotting.lag_plot(sales_data['sales'])
pd.plotting.autocorrelation_plot(sales_data['sales'])

# sales_data['sales'].corr(sales_data['sales'].shift(12))
# decomposed = seasonal_decompose(sales_data['sales'], model='additive')
# x = decomposed.plot()

sales_data['stationary']=sales_data['sales'].diff()
#creating model
# create train/test datasets
X = sales_data['stationary'].dropna()
train_data = X[1:len(X) - 12]
test_data = X[X[len(X) - 12:]]

# train the autoregression model
model = AR(train_data)
model_fitted = model.fit()

print('The lag value chose is: %s' % model_fitted.k_ar)
print('The coefficients of the model are:\n %s' % model_fitted.params)

predictions = model_fitted.predict(
    start=len(train_data),
    end=len(train_data) + len(test_data) - 1,
    dynamic=False)

# create a comparison dataframe
compare_df = pd.concat(
    [sales_data['stationary'].tail(12),
     predictions], axis=1).rename(
    columns={'stationary': 'actual', 0: 'predicted'})

# plot the two values
compare_df.plot()


#r2 score
r2 = r2_score(sales_data['stationary'].tail(12), predictions)
r2 = r2_score(sales_data['stationary'].tail(12), predictions)
print(r2)


plt.show()