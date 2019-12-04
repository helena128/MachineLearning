import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('candy-data.csv')

X = dataset[['chocolate','fruity','caramel',
			'peanutyalmondy','nougat','crispedricewafer',
			'hard','bar','pluribus','sugarpercent','pricepercent']].values
y = dataset['winpercent']
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
df = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
df1 = df.head(100)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)
print(model.coef_)
coef = model.coef_
prediction = 0 * coef[0] + 1 * coef[1] + 1 * coef[2] + 1 * coef[3] + \
			0 * coef[4] + 0 * coef[5] + 1 * coef[6] + 0 * coef[7] + 0 * coef[8] + \
			0.32 * coef[9] + 0.219 * coef[10]
print('Prediction: ', prediction)