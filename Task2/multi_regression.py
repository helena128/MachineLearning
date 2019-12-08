import pandas as pd 
from sklearn.linear_model import LinearRegression


DATA = pd.read_csv('candy-data.csv', delimiter=',', index_col='competitorname')
#print(DATA)

# chose data for training
train_data = DATA.drop(['Haribo Twin Snakes', 'Hersheys Krackel'])
#print(train_data)

# set predictors and Y
X = pd.DataFrame(train_data.drop(['winpercent', 'Y'], axis = 1))
y = pd.DataFrame(train_data['winpercent'])

# train model
reg = LinearRegression().fit(X, y)

# prediction for Haribo Twin Snakes
Haribo = DATA.loc['Haribo Twin Snakes', :].to_frame().T
print('For Haribo: ', reg.predict(Haribo.drop(['winpercent', 'Y'], axis = 1)))

# prediction for Hersheys Krackel
Hersheys = DATA.loc['Hersheys Krackel', :].to_frame().T
print('For Hersheys: ', reg.predict(Hersheys.drop(['winpercent', 'Y'], axis = 1)))

# prediction for candy with given params
print('For candy with given params: ', reg.predict([[0, 1, 1, 1, 0, 0, 1, 0, 0, 0.32, 0.219]]))