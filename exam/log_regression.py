import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from statistics import mean
from sklearn.neighbors import KNeighborsClassifier
from numpy import ravel

DATA = pd.read_csv('report.csv', delimiter=',')

print('Mean: ', mean(DATA['MIP']))

column_names_to_not_normalize = ['TARGET']
column_names_to_normalize = [x for x in list(DATA) if x not in column_names_to_not_normalize ]

x = DATA[column_names_to_normalize].values
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = DATA.index)
DATA[column_names_to_normalize] = df_temp
print(DATA)

# Mean
print('Mean for MIPS: ', mean(DATA['MIP']))

# Logistic regression
X = pd.DataFrame(DATA.drop(['TARGET'], 1))
y = pd.DataFrame(DATA['TARGET'])

reg = LogisticRegression(random_state=2019, solver='lbfgs').fit(X, y.values.ravel())
train_star = [0.157, 0.311, 0.676, 0.586, 0.307, 0.848, 0.673, 0.64]

print('Prediction: ', reg.predict_proba([train_star])[:,1])

# K-nearest neighbors
classifier = KNeighborsClassifier(n_neighbors=5)
#print(ravel(y))
classifier.fit(X, ravel(y))

print('Distance: ', classifier.kneighbors(return_distance = True, X = [train_star]))
print('Class: ', classifier.predict([train_star]))