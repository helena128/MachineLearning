import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


DATA = pd.read_csv('candy-data.csv', delimiter=',', index_col='competitorname')

train_data = DATA.drop(['Mike & Ike', 'Now & Later', 'Skittles original'])

X = pd.DataFrame(train_data.drop(['winpercent', 'Y'], axis=1))
y = pd.DataFrame(train_data['Y'])

reg = LogisticRegression(random_state=2019, solver='lbfgs').fit(X, y.values.ravel())


test_data = pd.read_csv('candy-test.csv', delimiter=',', index_col='competitorname')
X_test = pd.DataFrame(test_data.drop(['Y'], axis=1))
Y_pred = reg.predict(X_test)
Y_true = (test_data['Y'].to_frame().T).values.ravel()

print('Twix prediction: ', reg.predict_proba([[0,1,0,0,0,0,0,0,0,0.73543,0]])[:,1])
print('Tootsie: ', reg.predict_proba([[1,0,0,0,0,0,0,0,0,0.31299999,0.51099998]])[:,1])

# AUC
fpr, tpr, thresholds = metrics.roc_curve(Y_true, Y_pred)
print('AUC: ', metrics.auc(fpr, tpr))

# Recall
print('Recall: ', metrics.recall_score(Y_true, Y_pred))

# Precision
print('Precision: ', metrics.precision_score(Y_true, Y_pred))