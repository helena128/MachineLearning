import pandas as pd
import matplotlib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

NUM_OF_LINES = 520
IS_SICK = 0
TRAIN_DATA_PERCENT = 0.8
MAX_LEAF_NODES = 25
MIN_SAMPLES_LEAF = 15
RANDOM_STATE = 2020
PATIENT_NUMBERS = [719, 739, 748, 734]

df = pd.read_csv('diabetes.csv') 
task_data = df.head(NUM_OF_LINES)
print('Patient is not sick: ', len(task_data[task_data['Outcome'] == IS_SICK]))

train = task_data.head(int(len(task_data)*TRAIN_DATA_PERCENT))
test = task_data.tail(int(len(task_data)*(1 - TRAIN_DATA_PERCENT)))

# Set x and y
features = list(train.columns[:8])
x = train[features]
y = train['Outcome']

tree = DecisionTreeClassifier(criterion='entropy',
                              min_samples_leaf=MIN_SAMPLES_LEAF,
                              max_leaf_nodes=MAX_LEAF_NODES,
                              random_state=RANDOM_STATE)
clf=tree.fit(x, y)

print('Tree depth: ', clf.tree_.max_depth)

# Show tree
columns = list(x.columns)
export_graphviz(clf, out_file='tree.dot', 
                feature_names=columns,
                class_names=['0', '1'],
                rounded = True, proportion = False, 
                precision = 4, filled = True, label='all')

with open('tree.dot') as f:
    dot_graph = f.read()

graph = graphviz.Source(dot_graph)
graph.format = 'png'
graph.render('dtree_render',view=True)

# Test model
features = list(test.columns[:8])
x = test[features]
y_true = test['Outcome']
y_pred = clf.predict(x)
print('Accuracy: ', accuracy_score(y_true, y_pred))
print('F1: ', f1_score(y_true, y_pred, average='macro'))

for patient in PATIENT_NUMBERS:
	pr = clf.predict([df.loc[patient, features].tolist()])[0]
	print('Patient: ', patient, ' prediction: ', pr)