import numpy as np
from sklearn.linear_model import LinearRegression
import csv
import statistics as stat

x = []
y = []
first_row = True
with open('number_of_people_in_line.csv', 'r') as rf:
    reader = csv.reader(rf, delimiter=',')
    for row in reader:
        if not first_row:
            x.append(int(row[1]))
            y.append(int(row[2]))
        else:
            first_row = False
    #print('x: ', x, ' y: ', y)
    print('X_avg: ', stat.mean(x))
    print('Y_avg: ', stat.mean(y))

    # provide data 
    x_arr = np.array(x, dtype='float64').reshape((-1, 1))
    y_arr = np.array(y, dtype='float64')
    #print('x: ', x_arr, ' y: ', y_arr)

    # create model and fit it
    model = LinearRegression()
    model.fit(x_arr, y_arr)
    r_sq = model.score(x_arr, y_arr)
    print('Coefficient of determination: ', r_sq)
    print('Intercept: ', model.intercept_)
    print('Slope: ', model.coef_)

