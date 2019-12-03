import csv
import statistics 

with open('salary_and_population.csv', 'r') as file:
    csv_reader = csv.reader(file, delimiter=',')
    line_count = 0
    stat_array = []
    for row in csv_reader:
        if line_count == 0:
            #print('Skipping first row')
            line_count += 1
        elif row[0] != 'Вологодская область' and row[0] != 'Мурманская обл':
            stat_array.append(int(row[2]))
            line_count += 1
    #print(len(stat_array))
    print('Mean: ', number(statistics.mean(stat_array), 3))
    print('Median: ', number(statistics.median(stat_array), 3))
    print('Variance: ', number(statistics.pvariance(stat_array), 3))
    print('Std dev: ', number(statistics.pstdev(stat_array), 3))
