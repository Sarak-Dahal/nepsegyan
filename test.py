
#left to insert nepse index data to database and again bring that data for calculation of available data of NEPSE and
#reduce time for prediction

import pandas as pd
file = open('static/nepseindex.csv')
length = (len(file.readlines()))
read = pd.read_csv("static/nepseindex.csv")
file.close()
for i in range(length-1):
    print(read['Date'])
    print(read['Nepse Index Value'])
