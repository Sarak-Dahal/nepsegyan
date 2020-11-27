
import pandas as pd

x=input("Enter no of data you want: ")

df = pd.read_html("http://www.nepalstock.com/main/floorsheet/index/1/contract-no/asc/YTo1OntzOjExOiJjb250cmFjdC1ubyI7czowOiIiO3M6MTI6InN0b2NrLXN5bWJvbCI7czowOiIiO3M6NToiYnV5ZXIiO3M6MDoiIjtzOjY6InNlbGxlciI7czowOiIiO3M6NjoiX2xpbWl0IjtzOjM6IjUwMCI7fQ?contract-no=&stock-symbol=&buyer=&seller=&_limit="+x)
data=df[0]
data = data.iloc[2:]
data = data.iloc[:-3]
del data['8']
del data['9']

data.to_csv('manatwork.csv')





