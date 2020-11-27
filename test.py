
url='http://www.nepalstock.com/main/floorsheet/index/1/stock-symbol/asc/YTo1OntzOjExOiJjb250cmFjdC1ubyI7czowOiIiO3M6MTI6InN0b2NrLXN5bWJvbCI7czowOiIiO3M6NToiYnV5ZXIiO3M6MDoiIjtzOjY6InNlbGxlciI7czowOiIiO3M6NjoiX2xpbWl0IjtzOjM6IjUwMCI7fQ?contract-no=&stock-symbol=&buyer=&seller=&_limit=20000'
#Scraping for Today's Price Data
import pandas as pd
df = pd.read_html(url)
data=df[0].head(20000)
data.to_csv('test.csv', header=True,index=False)