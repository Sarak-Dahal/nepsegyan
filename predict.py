#importing libraries
import math
import numpy as np
import pandas as pd
from main import stock
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import plotly.express as px

#Retriving data source
sym='MDB'




#Todays Date and Time
today=date.today()
todaysDate = today.strftime("%Y/%m/%d")
url = 'http://www.nepalstock.com/main/stockwiseprices/index/1/Date/asc/YTo0Ont1via1qwwkc2ssfu0sy7c6qhr8e4curh64j8vglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mNi0wNinECzXshLZA5C9odmaiEfopX5DYvwMbnM4hqCMEu0sy7c6qhr8e4curh64j8vglc0pz0mVHpH3f54XshLPMwriqLdDo49bt1via1qwwkc2ssfu0sy7c6qhr8e4curh64j8vglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mglc0pz0mcr8flGUkQ6P1erDc0pz0mcr8flu0sy7c6qhr8e4curh64j8vglc0pz0m30?startDate=2005-06-05&endDate='+todaysDate+'&stock-symbol='+sym+'&_limit=10000000000'

df = pd.read_html(url,header=1)
data=df[0]
data = data.iloc[:-1]
data.to_csv('pastdata.csv')


#Visualizing closing price history
fig = px.line(data, x = 'Date', y = 'Close Price', title='Closing Price History')
fig.show()
fig.savefig('static/testplot.png')

#Creating Data Frame with only Close Column
datum = data.filter(['Close Price'])
dataset = datum.values
training_data_len = math.ceil(len(dataset) * .8)

#Scaling the data
scale = MinMaxScaler(feature_range=(0,1))
scaledData = scale.fit_transform(dataset)

#Sacled Training Data set
trainedData = scaledData[0:training_data_len, : ]

#Spliting Data in X and Y datasets
x_train=[]
y_train=[]

for i in range (60, len(trainedData)):
    x_train.append(trainedData[i-60:i, 0])
    y_train.append(trainedData[i, 0])

#x_train and y_train to numpy array
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

#Building LSTM Model
model=Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

#Compiling the model
model.compile(optimizer='adam',loss='mean_squared_error')

#Training the model
model.fit(x_train,y_train,batch_size=1,epochs=1)

#Testing Data Set

testData = scaledData[training_data_len-60:,:]

#Creating dataset x_test and y_test
x_test,y_test=[],dataset[training_data_len:,:]

for i in range(60, len(testData)):
    x_test.append(testData[i-60:i,0])

#Converting data to numpy array
x_test = np.array(x_test)

#Reshaping the data
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#Get Predicted Value
predict = model.predict(x_test)
predict = scale.inverse_transform(predict)

#Finding out RMSE lower value for actual prediction
rmse=np.sqrt(np.mean(((predict - y_test)**2)))
print(rmse)


#Creating new df to store filtered values
newDf = data.filter(['Close Price'])

#Converting last 60 days values to array
sixtyDays = newDf[-100:].values

#Scaling
sixtyDaysScaled = scale.transform(sixtyDays)


#appending data to list then taking to array and reshaping
X_test = []
X_test.append(sixtyDaysScaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

prediction = model.predict(X_test)

#undoing scaling
prediction = scale.inverse_transform(prediction)
predi=prediction







