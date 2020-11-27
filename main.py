from flask import Flask, render_template, request
from scrapper import intCi,intPc,intPcc,intSv,intTo,marketStat
import csv

#TO RUN
#set FLASK_APP=main.py
#set FLASK_DEBUG=1
#flask run

#***   FLASK   ***
app = Flask(__name__)


#***   Routing   ***
@app.route("/")
@app.route("/home")
@app.route("/index")
def index():
	return render_template('index.html',intCi=intCi,intPc=intPc,intPcc=intPcc,intSv=intSv,intTo=intTo,marketStat=marketStat)

@app.route("/layout")
def test():
	return render_template('layout.html')

@app.route("/login")
def login():
	return render_template('login.html')

@app.route("/predict")
def predict():
	return render_template('predict.html')



@app.route("/predict",methods=['POST'])
def stock():
	symbol=request.form['symbol']
	image = 'static/testplot.png'
	return render_template('predict.html',symbol=symbol,stockImage=image)





#Scraping for Today's Price Data
import pandas as pd
df = pd.read_html('https://merolagani.com/LatestMarket.aspx')
data=df[0].head(1000)
del data['Unnamed: 8']
del data['Unnamed: 9']
data.to_csv('today.csv', header=True,index=False)



@app.route("/today")
def show_tab():
	result=[]
	tData = open('today.csv')
	reader=csv.DictReader(tData)
	for row in reader:
		result.append(dict(row))
	fieldnames = [key for key in result[0].keys()]

	return render_template('today.html',result=result,fieldnames=fieldnames,len=len)
	



#***   Inserting into Database ***
#@app.route("/insertingtodb",methods=['POST'])
#def insertIntoTable():
