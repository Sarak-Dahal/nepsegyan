import mysql.connector
db = mysql.connector.connect(
    host='localhost',
    user='root',
    passwd='',
    database='stockMarket'
    )
#creating cursor object
myCursor = db.cursor()


#Creating Tables in Database
#using smallint (-32k to 32k) and varchar which is required for it and for less resource to consume

#mycursor.execute("Create table Company_Codes (code smallint UNSIGNED NOT NULL PRIMARY KEY, name varchar(10) NOT NULL)")

file = open("compCodes.txt","r")

for x in file.readlines():
    eCode = (x.split(' ')[0])
    eComp = (x.split(' ')[1])
    myCursor.execute("INSERT INTO company_codes (code,name) VALUES (%s,%s)", (eCode,eComp))

db.commit()
db.close()
