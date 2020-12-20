import mysql.connector
db = mysql.connector.connect(
    host='localhost',
    user='root',
    passwd='sarak123',
    database='stockMarket'
    )
#creating cursor object
myCursor = db.cursor()


#Creating Tables in Database
#using smallint (-32k to 32k) and varchar which is required for it and for less resource to consume

#myCursor.execute("Create table Company_Codes (code smallint UNSIGNED NOT NULL PRIMARY KEY, name varchar(10) NOT NULL)")

#file = open("compCodes.txt","r")

#for x in file.readlines():
 #   eCode = (x.split(' ')[0])
  #  eComp = (x.split(' ')[1])
   # myCursor.execute("INSERT INTO company_codes (code,name) VALUES (%s,%s)", (eCode,eComp))


#myCursor.execute("Create table Nepse_Index (tdate date NOT NULL PRIMARY KEY, nepse float(7,4) NOT NULL)")
#file = open("nepseindex.csv","r")
#for x in file.readlines():
 #   insertDate =(x[1])
  #  insertData =(x[2])
   # myCursor.execute("INSERT INTO Nepse_Index (tdate,nepse) VALUES (%s,%s)", (insertDate, insertData))
#file.close()


#Creating table for registering users
myCursor.execute("Create table register (name tinytext NOT NULL,number varchar(10) NOT NULL PRIMARY KEY,password varchar(32) NOT NULL)")


db.commit()


db.close()
