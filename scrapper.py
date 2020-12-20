from urllib.request import urlopen
from bs4 import BeautifulSoup


url = 'http://www.nepalstock.com/stocklive'
html = urlopen(url).read()
soup = BeautifulSoup(html, "html.parser")


#Scraping values using Beautiful Soup and assigning to variable
marketStat=soup.find('div',class_='top_marketinfo').text
nepseCurrentIndex= soup.find('div',class_='current-index').text.replace('\n','').replace(' ','').replace(',','')
nepsePointChange = soup.find('div',class_='point-change').text.replace('\n','').replace(' ','').replace(',','')
nepsePercentChange = soup.find('div',class_='percent-change').text.replace('\n','').replace(' ','').replace(',','').replace('%','')
currentIndex= soup.find('div',class_='pull-right').find('div',class_='current-index').text.replace('\n','').replace(' ','').replace(',','')
pointChange = soup.find('div',class_='pull-right').find('div',class_='point-change').text.replace('\n','').replace(' ','').replace(',','')
percentChange = soup.find('div',class_='pull-right').find('div',class_='percent-change').text.replace('\n','').replace(' ','').replace(',','').replace('%','')
shareVolume=soup.find('span',class_='left').text.replace('\n','').replace(' ','').replace(',','').replace('ShareVolume|','')
turnover=soup.find('span',class_='right').text.replace('\n','').replace(' ','').replace(',','').replace('Turnover|','')


#Converting Values to Float
intCi=float(nepseCurrentIndex)
intPc=float(nepsePointChange)
intPcc=float(nepsePercentChange)
intSci=float(currentIndex)
intSpc=float(pointChange)
intSpcc=float(percentChange)
intSv=float(shareVolume)
intTo=float(turnover)


