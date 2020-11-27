import pandas as pd
import numpy as np

df = pd.read_html('https://merolagani.com/LatestMarket.aspx')
data=df[0].head(1000)
conditions = [
	(data['% Change'] == 0),
    (data['% Change'] > 0),
	(data['% Change'] < 0)
	]
conditionsValue=['=','+','-']
data['Index'] = np.select(conditions,conditionsValue)
data.head()


