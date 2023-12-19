import pandas as pd
import yfinance as yf
import datetime 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

today = datetime.date.today()
Hundred_days_ago = today - datetime.timedelta(days =100)
hundred_days_ago=Hundred_days_ago
hundred_days_ago+=datetime.timedelta(days=1)

Digital_currency_list=["BTC-USD" ,"ETH-USD", "USDT-USD","BNB-USD","XRP-USD","SOL-USD","USDC-USD","ADA-USD",
                       "STETH-USD","AVAX-USD","DOGE-USD","WTRX-USD","TRX-USD","DOT-USD","LINK-USD","MATIC-USD",
                       "TON11419-USD","WBTC-USD","SHIB-USD","DAI-USD","LTC-USD","BCH-USD","ATOM-USD", "UNI7083-USD",
                       "XLM-USD","OKB-USD","LEO-USD","ICP-USD","XMR-USD","ETC-USD","IMX10603-USD","WHBAR-USD",
                       "HBAR-USD","INJ-USD","KAS-USD","BXC5168-USD","APT21794-USD","CRO-USD","TUSD-USD","NEAR-USD"]
#"WEOS-USD"
#placing dates since 100days ago in columns title
date_list=[]
while hundred_days_ago<=today:
    date_list.append(hundred_days_ago.strftime("%Y-%m-%d"))
    hundred_days_ago+=datetime.timedelta(days=1)
    final_data_frame=pd.DataFrame(columns=())

final_data_frame=pd.DataFrame(columns=[date_list] , index=[Digital_currency_list])

#adding each digital currency close column as rows to dataframe
for i in Digital_currency_list:
    data = pd.DataFrame(yf.download( i, period='max' , start=Hundred_days_ago , end=today ))
    close_column_list=data['Close'].tolist()
    final_data_frame.loc[i] = close_column_list
    
final_data_frame.to_excel("final_data_frame.xlsx")

#kmean
row_mean=final_data_frame.mean(axis=1)
row_mean = row_mean.astype(float)
sorted_currencies = row_mean.sort_values(ascending=False)
sorted_currencies.to_excel("sorted_currencies.xlsx")
df = pd.read_excel('sorted_currencies.xlsx')
list=df.iloc[:, 1].tolist()

data = pd.DataFrame({"Feature1":list,
                     "Feature2":list})

num_clusters = 3
features = data.values
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(features)
labels = kmeans.labels_
data['Cluster'] = labels
sns.scatterplot(data=data, x='Feature1', y='Feature2',s=70, hue='Cluster', palette='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='red', marker='X', label='cluster')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
