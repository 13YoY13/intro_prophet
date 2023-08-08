
import datetime
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf

from pandas_datareader import data as pdr
from prophet import Prophet
# from fbprophet.plot import plot_plotly
# import plotly.offline as po



ticker = "^N225"

start = datetime.date(1980, 1, 1)
end = datetime.datetime.now() + datetime.timedelta(days = 1)


yf.pdr_override()
df = pdr.get_data_yahoo(ticker, start, end)
df.to_csv(ticker+'_daily_data.csv', encoding='utf8')


df["SMA25"] = df["Close"].rolling(window=25).mean()
df["SMA50"] = df["Close"].rolling(window=50).mean()
df["SMA75"] = df["Close"].rolling(window=75).mean()

df.to_csv(ticker+'_daily_data.csv', encoding='utf8')




graphStart = datetime.date(2023, 1, 1)

#計算した移動平均線を表示するための準備。
apd_day_ave  = [
                        mpf.make_addplot(df[graphStart:]['SMA25'],
                                        panel=0,color='r', width=1,alpha=0.7),
                        mpf.make_addplot(df[graphStart:]['SMA50'],
                                        panel=0,color='g', width=1,alpha=0.7),
                        mpf.make_addplot(df[graphStart:]['SMA75'],
                                        panel=0,color='b', width=1,alpha=0.7),       
                ]

mpf.plot(df[graphStart:], type='candle', mav=(25, 50, 75), datetime_format='%Y/%m/d', 
         tight_layout=False, volume=True, style='yahoo')

mpf.plot(df[graphStart:], type='candle', mav=(25, 50, 75), datetime_format='%Y/%m/d', 
         tight_layout=False, volume=True, style='yahoo', savefig=str(ticker)+'_daily.png')


df.reset_index(inplace=True)

df = df.rename(columns={'Date': 'ds', 'Adj Close': 'y'})

N = Prophet()
N.fit(df)

future = N.make_future_dataframe(periods=365)

forecast = N.predict(future)
forecast[['ds','yhat', 'yhat_lower', 'yhat_upper']]


fig1 = N.plot(forecast)
plt.show()

N.plot_components(forecast)
plt.show()

