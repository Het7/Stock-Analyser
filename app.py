import streamlit as st
import tweepy
import pandas as pd
import numpy as np
import re
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from textblob import TextBlob
from wordcloud import WordCloud
from datetime import date
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go


consumerKey = "OLulNWGyTIpbHyL1tU1HyxMJ7"
consumerSecret = "rFI74JDu9PqQT4emDsM6pE6NsKgc8kSMCZA8J2hkapwVEgWCeW"
accessToken = "1305386993124290560-9GdZT5w0pZxOCYxCd8Izokg8WLfdmT"
accessTokenSecret = "vdXTRcB9gOocQEK2KQ6uhiJsODJzozOHwthmJimHGhw9Z"



authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret) 
authenticate.set_access_token(accessToken, accessTokenSecret) 
api = tweepy.API(authenticate, wait_on_rate_limit = True)


def app():


	activities=["Tweet Analyzer","Generate Twitter Data","Data"]
	choice = st.sidebar.selectbox("Select Your Activity",activities)

	def cleanTxt(text):
	 text = re.sub('@[A-Za-z0–9]+', '', text) 
	 text = re.sub('#', '', text) 
	 text = re.sub('RT[\s]+', '', text) 
	 text = re.sub('https?:\/\/\S+', '', text) 
	 return text
	

	if choice=="Tweet Analyzer":
		st.title("Twitter Web Scrapper")
		raw_text = st.text_area("Enter the exact twitter handle of the Personality (without @)")


		if st.button("Analyze"):
				
				def Plot_Analysis():

					posts = api.user_timeline(screen_name=raw_text, count = 100, lang ="en", tweet_mode="extended")
					df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])

					df['Tweets'] = df['Tweets'].apply(cleanTxt)


					def getSubjectivity(text):
					   return TextBlob(text).sentiment.subjectivity
					
					def getPolarity(text):
					   return  TextBlob(text).sentiment.polarity
					
					df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
					df['Polarity'] = df['Tweets'].apply(getPolarity)


					def getAnalysis(score):
					  if score < 0:
					    return 'Negative'
					  elif score == 0:
					    return 'Neutral'
					  else:
					    return 'Positive'
					    
					df['Analysis'] = df['Polarity'].apply(getAnalysis)

					return df

				df = Plot_Analysis()

				st.write(sns.countplot(x=df["Analysis"],data=df))
				st.pyplot(use_container_width=True)

	elif choice == "Generate Twitter Data":

		user_name = st.text_area("*Enter the exact twitter handle of the Personality (without @)*")

		def get_data(user_name):

			posts = api.user_timeline(screen_name=user_name, count = 100, lang ="en", tweet_mode="extended")
			df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
			df['Tweets'] = df['Tweets'].apply(cleanTxt)


			def getSubjectivity(text):
				return TextBlob(text).sentiment.subjectivity

			
			def getPolarity(text):
				return  TextBlob(text).sentiment.polarity

			df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
			df['Polarity'] = df['Tweets'].apply(getPolarity)

			def getAnalysis(score):
				if score < 0:
					return 'Negative'

				elif score == 0:
					return 'Neutral'
				else:
					return 'Positive'

			df['Analysis'] = df['Polarity'].apply(getAnalysis)
			return df

		if st.button("Show Data"):
			st.success("Fetching Last 100 Tweets")
			df=get_data(user_name)
			st.write(df)

	else:

		START = "2015-01-01"
		TODAY = date.today().strftime("%Y-%m-%d")

		st.title('Stock Forecast App')
		st.markdown("""
		Make educated investing decisions in the stock market by browsing through stocks 
		in the S&P 500 to see their performance in the past decade 
		and how they are expected to do in the coming 5 years.\n
		""")

		st.sidebar.header('Pick a Stock from the S&P 500')


		def load_stocks():
			url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
			table = pd.read_html(url, header = 0)
			stocks = table[0]['Symbol'].tolist()
			return stocks


		selected_stock = st.sidebar.selectbox('Pick a Stock', load_stocks())

		n_years = st.sidebar.selectbox('Timeframe:', list(range(1,4)))
		period = n_years * 365


		@st.cache
		def load_data(ticker):
			data = yf.download(ticker, START, TODAY)
			data.reset_index(inplace=True)
			return data

		data = load_data(selected_stock)

		
		def plot_raw_data():
			fig = go.Figure()
			open_price = st.sidebar.checkbox('Stock Opening Price')
			close_price = st.sidebar.checkbox('Stock Closing Price')

			if (open_price):
				fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
				fig.layout.update(title_text='Raw Data', xaxis_rangeslider_visible=True)
				st.plotly_chart(fig)
				
			
			if (close_price):
				fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
				fig.layout.update(title_text='Raw Data', xaxis_rangeslider_visible=True)
				st.plotly_chart(fig)
				
			
	
		plot_raw_data()

		
		data_frame = data[['Date','Close']]
		data_frame = data_frame.rename(columns={"Date": "ds", "Close": "y"})

		info = Prophet()
		info.fit(data_frame)
		future = info.make_future_dataframe(periods=period)
		projection = info.predict(future)

		
		st.subheader(f'Projected Data For {selected_stock}')
		st.write(f'Forecast plot for {n_years} years')
		graph = plot_plotly(info, projection)
		st.plotly_chart(graph)

		st.write("Forecast components")
		pic = info.plot_components(projection)
		st.write(pic)


if __name__ == "__main__":
        app()