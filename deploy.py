import streamlit as st
import tweepy
import re
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import twitterkey.key as access_key

key=access_key.validation()

with open("logi_model.pkl",'rb') as mod:
    model=pickle.load(mod)
with open("vector.pkl",'rb') as vec:
    vector=pickle.load(vec)
    
st.markdown("""<center><h1 style="background-color:#1DA1F2;color:#FFFFFF;padding-top:5%;
        padding-bottom:5%;border-radius: 15px;size=70%;">TWITTER SENTIMENTAL ANALYSIS</h1></center>"""
        ,unsafe_allow_html=True)
    
    
components.html(
        """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
    * {
      box-sizing: border-box;
    }
    
    body {
      margin: 0;
    }
    
    /* Clear floats after the columns */
.row:after {
  content: "";
  display: table;
  clear: both;
}

/* Responsive layout - makes the three columns stack on top of each other instead of next to each other */
@media screen and (max-width:600px) {
  .column {
    width: 100%;
  }
}
    
    /* Create three equal columns that floats next to each other */
    .column {
      float: left;
      width: 50%;
      padding: 15px;
    }
    .column h2{
        text-align: center;
    }
    .column p{
    text-align: justify;
    }
    
    
     .column2 {
    width: 100%;padding-top:38%;
  }
}
    
    /* Create three equal columns that floats next to each other */
    .column2 {
      float: center;
      width: 50%;
      padding: 15px;
    }
    .column2 h2{
        text-align: center;
    }
    .column2 p{
    text-align: justify;
    }
    
    /*  Body */
    *{
      margin:0;
      padding: 0;
    }
    s
    p{
      margin-left: 20%
    }
    
    
    </style>
       
        </head>
        <body>
            <br><br><br>  
    <div class="row">
    
        <div class="column">
        <h2>BUSINESS MONITORING</h2><br>
        <p>Online reputation is one of the most precious assets for brands. A bad review on social media can be costly to a company if it’s not handled effectively and swiftly. 
Twitter sentiment analysis allows you to keep track of what’s being said about your product or service on social media, and can help you detect negative mentions before they turn into a major crisis.
</p>
      </div>
      
      <div class="column">
        <h2>CUSTOMER SERVICE</h2><br>
        <p>Twitter has become an essential channel for customer service. In fact, 60% of the customers are complaining on social media to expect a response quickly.
Twitter sentiment analysis allows you to track and analyze all the interactions between brand and customers. This can be very useful to analyze customer satisfaction based on the type of feedback.
<br><br></p>
      </div>
      
      <div class="column2">
        <h2>POLITICAL CAMPAIGN</h2><br>
        <p>
       A huge part of Twitter conversation revolves around news and 
       politics. That makes it an excellent place to measure public opinion, especially during election campaigns. Twitter Sentiment Analysis can provide interesting insights on how people feel about a specific candidate.
<br><br></p>
      </div></div><center>
        </body>
    </html>    
        """,
        height=450,)

@st.cache
def sentiment(model,vector,u_name,count):
    consumer_key = key[0] 
    consumer_secret = key[1] 
    access_token_key = key[2] 
    access_token_secret = key[3] 
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token_key, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    # input for term to be searched and how many tweets to search
    # searching for tweets
    tweets = tweepy.Cursor(api.search, q=u_name, lang = "en").items(count)
        
    df=pd.DataFrame([tweet.text for tweet in tweets],columns=["clean_tweet"])

    for index,text in df["clean_tweet"].iteritems():
        df["clean_tweet"][index]=re.sub(r"https?:\/\/\S+"," ",text)

    for index,text in df["clean_tweet"].iteritems():
        df["clean_tweet"][index]=re.sub(r"RT[\s]+"," ",text)
        
    for index,text in df["clean_tweet"].iteritems(): #Using iteritem method in pandas generating the index and sentence
        df["clean_tweet"][index]=re.sub('[^A-Za-z0-9 ]+', ' ', text)
 #Remove Re-tweet


    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w)>2]))
    tokenized_tweet = df['clean_tweet'].apply(lambda x: x.split())
    tokenized_tweet.head()


    stemmer = PorterStemmer()

    tokenized_tweet = tokenized_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])
    tokenized_tweet.head()


    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = " ".join(tokenized_tweet[i])


    df['clean_tweet'] = tokenized_tweet

    x = vector.transform(df['clean_tweet'])
    
    predicted=model.predict(x)
    y=np.where(predicted==0,"negative","positive")
    return df["clean_tweet"],y

u_name=st.sidebar.text_input("Enter the user name")
tweet_count=st.sidebar.text_input("Enter the number of tweets","")
submit_button=st.sidebar.button("SUBMIT")

if submit_button:
    if u_name is not None and tweet_count!=0:
        try:
            x,y=sentiment(model,vector,u_name,int(tweet_count))
            st.markdown("""
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>.postive {
                    padding: 1px;size:30%;color: black;}
                    </style>
                    <h2></h2>
                    <div class="postive">
                    <span>
                    <strong><h2><b>RESULTS </strong> 
                    </span> </div>""",unsafe_allow_html=True)
            output_data=pd.DataFrame(data=[x,y]).T
            output_data.columns=["Tweet","Sentiment"]
            fig,ax = plt.subplots(1,2,figsize=[12,5])
            temp = dict(output_data['Sentiment'].value_counts())
            colors = ['lightgreen',  'salmon']
            sns.countplot(x=output_data["Sentiment"],palette="rocket_r",ax=ax[0])
            ax[1].pie(temp.values(),labels=temp.keys(),colors=colors, explode=[0.05]*2,autopct="%.1f%%")
            st.pyplot(fig)
            
            positive_perct=output_data["Sentiment"].value_counts(normalize=True)[0]
            st.text("Positive Tweet percentage : {}".format(positive_perct*100))
            
            negative_perct=output_data["Sentiment"].value_counts(normalize=True)[1]
            st.text("Negative Tweet percentage : {}".format(negative_perct*100))

            expander = st.beta_expander("Tweets")
            col1,col2=expander.beta_columns(2)
            col1.markdown("""
                            <meta name="viewport" content="width=device-width, initial-scale=1">
                            <style>.postive {
                                padding: 1px;size:30%;background-color: white;color: green;}
                                </style>
                                <h2></h2>
                                <div class="postive">
                                <span>
                                <strong><h2><b><center>Positive Tweets </strong> 
                                </span> </div>""",unsafe_allow_html=True)
            
            positve_tweet=output_data[output_data["Sentiment"]=="positive"]["Tweet"]
            
            col1.write(positve_tweet)
            
            col2.markdown("""
                            <meta name="viewport" content="width=device-width, initial-scale=1">
                            <style>.negative {
                                padding: 1px;size:30%;;background-color: white;color: red;}
                                </style>
                                <h2></h2>
                                <div class="negative">
                                <span>
                                <strong><h2><b><center>Negative Tweets </strong> 
                                </span> </div>""",unsafe_allow_html=True)
            
            negative_tweet=output_data[output_data["Sentiment"]=="negative"]["Tweet"]
            col2.write(negative_tweet)

            
            
        except:
            st.markdown("""
                            <meta name="viewport" content="width=device-width, initial-scale=1">
                            <style>.alert {
                                padding: 10px;background-color: #f44336;color: white;}
                                </style>
                                <h2></h2>
                                <div class="alert">
                                <span>
                                <strong>Warning!</strong> Please check with your inputs and Try Again...      
                                </span> </div>""",unsafe_allow_html=True)
        
        






