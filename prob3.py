"""
Created on Tue Mar  8 21:33:14 2016

@author: jjzhu
"""

import numpy as np
import statsmodels.api as sm
import datetime, time
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import json
import datetime, time
import collections
from sklearn.cross_validation import cross_val_predict

from sets import Set
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import sys

hashtags = ['gohawks' , 'gopatriots' , 'nfl', 'patriots' , 'sb49', 'superbowl']
#hashtags = ['sb49']
results = []

start_date = datetime.datetime(2015,01,01, 12,0,0)
end_date = datetime.datetime(2015,02,07, 0,0,0)

for tag in hashtags:
    print tag
    window=1
    mintime = int(time.mktime(start_date.timetuple()))
    endtime = mintime+3600*window
    maxtime = int(time.mktime(end_date.timetuple()))
    
    
#useful features
    sumvalue = []
    normal = []
    num_tweets=0
    sum_tweets=0
    
    num_retweets=0
    sum_retweets=0
    
    num_followers=0
    sum_followers=0
    
    max_followers=0
    sum_maxfollowers=0
    
    impression_count=0
    sum_impressioncount=0
    
    authors = []
    sum_authors=0
    
    url_ratio=0
    sum_urlratio=0
    
    ranking_score=0
    sum_rankingscore=0
    
    favorite_count=0
    sum_favoritecount=0
        
    
 
    X=[]
    Y=[]
    Xi=[]
    with open("tweet_data/tweets_#%s.txt" % (tag)) as data_file:
        flag=1      #flag=1 means the first line
        for line in data_file:
            tweet = json.loads(line)
            if flag==1:             #to make sure the first line stands in the first Xi
                while (tweet['firstpost_date']>endtime):
                    endtime=endtime+window*3600
                flag=0
                num_tweets =1
                num_retweets=tweet['metrics']['citations']['total']
                num_followers=tweet['tweet']['user']['followers_count']
                max_followers=max(max_followers, tweet['tweet']['user']['followers_count'])
                impression_count=tweet['metrics']['impressions']
                if tweet['original_author']['url'] not in authors:                    
                    authors.append(tweet['original_author']['url'])
                if tweet['tweet']['entities']['urls']:
                    url_ratio=1;
                ranking_score=tweet['metrics']['ranking_score']
                favorite_count=tweet['tweet']['favorite_count']
            else:
                if tweet['firstpost_date']>=mintime:
                    if tweet['firstpost_date']>=maxtime:
                        break;
                    if tweet['firstpost_date']<endtime:
                        num_tweets +=1
                        num_retweets+=tweet['metrics']['citations']['total']
                        num_followers+=tweet['tweet']['user']['followers_count']
                        max_followers=max(max_followers, tweet['tweet']['user']['followers_count'])
                        impression_count+=tweet['metrics']['impressions']
                        if tweet['original_author']['url'] not in authors:                    
                            authors.append(tweet['original_author']['url'])
                        if tweet['tweet']['entities']['urls']:
                            url_ratio+=1;
                        ranking_score+=tweet['metrics']['ranking_score']
                        favorite_count+=tweet['tweet']['favorite_count']
                    else:
                        while tweet['firstpost_date']>=endtime:
                            Xi.append(1)  #constant
                            
                            Xi.append(num_tweets)
                            Y.append(num_tweets)
                            sum_tweets+=num_tweets
                            num_tweets=0
                            
                            
                            Xi.append(num_retweets)
                            sum_retweets+=num_retweets
                            num_retweets=0
                            
                            
                            Xi.append(num_followers)
                            sum_followers+=num_followers
                            num_followers=0
                            
                            
                            Xi.append(max_followers)
                            sum_maxfollowers+=max_followers
                            max_followers=0
                            
                            
                            Xi.append(impression_count)
                            sum_impressioncount+=impression_count
                            impression_count=0
                            
                            
                            Xi.append(len(authors))
                            sum_authors+=len(authors)
                            authors = []
                             
                            
                            if num_tweets==0:
                                Xi.append(0)
                            else:
                                url_ratio=float(url_ratio)/num_tweets
                                Xi.append(url_ratio)
                            sum_urlratio+=url_ratio
                            
                            
                            Xi.append(ranking_score)
                            sum_rankingscore+=ranking_score
                            rankingscore=0
                            
                            
                            Xi.append(favorite_count)
                            sum_favoritecount+=favorite_count
                            favorite_count=0
                            
                            
                            #Xi.append(int(datetime.datetime.fromtimestamp(endtime).strftime("%H")))
                            hour = int(datetime.datetime.fromtimestamp(endtime).strftime("%H"))-2
                            onehottemp = [0] * 24
                            onehottemp[hour] = 1
                            Xi = Xi + onehottemp
                            endtime=endtime+window*3600
                            
                            X.append(Xi)
                            Xi=[]
                        num_tweets=1
                        num_retweets=tweet['metrics']['citations']['total']
                        num_followers=tweet['tweet']['user']['followers_count']
                        max_followers=tweet['tweet']['user']['followers_count']
                        impression_count=tweet['metrics']['impressions']
                        if tweet['original_author']['url'] not in authors:                    
                                authors.append(tweet['original_author']['url'])
                        if tweet['tweet']['entities']['urls']:
                                url_ratio=1
                        else:
                                url_ratio=0
                        ranking_score=tweet['metrics']['ranking_score']
                        favorite_count=tweet['tweet']['favorite_count']
    Y=collections.deque(Y)
    Y.rotate(-1)
    Y=list(Y)
    
    rf = RandomForestRegressor(n_estimators=40, max_depth=100, warm_start=True, oob_score=True)
    rf.fit(X, Y)
    pre = cross_val_predict(rf, X, Y, cv=10)


    print "mean_absolute_error", mean_absolute_error(pre, Y)
    print "Coeficient", rf.feature_importances_
    
    
    
    data_file.close()

    
 
