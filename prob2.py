# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 20:30:35 2016

@author: jjzhu
"""


import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
import json
import datetime, time
import collections

from sets import Set
import matplotlib.pyplot as plt
import sys

hashtags = ['gohawks' , 'gopatriots' , 'nfl', 'patriots' , 'sb49', 'superbowl']
#hashtags = ['superbowl']
for tag in hashtags:
    print tag
    flag=1
    start_time=0
    end_time=0
    num_tweet=0
    num_retweets=0
    num_followers=0
    max_followers=0
    window=1
    X=[]
    Y=[]
    Xi=[]
    with open("tweet_data/tweets_#%s.txt" % (tag)) as data_file:
        for line in data_file:
            tweet = json.loads(line)
            if flag==1:
                while (tweet['firstpost_date']>end_time):
                        end_time=end_time+window*3600
                flag=0
                start_time=tweet['firstpost_date']
                num_tweets=1
                num_retweets=tweet['metrics']['citations']['total']
                num_followers=tweet['tweet']['user']['followers_count']
                max_followers=tweet['tweet']['user']['followers_count']
                
                window=1;
                end_time=start_time+window*3600
                
            
            else:
                if tweet['firstpost_date']<end_time:
                    num_tweets +=1
                    num_retweets+=tweet['metrics']['citations']['total']
                    num_followers+=tweet['tweet']['user']['followers_count']
                    max_followers=max(max_followers, tweet['tweet']['user']['followers_count'])
                else:
                    end_time=end_time+window*3600
                    Xi.append(1)
                    Xi.append(num_tweets)
                    Y.append(num_tweets)
                    Xi.append(num_retweets)
                    Xi.append(num_followers)
                    Xi.append(max_followers)
                    hour = int(datetime.datetime.fromtimestamp(end_time).strftime("%H"))-2
                    onehottemp = [0] * 24
                    onehottemp[hour] = 1
                    Xi = Xi + onehottemp
                    X.append(Xi)
                    Xi=[]
                    while tweet['firstpost_date']>end_time:
                        Xi.append(1)
                        
                        Xi.append(0)
                        Y.append(0)
                    
                        
                        Xi.append(0)
                       
                        
                        
                        Xi.append(0)
                   
                        
                        
                        Xi.append(0)
          
                        
                        
                        #Xi.append(int(datetime.datetime.fromtimestamp(end_time).strftime("%H"))-2 )
                        hour = int(datetime.datetime.fromtimestamp(end_time).strftime("%H"))-2
                        onehottemp = [0] * 24
                        onehottemp[hour] = 1
                        Xi = Xi + onehottemp
                        end_time=end_time+window*3600
                        
                        X.append(Xi)
                        Xi=[]
                    num_tweets=1
                    num_retweets=tweet['metrics']['citations']['total']
                    num_followers=tweet['tweet']['user']['followers_count']
                    max_followers=tweet['tweet']['user']['followers_count']
    Y=collections.deque(Y)
    Y.rotate(-1)
    Y=list(Y)
    pre = sm.OLS(Y, X).fit()
    results.append(pre)
    
    print('Parameters: ', pre.params)
    print('Standard errors: ', pre.bse)
    #print('Predicted values: ', pre.predict())
    

    print('t_test: ', pre.t_test)
    print('p values: ', pre.pvalues)
    print('t values', pre.tvalues)
    print('mean absolute error', mean_absolute_error(pre.predict(), Y))
    data_file.close()
                
                
                