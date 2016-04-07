# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:52:01 2016

@author: cdd
"""

import json
#from sets import Set
import matplotlib.pyplot as plt
import sys

#hashtags = ['gohawks']
hashtags = ['gohawks', 'gopatriots', 'nfl' ,'patriots', 'sb49', 'superbowl']
#authors = set([])
#numoftweet = 0
#followers = 0
#retweet = 0
#mintime = sys.maxint
#maxtime = 0
#tweets = []
for tag in hashtags:
    #temp = []
    authors = set([])
    numoftweet = 0
    followers = 0
    retweet = 0
    mintime = sys.maxint
    maxtime = 0
    flag = 1
    tweetshour = {}
    #retweet2 = 0
    with open("tweet_data/tweets_#%s.txt" % (tag)) as data_file:    
        for line in data_file:
            tweet = json.loads(line)
            if flag == 1:
                mintime = tweet['citation_date']
                flag = 0
            if tweet['author']['nick'] not in authors:
                authors.add(tweet['author']['nick'])
                followers += tweet['author']['followers']
            if(tweet['type'] == 'tweet'):
                numoftweet = numoftweet + 1
                #retweet2 += tweet['tweet']['retweet_count']
            else:
                retweet += 1
                numoftweet = numoftweet + 1
            currenthour = tweet['citation_date'] / 3600
            if currenthour in tweetshour:
                tweetshour[currenthour] += 1
            else:
                tweetshour[currenthour] = 1 
            
        #temp.append(tweet)
    #tweets.append(temp)
    maxtime = tweet['citation_date']
    print 'Total Retweets : %d' % (retweet)
    #print 'Total Retweets : %d' % (retweet2)
    print 'Total tweets : %d' % (numoftweet)
    print 'Total Retweets per hour: %d' % (retweet / ((maxtime -mintime) / 3600 ))
    print 'Total Authors : %d' % (len(authors))
    print 'Total Follower per Author : %d' % (followers / len(authors))
    print 'StartTime %d , end time %d' % (mintime , maxtime)
    print 'Total tweets per hour : %d' %  (numoftweet / ((maxtime -mintime) / 3600 ))            
    plt.bar(tweetshour.keys(), tweetshour.values(), 1, color='b')
    plt.show()