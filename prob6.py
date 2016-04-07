# -*- coding: utf-8 -*-
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve
import urllib
from mpl_toolkits.basemap import Basemap


def top_n_feats(row,features, top_n=10):
   ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
   top_ids = np.argsort(row)[::-1][:top_n]
   top_feats = [(features[i]) for i in top_ids]
   print top_feats
   print '\n'
   return

googleUrl = 'http://maps.googleapis.com/maps/api/geocode/json?'

def get_coordinates(query, from_sensor=False):
   query = query.encode('utf-8')
   params = {
       'address': query,
       'sensor': "true" if from_sensor else "false"
   }
   url = googleUrl + urllib.urlencode(params)
   print url
   json_response = urllib.urlopen(url)
   response = json.loads(json_response.read())
   if response['results']:
       location = response['results'][0]['geometry']['location']
       latitude, longitude = location['lat'], location['lng']
       print query, latitude, longitude
   else:
       latitude, longitude = None, None
   return latitude, longitude
   
if __name__ == '__main__':  
    
   TweetTrainData = []
   data = []
   data_test = []
   target=[]
   target_test = []
   data_row = []
   tweet_time = []
   TweetOutput = []
   tweet_Hashes = []
   hashtags_test = [ 'nfl' ]
   hashtags = ['gopatriots','gohawks',]
   hawksfan = 0
   patsfan = 0
   locations = []
   blocked_words = ['nfl' ,'sb49', 'superbowl', 'seahawks', 'patriots', 'gopats','superbowi']
       
   for tag in hashtags_test:
       with open("tweet_data/tweets_#%s.txt" % (tag)) as data_file:
           for line in data_file: 
               
               tweet = json.loads(line)
               hawksfan = 0
               patsfan = 0
               tweet_Hashes=[]
               for tweet_hashtag in tweet['tweet']['entities']['hashtags']:
                   if tweet_hashtag['text'].lower() not in hashtags:
                       if all(x not in tweet_hashtag['text'].lower() for x in blocked_words):
                           tweet_Hashes.append(tweet_hashtag['text'].lower())
                   else:
                       if tweet_hashtag['text'].lower() == 'gopatriots':
                           patsfan = 1
                       elif tweet_hashtag['text'].lower() == 'gohawks':
                           hawksfan = 1
               if tweet_Hashes and  patsfan + hawksfan == 1 : 
                   temp=" ".join(tweet_Hashes)
                   data_test.append(temp)
                   if  len(tweet['tweet']['user']['location']) > 0 and tweet['tweet']['user']['location'] is not None:

                       locations.append(tweet['tweet']['user']['location'])
                   else:
                       locations.append('')
                   if patsfan:
                       target_test.append(0)
                   elif hawksfan:
                       target_test.append(1)
   
   for tag in hashtags:
       with open("tweet_data/tweets_#%s.txt" % (tag)) as data_file:
           for line in data_file: 
               
               tweet = json.loads(line)
               tweet_Hashes=[]
               for tweet_hashtag in tweet['tweet']['entities']['hashtags']:
                   if tweet_hashtag['text'].lower() != tag.lower():
                       if all(x not in tweet_hashtag['text'].lower() for x in blocked_words):
                           tweet_Hashes.append(tweet_hashtag['text'].lower())
               if tweet_Hashes:
                   temp=" ".join(tweet_Hashes)
                   data.append(temp)
                   if tag == 'gopatriots':
                       target.append(0)
                   elif tag == 'gohawks':
                       target.append(1)

count_vect = CountVectorizer()
count_vect.fit(data)
X_train_data = count_vect.transform(data)
X_test_counts  = count_vect.transform(data_test)

features = count_vect.get_feature_names()

tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(X_train_data)
X_train_data = tfidf_transformer.transform(X_train_data)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
print X_train_data.shape
Y = np.zeros(shape=(2,X_train_data.shape[1]))

for i in range(X_train_data.shape[0]):
   tclass = target[i]
   Y[tclass] += X_train_data[i]

row1 = Y[0]
row2 = Y[1]
top_n_feats(row1,features,10)
top_n_feats(row2,features,10)

svd=TruncatedSVD(n_components=50, n_iter=10,random_state=42)
X_train_data=svd.fit_transform(X_train_data)
vectors_svd_test=svd.transform(X_test_tfidf)

clf = GaussianNB()
X=X_train_data
Y=target
print X_train_data.shape

X_test=vectors_svd_test
Y_test=target_test
clf.fit(X,Y)
Y_score=clf.predict(X_test)
print X_train_data.shape

pre = clf.predict(X_test)

fpr,tpr,threshold=roc_curve(Y_test,Y_score)
plt.plot(fpr,tpr)

pat_lons = []
pat_lats = []
sea_lons = []
sea_lats = []

lons = []
lats = []

for i in range(0,len(locations)):
  if locations[i] and len(locations[i]) > 0:
      la,lo = get_coordinates(locations[i])
      if lo is not None and la is not None:
          if pre[i] == 0:
              pat_lons.append(lo)
              pat_lats.append(la)
          else:
              sea_lons.append(lo)
              sea_lats.append(la)
  
map =  Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
           llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')

map.drawcountries()
map.fillcontinents(color = 'red')
map.drawmapboundary()

x_1,y_1 = map(pat_lons, pat_lats)
x_2,y_2 = map(sea_lons, sea_lats)
map.plot(x_1, y_1, 'go', markersize=3)
map.plot(x_2, y_2, 'yo', markersize=3)

