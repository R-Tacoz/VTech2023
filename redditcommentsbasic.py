#%% Imports & Setup    
import sys
from string import punctuation, ascii_letters, whitespace
from collections.abc import Iterable
import time
import datetime as dt
#from zoneinfo import 
import itertools
import pickle
import logging
logging.basicConfig(filename="D:\\Python\\VTech\\testlogs.txt", 
                    filemode="w",
                    format="[%(asctime)s] %(name)s (%(levelname)s): %(message)s", 
                    datefmt="%m-%d, %H:%M:%S", 
                    level=logging.DEBUG,
                    force=True)
import smart_open
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import enchant
import nltk
from nltk.corpus import wordnet, stopwords, words
# nltk.download('punkt')
# nltk.download('words')
# nltk.download('omw-1.4')
# nltk.download('stopwords')
# nltk.download('wordnet')     
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
#import praw
from psaw import PushshiftAPI                          

en_dict = enchant.Dict("en_US")
lemmatizer = nltk.stem.WordNetLemmatizer()
# reddit = praw.Reddit(
#     client_id="5rBT27W8r4BvdS9w6xpajw",
#     client_secret="jA_kp8yfdKkDT1rlBGPO9_BAHdWe6Q",
#     user_agent="NLP Scraper 1.0 by u/rockedtaco",
# )
api = PushshiftAPI(domain="api", rate_limit_per_minute=60)

#%% Data Collection

def to_unixtime(year, month=1, day=1, hour=0, minute=0, second=0) -> int:
    return int(dt.datetime(year, month, day, hour, minute, second).timestamp())

def to_clocktime(unix_time):
    return dt.datetime.utcfromtimestamp(unix_time).strftime("%Y-%m-%d %H:%M")

def extract_comments(results: list) -> list: 
    """ Returns a list of only comment content """
    assert "body" in results[0][-1].keys(), f"Comments did not contain a body"
    
    return [comment[-1]["body"] for comment in results]

def store_comments(results: list, file_name: str, append=False) -> None:
    content = extract_comments(results)
    if append:
        content.extend(get_comments(file_name))
    with open(file_name, 'wb') as f:
        pickle.dump(len(content), f)
        pickle.dump(content, f)

def get_ncomments(file_name: str) -> int:
    with open(file_name, "rb") as f:
        return pickle.load(f)

def get_comments(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        _ = pickle.load(f)
        return pickle.load(f)

class RedditCommentsDataset:
    def __init__(self, filenames: Iterable[str], tagged=False, preprocess=None): 
        self.filenames = filenames
        self._lengths = [get_ncomments(filename) for filename in filenames]
        self.data = None
        #self.iter_docs = iter_docs # return full comments vs tokens in comments
        self.tagged = tagged #doc2vec  only accepts TaggedDocument
        self.startidx = 0
        self.preprocess = preprocess
    
    @property
    def lengths(self) -> list:
        return self._lengths        
    
    def __len__(self):
        return sum(self.lengths)
    
    def __iter__(self):
        self.startidx=0
        for fileidx in range(len(self.filenames)):
            self.data = get_comments(self.filenames[fileidx])
            for i, comment in enumerate(self.data):
                text = self.preprocess(comment) if self.preprocess is not None else comment
                if self.tagged:
                    yield gensim.models.doc2vec.TaggedDocument(text, [self.startidx+i])#, 1 if fileidx==0 else 0
                else:
                    yield text#, 1 if fileidx==0 else 0
            self.startidx += len(self.data)


#%% Data Processing

match_words = re.compile(r"\b\w\w+\b")
def process_text(text):
    '''
    Takes in a string of text, then remove all PUNCTUATION, DIGITS, and 
    STOPWORDS and returns a TOKENIZED list of the remaining words LEMMATIZED
    '''
    cleaned = match_words.findall(text)
    # cleaned = [char for char in text if char not in punctuation]
    # cleaned = ''.join([i for i in cleaned if not i.isdigit()])
    cleaned =  [word.lower() for word in cleaned if word not in stopwords.words('english')]
    return [lemmatizer.lemmatize(word) for word in cleaned]


def pull_data(year: int, subreddit: str, file: str, count: int=5000) -> None:
    leap_day = 86400 if year%4==0 and (year%100!=0 or year%400==0) else 0
    #np.random.seed(521168) # from random.org
    
    results = [None]*count
    for i in range(count):
        print(f"Comment {i+1}/{count}", end='\r')
        
        utime = to_unixtime(year) + np.random.randint(31536000 + leap_day)
        comment = next(api.search_comments(
            subreddit = subreddit,
            after = utime,
            limit = 1,
            metadata = "true",
        ))
        comment = repr(comment[-1]["body"])
        #comment = "hurray!"
        results[i] = comment
    
    with open(file, 'wb') as f:
        pickle.dump(results, f)
        
#%% Main

def main():
    t1 = time.perf_counter()
    
    comment_files = ("adhd3000_comments", "notadhd_comments", "memes_comments", "news_comments", "showerthoughts_comments")
    dataset = RedditCommentsDataset( comment_files, tagged=True, preprocess=process_text)
    #dataset = RedditCommentsDataset( comment_files, tagged=False, preprocess=None)
    n_adhd_comments = dataset.lengths[0]
    n_neutral_comments = sum(dataset.lengths[1:])
    print(f"Current dataset's ADHD comments: {n_adhd_comments}/{n_adhd_comments + n_neutral_comments}")
    
    doc_model = gensim.models.Doc2Vec(documents=dataset,vector_size=50, min_count=2, epochs=50)
    vectors = [doc_model.dv[i] for i in range(n_adhd_comments+n_neutral_comments)] 
    # vectorizer = TfidfVectorizer(analyzer=process_text)
    # vectorizer.fit(dataset)
    # vectors = vectorizer.transform(dataset).toarray()
    labels = np.append(np.ones(n_adhd_comments), np.zeros(n_neutral_comments))
    
    t2 = time.perf_counter()
    print(f"Data prepared in {t2 - t1}")
    
    clf = LogisticRegression()
    clf.fit(vectors, labels)
    print("Training done in {(t1 := time.perf_counter()) - t2}")
    
    
    corrects = 0
    for i in range(len(dataset)):
        comment_vector = [vectors[i]]
        ground_truth = labels[i]
        pred = clf.predict(comment_vector)
        
        if abs(pred - ground_truth) < 0.5:
            corrects += 1
            
    print("Training accuracy:", corrects/len(dataset))
    
    testdataset = RedditCommentsDataset( ["testadhd_comments"], tagged=True, preprocess=process_text)
    #testdataset = RedditCommentsDataset( ["testadhd_comments"], tagged=False, preprocess=None)
    test_vectors = [doc_model.infer_vector(doc.words) for doc in dataset]
    #test_vectors = vectorizer.transform(testdataset).toarray()
    
    corrects = 0
    for i in range(len(testdataset)):
        comment_vector = [test_vectors[i]]
        ground_truth = 1
        pred = clf.predict(comment_vector)
        if abs(pred - ground_truth) < 0.5:
            corrects += 1
            
    print("Pure ADHD val acc:", corrects/len(testdataset))
    
    testdataset = RedditCommentsDataset( ["testmemes_comments"], tagged=True, preprocess=process_text)
    #testdataset = RedditCommentsDataset( ["testmemes_comments"], tagged=False, preprocess=None)
    test_vectors = [doc_model.infer_vector(doc.words) for doc in dataset]
    #test_vectors = vectorizer.transform(testdataset).toarray()
    
    corrects = 0
    for i in range(len(testdataset)):
        comment_vector = [test_vectors[i]]
        ground_truth = 0
        pred = clf.predict(comment_vector)
        if abs(pred - ground_truth) < 0.5:
            corrects += 1
            
    print("Pure memes val acc:", corrects/len(testdataset))
 
               
    # model = KMeans(n_clusters=3, n_init=10)
    
    # model = model.fit(vectors)
    
    # print(model.cluster_centers_)
    
   
    # start_time = to_unixtime(2023)
    # end_time = to_unixtime(2023)
    # num_results = 1000
    # results = list(api.search_comments(
    #     #q="",
    #     subreddit="adhd",
    #     after = start_time,
    #     #before = end_time,
    #     limit=num_results,
    #     #filter=["created_utc"], # can't apply filters for some reason
    #     metadata="true",
    #     # sort="desc", # doesn't apply to comments for some reason
    #     # sort_type="created_utc",
    # ))
    
    # store_comments(results, "adhd1000_comments", append=False)


if __name__== "__main__" : main()