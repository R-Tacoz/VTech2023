#%% Imports & Setup
import logging, sys, random

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import nltk
from modAL.models import ActiveLearner, Committee
from modAL.uncertainty import entropy_sampling
from modAL.disagreement import KL_max_disagreement
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model, sklearn.neighbors, sklearn.ensemble, sklearn.svm
# from rake_nltk import Rake
# from pke.unsupervised import YAKE
from wordcloud import WordCloud

import torch
from torch import nn
from transformers import logging as transformerslogging
from transformers import BertConfig, BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer

from utils import to_unixtime, to_clocktime, full_preprocess, light_preprocess, \
                  IDF, TF_IDF, RBipolarDatasetBuilder, w2vAverager, BertVectorizer, \
                  TFIDFVectorizerWrapper, ListDataset

HOME = "D:/Python/VTech/"
logging.basicConfig(
    filename = HOME+"project.log", 
    filemode = "w",
    format = "[%(asctime)s] %(name)s (%(levelname)s): %(message)s", 
    datefmt = "%m/%d-%H:%M:%S", 
    level = logging.DEBUG,
    force = True,
)
transformerslogging.set_verbosity_error()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Models

logreg = sklearn.linear_model.LogisticRegression(C=0.5, max_iter=1000)
svm = sklearn.svm.SVC(C=0.8, kernel="rbf")
knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
rdmforest = sklearn.ensemble.RandomForestClassifier(n_estimators=20)
mlp = nn.Sequential(
    nn.Linear(768, 512), nn.ReLU(),
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 3), nn.Sigmoid(),
)

#%% Main

def make_dataset():
    portion = 46
    order = ['created_utc','id','selftext','title','subreddit','anxiety','bipolar','depression']
    
    df = pd.read_csv("D:/Python/VTech/bysubredditdata/rbipolar_2020&2021.csv",header=0, index_col=0)
    df = df.iloc[np.r_[0:1915, 5001:5931], [1,2,3,4,5,7,8,9]]
    #df = df.sample(frac=1)
    df0 = pd.DataFrame(columns=order)
    a, b, d, z = 0,0,0,0
    for idx, row in df.iterrows():
        anx, dep, bip = row[['anxiety', 'depression', 'bipolar']]
        if anx == 1 or dep == 1:
            df0.loc[len(df0.index)] = row
        elif anx == 0 and dep == 0 and bip == 0:
            z += 1
            #if z < 46:
            df0.loc[len(df0.index)] = row
    #print(a, b, d, z)
    a, b, d = 0,0,0
    for idx, row in df0.iterrows():
        if row['anxiety'] == 1: a += 1
        if row['bipolar'] == 1: b += 1
        if row['depression'] == 1: 
            d += 1
    print(a, b, d, z)
    
    df1 = pd.read_csv("D:/Python/VTech/bysubredditdata/raskreddit_1M2021.csv")
    df1['subreddit'] = "askreddit"
    df1['anxiety'] = 0; df1['bipolar'] = 0; df1['depression'] = 0;
    df1 = df1[order]
    df1 = df1.sample(portion)
    
    df2 = pd.read_csv("D:/Python/VTech/bysubredditdata/rjokes_1M2015-2020.csv")
    df2['subreddit'] = "jokes"
    df2['anxiety'] = 0; df2['bipolar'] = 0; df2['depression'] = 0;
    df2 = df2[order]
    df2 = df2.sample(portion)
    
    df3 = pd.read_csv("D:/Python/VTech/bysubredditdata/rcozyplaces&rcasualconversation&etc_2021&2022.csv")# 4 subreddits
    df3.rename(columns={"date":"created_utc","text":"selftext"}, inplace=True)
    df3['title'] = "removed" # will get removed in processing
    df3['subreddit'] = 'cozyplaces,CasualConversation,etc.'
    df3['anxiety'] = 0; df3['bipolar'] = 0; df3['depression'] = 0;
    df3 = df3[order]
    df3 = df3.sample(4*portion)
    
    collective = pd.concat([df0, df1, df2, df3])
    collective.reset_index(inplace=True)
    collective.drop(columns='index', inplace=True)
    
    a, b, d, z = 0,0,0,0
    for idx, row in collective.iterrows():
        anx, dep, bip = row[['anxiety', 'depression', 'bipolar']]
        if anx == 1: a += 1
        if bip == 1: b += 1
        if dep == 1: d += 1
        if anx==0 and bip == 0 and dep == 0: z += 1
    print("count of anxiety, bipolar, depression, none:",a,b,d,z)
    
    collective.to_csv("D:/Python/VTech/bysubredditdata/bipolar_dataset.csv")

def save_vectors(dataset: RBipolarDatasetBuilder, path):
    stack = [(vec, label) for vec, label in dataset ]
    torch.save(stack, path)

def make_embeddings(path):    
    dataset = RBipolarDatasetBuilder(
        file_path = path, 
        preprocess = full_preprocess, 
        vectorizer = None, 
        n_grams = 0
    )
    
    # tfidf = TfidfVectorizer()
    # tfidf.fit([' '.join(sample[0]) for sample in dataset])
    # dataset.n_grams = 0
    # dataset.vectorizer = TFIDFVectorizerWrapper(tfidf)
    # print("Saving TF-IDFs")
    # save_vectors(dataset, HOME+"data/tfidf_embeddings.pt")
    
    # dataset.n_grams = 0; dataset.vectorizer = None
    # # d2v = Doc2Vec(
    # #     documents = [ TaggedDocument(sample[0], [i]) for i,sample in enumerate(dataset) ],
    # #     vector_size = 200,
    # #     epochs = 69,
    # # )
    # # d2v.save("D:/Python/VTech/models/Doc2Vec_v2")
    # d2v = Doc2Vec.load("D:/Python/VTech/models/Doc2Vec_v2")
    # dataset.vectorizer = d2v.infer_vector
    # print("Saving doc2vecs")
    # save_vectors(dataset, HOME+"data/doc2vec_embeddings.pt")
    
    dataset.ngrams = 0; dataset.vectorizer = None
    # w2v = Word2Vec(
    #     sentences = [sample[0] for sample in dataset],
    #     vector_size = 200,
    #     epochs = 69,
    # )
    # w2v.save("D:/Python/VTech/models/Word2Vec_v2")
    w2v = Word2Vec.load("D:/Python/VTech/models/Word2Vec_v2")
    dataset.vectorizer = w2vAverager(w2v.wv)
    print("Saving word2vec avgs")
    save_vectors(dataset, HOME+"data/word2vecavg_embeddings.pt")
    
    # dataset.preprocess = light_preprocess
    # bert = BertModel.from_pretrained('bert-base-uncased').to(device)
    # #bert = BertModel.from_pretrained("D:/Python/VTech/models/BERTModel")
    # bert.eval()
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # #tokenizer = BertTokenizer.from_pretrained("D:/Python/VTech/models/BERTTokenizer")
    # dataset.vectorizer = BertVectorizer(bert, tokenizer, device)
    # with torch.no_grad():
    #     print("Saving BERT avgs")
    #     cpu = torch.device('cpu')
    #     stack = [(vec.to(cpu), label.to(cpu)) for vec, label in tqdm(dataset) ]
    #     torch.save(stack, HOME+"data/BERTavg_embeddings.pt")
    
    # sbert = SentenceTransformer('all-MiniLM-L6-v2')
    # # sbert = SentenceTransformer.load("D:/Python/VTech/models/SBERTModel")
    # dataset.preprocess = None
    # dataset.vectorizer = None
    # texts = []
    # labels = []
    # print("Saving SBERTs")
    # for text, label in dataset:
    #     texts.append(text)
    #     labels.append(label)
    # vecs = sbert.encode(texts)
    # torch.save([(vec.squeeze(), label) for vec, label in zip(vecs, labels)], HOME+"data/SBERT_embeddings.pt")
    return

def main():
    data = torch.load(HOME+"data/SBERT_embeddings.pt")
    raw_text = [ "ROW "+str(i)+": "+str(row['title'])+"\n"+str(row['selftext'])
                for i,row in pd.read_csv(HOME+"data/rbipolar_dataset-balanced.csv").loc[:,['selftext','title']].iterrows() ]
    _temp = list(zip(data, raw_text))
    random.shuffle(_temp)
    data, raw_text = zip(*_temp)
    train_data = ListDataset(data[:1000])
    val_data = ListDataset(data[1000:])
    
    # train_x = [sample[0].to_dense().squeeze() for sample in train_data] # when loading tfidf embeddings
    train_x = [sample[0] for sample in train_data]
    train_y = [sample[1][1] for sample in train_data]
    # val_x = [sample[0].to_dense().squeeze() for sample in val_data]
    val_x = np.array([sample[0] for sample in val_data])
    val_y = np.array([sample[1][1] for sample in val_data])
    
    logreg = sklearn.linear_model.LogisticRegression(C=0.5, max_iter=1000)
    svm = sklearn.svm.SVC(C=0.8, kernel="rbf", probability=True)
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
    rdmforest = sklearn.ensemble.RandomForestClassifier(n_estimators=20)
    
    logreg.fit(train_x, train_y)
    knn.fit(train_x, train_y)
    svm.fit(train_x, train_y)
    rdmforest.fit(train_x, train_y)
    
    def KL_max_disagreement_sampling(committee, X, **kwargs):
        max_disagreements = KL_max_disagreement(committee, X, **kwargs)
        return sorted(range(len(X)), key=( lambda i: max_disagreements[i] ), reverse=True)[:10]
    
    AL_committee = Committee(
        learner_list = [
            ActiveLearner( logreg, entropy_sampling ),
            ActiveLearner( knn, entropy_sampling ),
            ActiveLearner( svm, entropy_sampling ),
            ActiveLearner( rdmforest, entropy_sampling ),
        ],
        query_strategy = KL_max_disagreement_sampling,
    )
    
    
    
    AL_committee.teach(val_x, val_y)
    idxs,_ = AL_committee.query(val_x)
    for i, X in zip(idxs, _):
        X = [X]
        print(raw_text[i])
        print(logreg.predict_proba(X), knn.predict_proba(X), svm.predict_proba(X), rdmforest.predict_proba(X))
    
     
# make_embeddings(HOME+"data/rbipolar_dataset-balanced.csv")
if __name__=="__main__": main()