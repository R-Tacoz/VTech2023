#%% Imports & Setup
import logging, random, itertools, time, copy
from typing import Iterable

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import nltk
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
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
from torchsummary import summary

from utils import to_unixtime, to_clocktime, full_preprocess, light_preprocess, \
                  IDF, TF_IDF, RBipolarDatasetBuilder, ListDataset, w2vAverager, BertVectorizer, \
                  TFIDFVectorizerWrapper, train

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

#%% Test Models
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

#%% Main Model
        
class MultiTaskModule(torch.nn.Module):
    ''' works for hard-sharing and soft-sharing parameters '''
    def __init__(self, task_module: nn.Module, n_tasks: int=1):
        super().__init__()
        self.n_tasks = n_tasks
        self.task_modules = nn.ModuleList([copy.deepcopy(task_module) for task in range(self.n_tasks)])
        with torch.no_grad(): # TODO try removing
            for param in self.task_modules.parameters():
                if param.requires_grad:
                    nn.init.uniform_(param, -1.0, 1.0)
        self.combinations = [pair for pair in itertools.combinations(range(self.n_tasks), 2)]
        
    def forward(self, x) -> torch.Tensor:
        return torch.stack([task_module(x) for task_module in self.task_modules])
        
    @staticmethod
    def L1_distance(A: nn.Module, B: nn.Module) -> torch.Tensor:
        ''' Absolute distance between params/Lasso regularization; sum( A-B ) '''
        d = torch.zeros(1)
        for p1, p2 in zip(A.parameters(), B.parameters()):
            if p1.requires_grad and p2.requires_grad:
                d += p1.sub(p2).abs().sum()
        return d
    
    @staticmethod
    def L2_distance(A: nn.Module, B: nn.Module) -> torch.Tensor:
        ''' Euclidean distance between params/Ridge regularization/Frobenius norm; sqrt( sum( (A-B)^2 ) ) '''
        d = torch.zeros(1)
        for p1, p2 in zip(A.parameters(), B.parameters()):
           if p1.requires_grad and p2.requires_grad:
               d += p1.dist(p2) #p1.sub(p2).square().sum() # p1.dist(p2)
        return d#.sqrt() # d
    
    def multitask_regularization_loss(self, p: int=2, task_idx: int=None) -> torch.Tensor: #TODO depth-varying & single task
        loss = torch.zeros(1)
        distance = self.L2_distance if p==2 else self.L1_distance
        if task_idx is None:
            for i1, i2 in self.combinations:
                loss += distance(self.task_modules[i1], self.task_modules[i2])
        else:
            for i in range(self.n_tasks):
                if i != task_idx:
                    loss += distance(self.task_modules[task_idx], self.task_modules[i])
        return loss

class SharedFeatureModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5)
        self.dropout = nn.Dropout(0.5)
        self.pooling = nn.MaxPool1d(kernel_size=3, stride=3)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.dropout(x)
        out = self.pooling(x)
        return out

class TaskSpecificModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Sequential(
            nn.LayerNorm(300),
            nn.Linear(300, 512), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(512, 512), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(128, 2),
        )
        
    def forward(self, x):
        out = self.dense(x)
        return out
    
class GMMTL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = SharedFeatureModule()
        task_specific = TaskSpecificModule()
        self.mtm = MultiTaskModule(task_specific, 3)
     
    def forward(self, x, topic):
        
        return x

#%% Data

def make_dataset():
    portion = 80
    order = ['created_utc','id','selftext','title','subreddit','anxiety','bipolar','depression']
    
    df = pd.read_csv("D:/Python/VTech/bysubredditdata/rbipolar_2020&2021.csv",header=0, index_col=0)
    df = df.iloc[np.r_[0:2001, 5001:5931], [1,2,3,4,5,7,8,9]]
    #df = df.sample(frac=1)
    df0 = pd.DataFrame(columns=order)
    a, b, d, z = 0,0,0,0
    for idx, row in df.iterrows():
        anx, dep, bip = row[['anxiety', 'depression', 'bipolar']]
        if anx == 1 or dep == 1:
            df0.loc[len(df0.index)] = row
        elif anx == 0 and dep == 0 and bip == 0:
            z += 1
            if z <= portion:
                df0.loc[len(df0.index)] = row
    #print(a, b, d, z)
    # a, b, d, z = 0,0,0,0
    # for idx, row in df0.iterrows():
    #     if row['anxiety'] == 1: a += 1
    #     if row['bipolar'] == 1: b += 1
    #     if row['depression'] == 1: d += 1
    #     if row['anxiety']==0 and row['bipolar']==0 and row['depression']==0: z += 1
    # print(a, b, d, z)
    
    # Control
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
    
    collective.to_csv("D:/Python/VTech/data/rbipolar_dataset-balanced.csv")
    return

def save_vectors(dataset: RBipolarDatasetBuilder, path):
    stack = [(vec, label) for vec, label in tqdm(dataset) ]
    torch.save(stack, path)

def make_embeddings(path):    
    dataset = RBipolarDatasetBuilder(
        file_path = path, 
        preprocess = full_preprocess, 
        vectorizer = None, 
        n_grams = 0
    )
    
    tfidf = TfidfVectorizer()
    tfidf.fit([' '.join(sample[0]) for sample in dataset])
    dataset.n_grams = 0
    dataset.vectorizer = TFIDFVectorizerWrapper(tfidf)
    print("Saving TF-IDFs")
    save_vectors(dataset, HOME+"data/tfidf_embeddings.pt")
    
    dataset.n_grams = 0; dataset.vectorizer = None
    # d2v = Doc2Vec(
    #     documents = [ TaggedDocument(sample[0], [i]) for i,sample in enumerate(dataset) ],
    #     vector_size = 200,
    #     epochs = 69,
    # )
    # d2v.save("D:/Python/VTech/models/Doc2Vec_v2")
    d2v = Doc2Vec.load("D:/Python/VTech/models/Doc2Vec_v2")
    dataset.vectorizer = d2v.infer_vector
    print("Saving doc2vecs")
    save_vectors(dataset, HOME+"data/doc2vec_embeddings.pt")
    
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
    
    dataset.preprocess = light_preprocess
    bert = BertModel.from_pretrained('bert-base-uncased').to(device)
    #bert = BertModel.from_pretrained("D:/Python/VTech/models/BERTModel")
    bert.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #tokenizer = BertTokenizer.from_pretrained("D:/Python/VTech/models/BERTTokenizer")
    dataset.vectorizer = BertVectorizer(bert, tokenizer, device)
    with torch.no_grad():
        print("Saving BERT avgs")
        cpu = torch.device('cpu')
        stack = [(vec.to(cpu), label.to(cpu)) for vec, label in tqdm(dataset) ]
        torch.save(stack, HOME+"data/BERTavg_embeddings.pt")
    
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    # sbert = SentenceTransformer.load("D:/Python/VTech/models/SBERTModel")
    dataset.preprocess = None
    dataset.vectorizer = None
    texts = []
    labels = []
    print("Saving SBERTs")
    for text, label in dataset:
        texts.append(text)
        labels.append(label)
    vecs = sbert.encode(texts)
    torch.save([(vec.squeeze(), label) for vec, label in zip(vecs, labels)], HOME+"data/SBERT_embeddings.pt")
    return

#%% Main
def main():
    test = '''I am 18yo M who has previously been diagnosed with depression, 
I have since come to realise that it isn't just that and have had weeks where I 
think my depression is coming back and then suddenly a few days of manic energy.
&amp;#x200B;
This post is inspired because I just came out of a fairly big for me at least 
phase over about 4 days of energy and elation where I bought and built a gaming 
PC with all my savings. I have since realised this is a terrible mistake and am 
not entirely sure why I did it and don't remember why I thought it was a good 
idea in the first place. I am almost certain I'd get a positive diagnosis and 
my family would aggree if I suggested that it may be the cause of these severe 
swings.
&amp;#x200B;
I've been pretty anxious of the affect getting a diagnosis will have on my family
  and am scared of going back on medication as getting off it for depression wasn't 
  fun. Is getting an official diagnosis worth it is basically what I'm asking? Thanks '''
    # vecer = w2vAverager(Word2Vec.load(HOME+"models/Word2Vec_v2").wv)
    # test_vec = vecer(full_preprocess(test))

    
    data = torch.load(HOME+"data/BERTavg_embeddings.pt"); random.shuffle(data)
    data = [(_,labels) for _,labels in data]
    train_data = ListDataset(data[:1000])
    val_data = ListDataset(data[1000:])
    
#%% sci-kit learn's Logistic Regression, k-NN, SVM, Random Forest  
    # # train_x = [sample[0].to_dense().squeeze() for sample in train_data]
    # train_x = [sample[0] for sample in train_data]
    # train_y = [sample[1].item() for sample in train_data]
    # # val_x = [sample[0].to_dense().squeeze() for sample in val_data]
    # val_x = [sample[0] for sample in val_data]
    # val_y = [sample[1].item() for sample in val_data]
    
    # logreg.fit(train_x, train_y, )
    # correct = 0; total = len(val_x)
    # pred = logreg.predict(val_x)
    # for y,p in zip(val_y, pred):
    #     if abs(p - y) < 0.5: correct += 1
    # print("Logistic Regression val acc:",correct / total)
    
    # knn.fit(train_x, train_y)
    # pred = knn.predict(val_x)
    # correct = 0
    # for y,p in zip(val_y, pred):
    #     if abs(p - y) < 0.5: correct += 1
    # print("k-Nearest Neighbors val acc:",correct / total)
    
    # svm.fit(train_x, train_y)
    # pred = svm.predict(val_x)
    # correct = 0
    # for y,p in zip(val_y, pred):
    #     if abs(p - y) < 0.5: correct += 1
    # print("Support Vector Machine val acc:",correct / total)
    
    # rdmforest.fit(train_x, train_y)
    # pred = rdmforest.predict(val_x)
    # correct = 0
    # for y,p in zip(val_y, pred):
    #     if abs(p - y) < 0.5: correct += 1
    # print("Random Forest val acc:",correct / total)
#%%

    # print(logreg.predict([test_vec]))    
    # print(knn.predict([test_vec]))
    # print(svm.predict([test_vec]))
    # print(rdmforest.predict([test_vec]))

    model = mlp
    #optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
    criterion = nn.BCELoss()
    model = train(
        model_name = "model",
        model = model, 
        train_data = train_data,
        val_data = val_data,
        criterion = criterion, 
        optimizer = optimizer, 
        num_epochs = 20,
        batch_size = 8,
        device = device,
        verbose = True,
    )
    
    return

# make_embeddings(HOME+"data/rbipolar_dataset-balanced.csv")
# make_dataset()
# a = torch.Tensor([[-5,3],[1,4]])
# b = torch.Tensor([[0,4],[2,3]])
# A = nn.Linear(2,1); B = nn.Linear(2,1)
# t1 = time.perf_counter()
# for i in range(100000):
#     x = (a - b).sum()
# print(x)
# print(time.perf_counter() - t1)
# t1 = time.perf_counter()
# for i in range(100000):
#     x = a.sub(b).sum()
# print(x)
# print(time.perf_counter() - t1)
# s = nn.Sequential(
#     nn.Linear(2,1),
#     nn.Linear(1,1),
# )
# m = MultiTaskModule(s)
# t = time.perf_counter()
# print(m.multitask_regularization_loss())
# print(time.perf_counter() - t)
from bertopic import BERTopic
# if __name__=="__main__": main()