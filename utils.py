#%% Imports
import os, time, math, re, copy
import datetime as dt
from typing import Callable
from collections.abc import Iterable

import wmi
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from gensim.models import KeyedVectors
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer
# from rake_nltk import Rake
# from pke.unsupervised import YAKE

import torch
from torch.utils.data import DataLoader

MAX_OPERATING_TEMP = 85 # Celsius
#%% 

def to_unixtime(year, month=1, day=1, hour=0, minute=0, second=0) -> int:
    return int(dt.datetime(year, month, day, hour, minute, second).timestamp())

def to_clocktime(unix_time) -> str:
    return dt.datetime.utcfromtimestamp(unix_time).strftime("%Y-%m-%d %H:%M")

#punctuation = set("""!"#$%&'()*+,-—./:;<=>?@[\]^_`{|}~’“”…""")
match_nonseparator: re.Pattern = re.compile(r"['’]")
match_puncdigit: re.Pattern = re.compile(r"[^a-zA-Z \n\r\t\f]")
match_word: re.Pattern = re.compile(r"\b\w+\b")
tokenizer = TreebankWordTokenizer()
en_stopwords = set([ match_puncdigit.sub("", stop_word) 
    for stop_word in
        stopwords.words("english") + ['im','id','ive','hed','shed','yall',
                                      'could','couldve','would', ]
])
reddit_words = {'nan','removed','com','org','gov','net','edu','io','ly',
                'amp','&amp;','#x200b;','&amp;#x200B;','[removed]','[deleted]'}
lemmatizer = WordNetLemmatizer()
def full_preprocess(text: str, join=True):
    ''' lowers case, removes punc, tokenizes, removes stop words, lemmatizes '''
    text = text.lower()
    text = match_nonseparator.sub("", text)
    text = match_puncdigit.sub(" ", text) # ''.join is faster when there are a lot of matches
    text = match_word.findall(text) # basic regex tokenizer
    #text = tokenizer.tokenize(text) # Penn Treebank (regex) tokenizer
    text = [ lemmatizer.lemmatize(word)
        for word in text 
            if word not in en_stopwords 
            and word not in reddit_words
            and len(word) > 2
    ]
    if join: text = ' '.join(text)
    return text

def light_preprocess(text: str) -> str:
    ''' removes reddit-specific words '''
    return ' '.join([ word 
                     for word in text.split()
                         if word not in reddit_words
                         and word not in en_stopwords
                    ])

def ngrams(tokens: list, n: int=2) -> list:
    return [ ' '.join(ngram)
        for ngram in zip( *[ tokens[i:] for i in range(0, n)] )
    ]

class TFIDFVectorizerWrapper:
    def __init__(self, tfidf): self.tfidf = tfidf
    def __call__(self, document):
        out = self.tfidf.transform([' '.join(document)])
        return torch.from_numpy(out.tocoo().todense()).to_sparse()

class w2vAverager:
    def __init__(self, kv: KeyedVectors): self.kv = kv
    def __call__(self, document: list):
        vec = np.zeros(self.kv.vector_size)
        n = 0
        for word in document:
            if word in self.kv:
                n += 1
                vec += np.array(self.kv[word])
        if n==0:
            return vec
        return vec / n

class BertVectorizer:
    def __init__(self, bert, tokenizer, device=None):
        self.bert = bert
        self.tokenizer = tokenizer
        self.device = device
    
    def __call__(self, document: list) -> torch.Tensor:
        out = self.tokenizer(document, max_length=512, truncation=True, padding=False, return_tensors='pt')
        if self.device is not None: out.to(self.device)
        out = self.bert(**out).last_hidden_state.squeeze()
        out = torch.mean(out, 0)
        return out
        
class RBipolarDatasetBuilder():
    def __init__(
        self, 
        file_path: pd.DataFrame = None, 
        preprocess: Callable=None, 
        vectorizer: Callable=None,
        n_grams: int=0
    ):
        if file_path is not None:
            self.df = pd.read_csv(file_path) if file_path[-3:]=="csv" else pd.read_excel(file_path)
        else:
            self.df = None
        self.preprocess = preprocess
        self.vectorizer = vectorizer
        self.n_grams = n_grams # word grams
        self._iter_idx = -1
        
    def __len__(self): return len(self.df)
    
    def process_text(self, text):
        if self.preprocess is not None:
            text = self.preprocess(text)
            if self.n_grams > 1 and isinstance(text, list):
                text = ngrams(text, self.n_grams)
        if self.vectorizer is not None:
            text = self.vectorizer(text)
            if not isinstance(text, torch.Tensor):
                #text = torch.Tensor(text)
                text = torch.as_tensor(text, dtype=torch.float32)
        return text
    
    def __getitem__(self, index):
        body, title = self.df.iloc[index][['selftext','title']]
        text = self.process_text(str(title) +' '+ str(body))
        # text = self.process_text(str(self.df.iloc[index]['text']))
        labels = torch.as_tensor(self.df.iloc[index][['anxiety','bipolar','depression']], dtype=torch.float32)
        return text, labels
    
    def __iter__(self):
        self._reset_iter()
        return self
    
    def __next__(self):
        self._iter_idx += 1
        if self._iter_idx < len(self): return self.__getitem__(self._iter_idx)
        else: raise StopIteration
    
    def _reset_iter(self): self._iter_idx = -1
    

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, data: list):            self._data = data
    def __len__(self):              return len(self._data)
    def __getitem__(self, index):       return self._data[index] # returns (x,y) tuples

def IDF(docs: Iterable[Iterable]) -> dict:
    doc_freqs = {}
    n_docs = 0
    for document in docs:
        n_docs += 1
        
        vocab = set(document)
        for token in vocab:
            if token not in doc_freqs.keys():
                doc_freqs[token] = 1
            else:
                doc_freqs[token] += 1
    
    # return tokens mapped to their IDF score
    return {token: math.log2(n_docs / frequency)
              for token, frequency in doc_freqs.items()}

def TF_IDF(document: Iterable[str], idf_scores: dict) -> dict:
    token_freqs = {}
    n_tokens = 0
    for token in document:
        n_tokens += 1
        if token not in token_freqs.keys():
            token_freqs[token] = 1
        else:
            token_freqs[token] += 1
    
    return {token: (frequency / n_tokens) * idf_scores[token]
            for token, frequency in token_freqs.items()}

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=(1,1,1), from_logits=True):
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.nn.Parameter(torch.Tensor(alpha))
        self.from_logits = from_logits
        self._epsilon = 1e-4

    def forward(self, outputs, truths):
        p = torch.sigmoid(outputs.reshape(-1)) if self.from_logits else outputs.reshape(-1)
        p = torch.where(truths.reshape(-1) >= 0.5, p, 1-p)
        logp = - torch.log(torch.clamp(p, self._epsilon, 1-self._epsilon))
        loss = ((1-p)**self.gamma)*logp
        return loss.mean()

def log_print(file, msg):
    if os.path.isfile(file):
        with open(file, 'a') as f:
            f.write(f"{msg}\n")
    print(msg)

def train(
    model_name:  str,
    model:       torch.nn.Module, 
    train_data:  torch.utils.data.Dataset,
    val_data:    torch.utils.data.Dataset,
    criterion:   torch.nn.Module, 
    optimizer:   torch.optim.Optimizer, 
    scheduler:   object = None, 
    num_epochs:  int = 0, 
    batch_size:  int = 1,
    num_workers: int = 0,
    logits:      bool = True,
    device:      torch.device = torch.device("cpu"),
    verbose:     bool = True,
    exp:         int = -1,
    delay:       int = 100,
    logging:     str = None, # file path
) -> torch.nn.Module:
    
    print(f"Begining Training of {model_name}...")
    if verbose: print("\t\t\t Progress \t\t\t\t\t\t\t Last Epoch Stats:")
    start_time = time.perf_counter()
    
    loss_history, train_acc_history, val_acc_history = [], [], []
    last_acc = 0.0

    dataset_size = torch.Tensor([len(train_data)])
    data_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True, num_workers=num_workers)
    
    model.to(device)
    model.train()
    epoch_loss, epoch_acc, acc_dif, val_acc = torch.zeros(1),torch.zeros(1),torch.zeros(1), torch.zeros(1)
    epoch_iterator = tqdm(range(num_epochs), "Training") if not verbose else range(num_epochs)
    for epoch in epoch_iterator:
        if not verbose: epoch_iterator.set_description(f"Training; Loss: {epoch_loss.item():.3f} Val Acc: {val_acc.item():.2%}")
        running_loss = 0
        running_corrects = 0

        batch_iterator = tqdm(
            iterable = data_loader,
            desc = f"Epoch {epoch+1}/{num_epochs}",
            bar_format = "{desc}: |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}",
            postfix = f"Loss: {epoch_loss.item():.4f}\t\
Accuracy: {epoch_acc.item():.2%} ({'+' if acc_dif.item()>=0 else ''}{acc_dif.item():.2%})",
        ) if verbose else data_loader
        i = 0
        for x, gt in batch_iterator:
            x = x.to(dtype=torch.float32, device=device)
            gt = gt.to(dtype=torch.float32, device=device)
            
            # forward
            output = model(x)
            #_, preds = torch.max(output, 1)
            loss = criterion(output, gt) #TODO: loss for param share
            
            # backward                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistics
            if i % delay == 0: loss_history.append(loss.detach().cpu().item())
            current_batch_size = x.size(0)
            running_loss += loss.item() * current_batch_size
            running_corrects += torch.sum(abs( (torch.sigmoid(output) if logits else output) - (gt >= 0.5).int() < 0.5)).cpu()
            #if abs((output - y).item()) < 0.5: running_corrects += 1
            
            i += 1

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / (gt.size(1) * dataset_size)
        #epoch_acc = torch.Tensor([running_corrects / dataset_size])
        val_acc = evaluate(model, val_data, device)
        
        if scheduler != None:
            if type(scheduler) == torch.optim.lr_scheduler.OneCycleLR:
                if not epoch > scheduler.total_steps:
                    scheduler.step()
            else:
                scheduler.step(epoch_acc)

        train_acc_history.append(epoch_acc.detach().cpu().item())
        val_acc_history.append(val_acc.detach().cpu().item())
        acc_dif = epoch_acc - last_acc
        last_acc = epoch_acc
        
    time_elapsed = time.perf_counter() - start_time
    print(f'Training completed in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    print(f'Final Training Accuracy: {epoch_acc.item():.2%}')
    print(f"Final Validation Accuracy: {val_acc.item():.2%}")
    
    #plot loss
    plt.xlabel("Iteration"); plt.ylabel("Loss")
    plt.plot(range(0,len(loss_history)*delay, delay), loss_history)
    plt.grid(True)
    # if exp >= 0:
    #     plt.savefig(f"./runs/{model_name}/exp{exp}/iter_loss.png")
    plt.show()
    
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.plot(range(0,len(train_acc_history)), train_acc_history, label="Train")
    plt.plot(range(0, len(val_acc_history)), val_acc_history, label="Validation")
    plt.legend()
    plt.grid(True)
    # if exp >= 0:
    #     plt.savefig(f"./runs/{model_name}/exp{exp}/iter_accuracy.png")
    plt.show()

    return model

def evaluate(
    model:         torch.nn.Module,
    val_data:      torch.utils.data.Dataset,
    device:        torch.device=torch.device("cpu"),
    logits:        bool = True,
) -> torch.Tensor: 
    dataloader = DataLoader(val_data, batch_size=16)
    model.eval()
    corrects, total = 0,0
    with torch.no_grad():
        for x, gt in dataloader:
            x = x.to(device)
            gt = (gt >= 0.5).to(device=device, dtype=torch.int32)
            
            output = model(x)
            if logits: output = torch.sigmoid(output)
            
            corrects += torch.sum(abs(output - gt) < 0.5)
            total += x.size(0) * gt.size(1)
    
    return torch.Tensor([corrects / total])

def train_topicmtm(
    model_name:  str,
    model:       torch.nn.Module, 
    train_data:  torch.utils.data.Dataset,
    val_data:    torch.utils.data.Dataset,
    criterion:   torch.nn.Module, 
    optimizer:   torch.optim.Optimizer, 
    scheduler:   object = None, 
    # decay_regmtm:float = 1.,
    decay_reg:   float = 1.,
    num_epochs:  int = 0, 
    batch_size:  int = 1,
    num_workers: int = 0,
    device:      torch.device = torch.device("cpu"),
    verbose:     bool = True,
    exp:         int = -1,
    delay:       int = 2,
    logging:     str = None, # file path
) -> torch.nn.Module:
    
    global MAX_OPERATING_TEMP
    print(f"Begining Training of {model_name}...")
    if verbose: print("\t\t\t Progress \t\t\t\t\t\t\t Last Epoch Stats:")
    start_time = time.perf_counter()
    w = wmi.WMI(namespace="root\OpenHardwareMonitor")
    
    loss_history, train_acc_history, val_acc_history, f1macro_history, f1micro_history = [], [], [], [], []
    last_acc = 0.0

    dataset_size = torch.Tensor([len(train_data)])
    data_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True, num_workers=num_workers)
    
    model.to(device)
    model.train()
    epoch_loss, epoch_acc, acc_dif, val_acc = torch.zeros(1),torch.zeros(1),torch.zeros(1), torch.zeros(1)
    f1micro, f1macro = 0,0
    epoch_iterator = tqdm(range(num_epochs), "Training") if not verbose else range(num_epochs)
    for epoch in epoch_iterator:
        if epoch % 100 == 0:
            temperature_infos = w.Sensor()
            for sensor in temperature_infos:
                if sensor.SensorType==u'Temperature' and sensor.Value > MAX_OPERATING_TEMP:
                    if not verbose: epoch_iterator.set_description("Cooling...")
                    time.sleep(10)
        
        if not verbose: epoch_iterator.set_description(f"Training; Loss: {epoch_loss.item():.3f} Val Acc: {val_acc.item():.2%} F1-Macro: {f1macro:.2%}")
        running_loss = 0
        running_corrects = 0

        batch_iterator = tqdm(
            iterable = data_loader,
            desc = f"Epoch {epoch+1}/{num_epochs}",
            bar_format = "{desc}: |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}",
            postfix = f"Loss: {epoch_loss.item():.4f}\t\
Accuracy: {epoch_acc.item():.2%} ({'+' if acc_dif.item()>=0 else ''}{acc_dif.item():.2%})",
        ) if verbose else data_loader
        i = 0
        for (x,topic), gt in batch_iterator:
            x = x.to(dtype=torch.float32, device=device)
            topic = topic.to(dtype=torch.float32, device=device)
            gt = gt.to(dtype=torch.float32, device=device)
            
            # forward
            output = model(x, topic)
            #_, preds = torch.max(output, 1)
            loss = criterion(output, gt) #TODO: loss for param share
            
            # backward                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistics
            current_batch_size = x.size(0)
            running_loss += loss.item() * current_batch_size
            running_corrects += torch.sum(torch.abs(torch.sigmoid(output) - gt) < 0.5).detach()
            #if abs((output - y).item()) < 0.5: running_corrects += 1
            
            i += 1

        loss_history.append(loss.detach().cpu().item())
        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double().cpu() / (gt.size(1) * dataset_size)
        #epoch_acc = torch.Tensor([running_corrects / dataset_size])
        #val_acc = evaluate_topic(model, val_data, device)
        
        if scheduler != None:
            if type(scheduler) == torch.optim.lr_scheduler.OneCycleLR or type(scheduler) == torch.optim.lr_scheduler.StepLR:
                if epoch < scheduler.total_steps:
                    scheduler.step()
            else:
                scheduler.step(epoch_acc)

        if decay_reg < 1 and model.mtmreg_delta > 1e-9 and epoch % (num_epochs // 10) == 0 and epoch > 0:
            model.mtmreg_delta *= decay_reg
            if verbose: print(f"Updating Regularization Delta to {model.mtmreg_delta.data}")
        if epoch % delay == 0:
            f1macro, f1micro = evaluate_topicreport(model, val_data, device)
            val_acc = evaluate_topic(model, val_data, device)
            f1macro_history.append(f1macro)
            f1micro_history.append(f1micro)
            train_acc_history.append(epoch_acc.detach().cpu().item())
            val_acc_history.append(val_acc.detach().cpu().item())
        acc_dif = epoch_acc - last_acc
        last_acc = epoch_acc
        
    time_elapsed = time.perf_counter() - start_time
    if verbose: print(f'Training completed in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s                ')
    print(f'Final Training Accuracy:\t {epoch_acc.item():.2%}')
    print(f"Final Validation Accuracy:\t {val_acc.item():.2%}")
    print(f"Final Unweighted Avg F1-score:\t {f1macro:.2%}")
    print(f"Final Weighted Avg F1-score:\t {f1micro:.2%}")
    
    #plot loss
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.plot(range(0,len(loss_history), 1), loss_history)
    plt.grid(True)
    # if exp >= 0:
    #     plt.savefig(f"./runs/{model_name}/exp{exp}/iter_loss.png")
    plt.show()
    
    plt.xlabel("Epoch"); plt.ylabel("Percent")
    plt.plot(range(0,len(train_acc_history)*delay, delay), train_acc_history, label="Train")
    plt.plot(range(0, len(val_acc_history)*delay, delay), val_acc_history, label="Validation")
    plt.plot(range(0, len(f1macro_history)*delay, delay), f1macro_history, label="Macro-F1")
    plt.plot(range(0, len(f1micro_history)*delay, delay), f1micro_history, label="Micro-F1")
    plt.legend()
    plt.grid(True)
    # if exp >= 0:
    #     plt.savefig(f"./runs/{model_name}/exp{exp}/iter_accuracy.png")
    plt.show()

    return model, (epoch_acc.item(), val_acc.item(), f1macro, f1micro)

def evaluate_topic(
    model:         torch.nn.Module,
    val_data:      torch.utils.data.Dataset,
    device:        torch.device=torch.device("cpu"),
) -> torch.Tensor: 
    dataloader = DataLoader(val_data, batch_size=32)
    model.eval()
    corrects, total = 0,0
    with torch.no_grad():
        for (x,topic), gt in dataloader:
            x = x.to(dtype=torch.float32, device=device)
            topic = topic.to(dtype=torch.float32, device=device)
            gt = (gt >= 0.5).to(dtype=torch.float32, device=device)
            
            output = torch.sigmoid(model(x, topic))
            
            corrects += torch.sum(abs(output - gt) < 0.5)
            total += x.size(0) * gt.size(1)
    
    return torch.Tensor([corrects / total])
        
def evaluate_topicreport(
    model:         torch.nn.Module,
    val_data:      torch.utils.data.Dataset,
    device:        torch.device=torch.device("cpu"),
) -> torch.Tensor: 
    dataloader = DataLoader(val_data, batch_size=1024)
    model.eval()
    tot_f1macro, tot_f1micro, total = 0,0,0
    with torch.no_grad():
        for (x,topic), gt in dataloader:
            x = x.to(dtype=torch.float32, device=device)
            topic = topic.to(dtype=torch.float32, device=device)
            gt = (gt >= 0.5).to(dtype=torch.int32).transpose(0,1).cpu()
            
            output = torch.sigmoid(model(x, topic)).cpu()
            output = (output.transpose(0,1) >= 0.5).to(torch.int32)
            f1_macro, f1_micro = 0,0
            for task in range(3):
                task_output = output[task]
                task_gt = gt[task]
                # print(task_output)
                # print(task_gt)
                report = classification_report(task_gt, task_output, output_dict=True)
                f1_macro += report['macro avg']['f1-score']
                f1_micro += report['weighted avg']['f1-score']
            
            #corrects += torch.sum(abs(output - gt) < 0.5)
            tot_f1macro += f1_macro / 3
            tot_f1micro += f1_micro / 3
        
            total += 1
            #total += x.size(0) * gt.size(1)
    
    return tot_f1macro / total, tot_f1micro / total