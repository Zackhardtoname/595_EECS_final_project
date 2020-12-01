try:
  from transformers import AutoTokenizer, AutoModel, AutoModelForNextSentencePrediction
except:
  !pip install transformers
  from transformers import AutoTokenizer, AutoModel, AutoModelForNextSentencePrediction
import torch
import numpy as np
import json
from glob import glob
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import tqdm
try:
  import pytorch_lightning as pl
except:
  !pip install pytorch_lightning
  import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
import tqdm
import math

def pad_sentences(sents):
  return sents + ['' for x in range(7-len(sents))]

def kfold(tensor, index):
  return torch.cat([tensor[:index*tensor.shape[0]//10], tensor[(index+1)*tensor.shape[0]//10:]])

def kfoldtest(tensor, index):
  return tensor[index*tensor.shape[0]//10:(index+1)*tensor.shape[0]//10]

bertmodel = 'bert-large-uncased'

class Headv1(nn.Module):
    def __init__(self):
      super().__init__()
      self.layer = nn.Sequential(nn.Linear(768, 1000), nn.ReLU(), nn.Linear(1000, 1))
    
    def forward(self, x):
      return self.layer(x)

class Headv2(nn.Module):
    def __init__(self):
      super().__init__()
      self.dim = 200
      self.numheads = 8
      self.q1 = nn.Linear(768+4, self.dim*self.numheads)
      self.k1 = nn.Linear(768+4, self.dim*self.numheads)
      self.v1 = nn.Linear(768+4, 400*self.numheads)


      self.relu = nn.ReLU()
      self.conv = nn.Conv2d(self.numheads, 1, 1)
      
      self.q2 = nn.Linear(400+4, self.dim)
      self.k2 = nn.Linear(400+4, self.dim)
      self.v2 = nn.Linear(400+4, 1)
    
    def forward(self, x):
      t = torch.linspace(0, 2*math.pi*7/8, 7, device=x.device)
      embeddings = torch.stack([torch.sin(t/2.0), torch.cos(t/2.0), torch.sin(t), torch.cos(t)], dim=1).unsqueeze(0).repeat(x.shape[0], 1, 1)
      x = torch.cat([x, embeddings], dim=2)
      qs = self.q1(x)
      qs = qs.view(*qs.shape[:-1], self.numheads, -1).transpose(-2,-3)
      ks = self.k1(x)
      ks = ks.view(*ks.shape[:-1], self.numheads, -1).transpose(-2,-3)
      attention = torch.softmax(torch.matmul(qs, ks.transpose(-1,-2))/math.sqrt(self.dim), dim=-1)
      vs = self.v1(x)
      vs = vs.view(*vs.shape[:-1], self.numheads, -1).transpose(-2,-3)
      out = torch.matmul(attention, vs)
      out = self.conv(self.relu(out)).squeeze(1)
      out = torch.cat([out, embeddings], dim=2)
      qs = self.q2(out)
      ks = self.k2(out)
      attention = torch.softmax(torch.bmm(qs, ks.transpose(-1,-2))/math.sqrt(self.dim), dim=-1)
      out = torch.bmm(attention, self.v2(out)).squeeze(-1)
      # Pad for no breaking point prediction
      #out = torch.nn.functional.pad(out, (0,1))
      return out

class Model(pl.LightningModule):

    def __init__(self, weight=None):
        super().__init__()
        self.encoder = AutoModelForNextSentencePrediction.from_pretrained(bertmodel, return_dict=True)
        self.encoder.eval()
        for param in self.encoder.base_model.parameters():
          param.requires_grad = False
        self.head = Headv2()
        if weight is not None:
          self.criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=weight)
        else:
          self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    def forward(self, x):
        self.encoder.eval()
        # in lightning, forward defines the prediction/inference actions
        input_ids, token_type_ids, attention_mask = x
        s = input_ids.shape
        input_ids = input_ids.view(-1, s[-1])
        token_type_ids = token_type_ids.view(-1, s[-1])
        attention_mask = attention_mask.view(-1, s[-1])
        
        out = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logits = out.logits
        return logits

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        a, b, c, y = batch
        x = (a,b,c)
        logits = self(x)
        logits = torch.softmax(logits.view(-1, 7, 2), dim=-1)
        print(y[:10])
        print(logits[:10])
        #y[y == -1] = 7
        1/0
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).view(-1)
        predictions = ((logits > 0) & (y == 1)) | ((logits < 0) & (y == 0))
        self.log('accuracy', predictions.float().sum()/(y != -1).float().sum())


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0003)
        return optimizer

'''tokenizer = BertTokenizer.from_pretrained(bertmodel)

prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
next_sentence = "It usually has more cheese."
encoding = tokenizer(prompt, next_sentence, return_tensors='pt')
encoding = merge([encoding, encoding])
outputs = model(**encoding)
print(outputs)
logits = torch.softmax(outputs.logits, dim=1)
print(logits)'''

# Prepare Input
tokenizer = AutoTokenizer.from_pretrained(bertmodel)

dev_file = '/content/drive/Shared drives/EECS595-Fall2020/Final_Project_Common/EAT/eat_train.json'
dev_data = json.load(open(dev_file))

sentences = [x['story'] for x in dev_data]
labels = torch.tensor([x['breakpoint'] for x in dev_data])
tokenized_sents = sum([list(zip(pad_sentences(x[:-1]), pad_sentences(x[1:]))) for x in sentences], [])
first, second = zip(*tokenized_sents)
input_ids = tokenizer(first, second, return_tensors='pt', padding=True)
maxlen = input_ids['input_ids'].shape[-1]

batchsize = 40
accuracy = pl.metrics.Accuracy()
fbeta = pl.metrics.Fbeta(7)
totalaccuracy = 0.0
for split in range(1,10):
  traindata = DataLoader(TensorDataset(kfold(input_ids['input_ids'].view(-1, 7, maxlen), split), kfold(input_ids['token_type_ids'].view(-1, 7, maxlen), split), kfold(input_ids['attention_mask'].view(-1, 7, maxlen), split), kfold(labels, split)), batch_size=batchsize)
  testdata = DataLoader(TensorDataset(kfoldtest(input_ids['input_ids'].view(-1, 7, maxlen), split), kfoldtest(input_ids['token_type_ids'].view(-1, 7, maxlen), split), kfoldtest(input_ids['attention_mask'].view(-1, 7, maxlen), split), kfoldtest(labels, split)), batch_size=batchsize)
  
  counts = torch.zeros(7)
  counts[:-1] = torch.bincount(labels[labels != -1])
  counts = counts/counts.sum()
  counts = 0.2/counts

  model = Model(counts)
  trainer = pl.Trainer(max_epochs=5, gpus=1)
  trainer.fit(model, traindata)
  model.cuda()
  model.eval()
  prob = 0.0
  total = 0.0
  with torch.no_grad():
    for x, y in testdata:
      yhat = torch.softmax(model(x.cuda()), dim=-1).cpu()
      prob += yhat[y != -1][range(len(y[y != -1])), y[y != -1]].sum()
      total += len(y[y != -1])
      accuracy(yhat[y != -1,:], y[y != -1].cpu())
      fbeta(yhat[y != -1,:], y[y != -1].cpu())
  a = accuracy.compute()
  totalaccuracy += a
  print(a)
  print(fbeta.compute())
print(totalaccuracy/10.0)