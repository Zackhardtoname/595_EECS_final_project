import torch
try:
  import omegaconf
except:
  !pip install omegaconf
  !pip install hydra-core
  !pip install pytorch_lightning
import json
import random
import tqdm
#from pytorch_lightning.metrics.functional.classification import auroc, f1_score, accuracy
import pytorch_lightning as pl


def smooth_max(x, temp=1.0):
  return torch.sum(torch.softmax(x/temp, dim=-1) * x, dim=-1)

def pad_sentences(sents):
  return sents + ['' for x in range(7-len(sents))]

def get_pairs(words, label):
  if label == -1:
    allpairs = random.sample([(i,j) for i in range(len(words)) for j in range(len(words)) if i < j and i!=j], 7)
  else:
    allpairs = [(i, label) for i in range(label)]
  return [(words[i], words[j]) for i,j in allpairs]

def log1mexp(x):
    return torch.where(x > -0.693, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))

class Model(pl.LightningModule):

    def __init__(self, weight=None):
        super().__init__()
        # Download RoBERTa already finetuned for MNLI
        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
        self.roberta.register_classification_head('eat', num_classes=2)
    

    def forward(self, x):
        self.roberta.eval() # disable dropout

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
        words, label = batch
        pairs = get_pairs(words, label)
        batch = collate_tokens(
          [roberta.encode(pair[0], pair[1]) for pair in pairs], pad_idx=1
        )
        logits = roberta.predict('eat', batch)[:,0]
        prediction = smooth_max(logits, 0.1)
        if label != -1:
          loss = -prediction
        else:
          loss = -log1mexp(prediction)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pairs = get_pairs(words, label)
        batch = collate_tokens(
          [roberta.encode(pair[0], pair[1]) for pair in pairs], pad_idx=1
        )
        logits = roberta.predict('eat', batch)[:,0]
        prediction = smooth_max(logits, 0.1)
        preds.append(torch.exp(prediction))
        ys.append((label != -1).float())
        if label != -1:
          loss = -prediction
        else:
          loss = -log1mexp(prediction)
        self.log('val_loss', loss.item())


    def configure_optimizers(self):
        params = [v for n,v in self.roberta.named_parameters() if '.eat.' in n]
        opt = torch.optim.AdamW(params,lr=1e-5,eps=1e-6)
        return opt



model = Model()
# Need import this after loading the model
from fairseq.data.data_utils import collate_tokens
dev_file = '/content/drive/Shared drives/EECS595-Fall2020/Final_Project_Common/EAT/eat_train.json'
dev_data = json.load(open(dev_file))

sentences = [x['story'] for x in dev_data]
labels = torch.tensor([x['breakpoint'] for x in dev_data])
tokenized_sents = sum([list(zip(pad_sentences(x[:-1]), pad_sentences(x[1:]))) for x in sentences], [])
batch = collate_tokens(
    [model.roberta.encode(pair[0], pair[1]) for pair in tokenized_sents[:7]], pad_idx=1
)

torch.set_printoptions(sci_mode=False)
# Encode a pair of sentences and make a prediction
ema = 0
denom = 0
traindata = zip(sentences[len(sentences)//10:], labels[len(sentences)//10:])
testdata = zip(sentences[:len(sentences)//10], labels[:len(sentences)//10])

trainer = pl.Trainer(max_epochs=5, gpus=1)
trainer.fit(model, traindata, testdata)