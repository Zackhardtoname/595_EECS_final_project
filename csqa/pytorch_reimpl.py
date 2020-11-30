import torch
import os
from datasets import load_dataset
from transformers import BertTokenizer, BertForMultipleChoice
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch

dataset = load_dataset("commonsense_qa")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMultipleChoice.from_pretrained('bert-base-uncased', return_dict=True)

q = dataset["train"]["question"]
rep_q = [item for item in q for _ in range(5)]
c = dataset["train"]["choices"]
expanded_c = [e for ele in c for e in ele["text"]]
labels = dataset["train"]["answerKey"]

encoding = tokenizer(rep_q, expanded_c, return_tensors='pt', padding=True, truncation=True, max_length=32)
outputs = model(**{k: v for k,v in encoding.items()}, labels=labels)  # batch size is 1

# the linear classifier still needs to be trained
loss = outputs.loss
logits = outputs.logits

class Model(pl.LightningModule):


if __name__ == "__main__":
    early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.0,
            patience=3,
            verbose=True,
            mode="min"
            )

    trainer = pl.Trainer(
            gpus=1,
            early_stop_callback=early_stop_callback,
            )

    model = Model()

    trainer.fit(model)
    trainer.test()

# for ele in c:
#     if len(ele["text"]) != 5:
#         print(ele["text"])

# lens = [len(e) for e in rep_q]
# max_len = max(lens)
# max_len_idx = lens.index(max_len)
# print(max_len, max_len_idx)
# to_decode = tokenizer(rep_q[max_len_idx], rep_q[max_len_idx], return_tensors='pt', padding=True, truncation=True, max_length=32)["input_ids"][0]
# print(tokenizer.decode(to_decode))