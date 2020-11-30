import torch
import os
from datasets import load_dataset
from transformers import BertTokenizer, BertForMultipleChoice
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DictDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, x, y):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x["input_ids"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {key: x[key][idx] for key in x.keys()}
        sample["labels"] = y[idx]

        return sample


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = BertForMultipleChoice.from_pretrained('bert-base-uncased', return_dict=True)
        self.accuracy = pl.metrics.Accuracy()

    def training_step(self, batch, batch_idx):
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        self.log(f'batch {batch_idx} training acc', self.accuracy(logits, y))
        return loss

    # def forward(self, x):
    #     # in lightning, forward defines the prediction/inference actions
    #     outputs = model.forward(**x)
    #     return outputs.logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=3,
        verbose=True,
        mode="min"
    )

    set_seed(42)
    dataset = load_dataset("commonsense_qa")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    q = dataset["train"]["question"]
    rep_q = [item for item in q for _ in range(5)]
    c = dataset["train"]["choices"]
    expanded_c = [e for ele in c for e in ele["text"]]
    x = tokenizer(rep_q, expanded_c, return_tensors='pt', padding=True, truncation=True, max_length=32).data
    y = dataset["train"]["answerKey"]
    rep_y = [item for item in y for _ in range(5)]

    trainer = pl.Trainer(
        gpus=1,
    )
    model = Model()

    trainer.fit(model, DataLoader(DictDataset(x, rep_y)))
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
