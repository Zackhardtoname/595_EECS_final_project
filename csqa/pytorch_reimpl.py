import torch
import os
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer, BertTokenizerFast, BertForMultipleChoice
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import hjson
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.loggers import TensorBoardLogger

NUM_CHOICES = 5
writer = SummaryWriter()


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
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        sample = {key: self.x[key][idx] for key in self.x.keys()}
        sample["labels"] = self.y[idx]

        return sample  # , sample["labels"]


def acc_from_logits_and_labels(logits, labels):
    pred = torch.argmax(logits, dim=1)
    labels.to(pred.device)
    # acc = CrossEntropyLoss(logits, labels)
    acc = labels.eq(pred).sum() / labels.shape[0]
    return acc


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.pretrained_model = BertForMultipleChoice.from_pretrained('bert-base-uncased', return_dict=True)

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        # batch.pop("labels", None)
        outputs = self.forward(**batch)
        loss = outputs.loss
        logits = outputs.logits
        self.log(f'train_loss', loss)
        self.log(f'train_acc', acc_from_logits_and_labels(logits, labels))
        return loss

    def forward(self, **x):
        # in lightning, forward defines the prediction/inference actions
        outputs = self.pretrained_model(**x)
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            **batch
        )
        loss = outputs.loss
        reshaped_logits = outputs.logits.view(-1, NUM_CHOICES)
        acc = acc_from_logits_and_labels(reshaped_logits, batch["labels"])
        self.log(f"val_loss", loss)
        self.log(f"val_acc", acc)
        return loss

    def testing_step(self, batch, batch_idx):
        labels = batch["labels"]
        batch.pop("labels", None)

        outputs = self.forward(
            **batch
        )

        loss = outputs.loss
        reshaped_logits = outputs.logits.view(-1, NUM_CHOICES)
        acc = acc_from_logits_and_labels(reshaped_logits, labels)

        return {
            "loss": loss,
            "acc": acc
        }

    def testing_epoch_end(self, outputs):
        acc_li = [output["acc"] for output in outputs]
        self.log("test acc", sum(acc_li) / len(acc_li))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def preprocess(data, trim=False):
    q = data["question"]
    rep_q = [item for item in q for _ in range(5)]
    c = data["choices"]
    expanded_c = [e for ele in c for e in ele["text"]]
    x = tokenizer(rep_q, expanded_c, return_tensors='pt', padding=True, truncation=True,
                  max_length=config["max_seq_length"]).data
    end = 20 if trim else len(x["input_ids"])
    x = {k: v.view(-1, NUM_CHOICES, config["max_seq_length"])[:end] for k, v in x.items()}
    y = data["answerKey"][:end]
    y = torch.tensor([ord(item) - ord("A") for item in y])

    return x, y


if __name__ == "__main__":
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=3,
        verbose=True,
        mode="min"
    )

    pl.seed_everything(42)

    with open("config.hjson") as f:
        config = hjson.load(f)

    dataset = load_dataset("commonsense_qa")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    logger = TensorBoardLogger('lightning_logs', name='my_model')

    trainer = pl.Trainer(
        logger=logger,
        gpus=1,
        log_every_n_steps=5,  # each batch is a step
        max_epochs=config["max_epochs"]
    )
    model = Model()
    x_train, y_train = preprocess(dataset["train"])
    x_val, y_val = preprocess(dataset["validation"])
    # x_test, y_test = preprocess(dataset["test"])

    trainer.fit(
        model,
        DataLoader(DictDataset(x_train, y_train), batch_size=config["train_batch_size"], num_workers=8),
        DataLoader(DictDataset(x_val, y_val), batch_size=config["val_batch_size"], num_workers=8),
    )

    # trainer.test(test_dataloaders=DataLoader(DictDataset(x_test, y_test), batch_size=config["test_batch_size"]))
