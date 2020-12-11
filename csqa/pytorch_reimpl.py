from functools import partial

import hjson
import pytorch_lightning as pl
import ray
import ray.tune as tune
import torch
from datasets import load_dataset
from pytorch_lightning.callbacks import EarlyStopping
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMultipleChoice

NUM_CHOICES = 5
with open("config.hjson") as f:
    config = hjson.load(f)

pl.seed_everything(config["seed"])


def preprocess(data, tokenizer, trim=False):
    q = data["question"]
    rep_q = [item for item in q for _ in range(5)]
    c = data["choices"]
    expanded_c = [e for ele in c for e in ele["text"]]
    x = tokenizer(rep_q, expanded_c, return_tensors='pt', padding=True, truncation=True,
                  max_length=config["max_seq_length"]).data
    end = 200 if trim else len(x["input_ids"])
    x = {k: v.view(-1, NUM_CHOICES, config["max_seq_length"])[:end] for k, v in x.items()}
    y = data["answerKey"][:end]
    y = torch.tensor([ord(item) - ord("A") for item in y])

    return x, y


class DictDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x["input_ids"])

    def __getitem__(self, idx):
        sample = {key: self.x[key][idx] for key in self.x.keys()}
        sample["labels"] = self.y[idx]
        return sample


def acc_from_logits_and_labels(logits, labels, accuracy_fn):
    acc = accuracy_fn(logits.cpu(), labels.cpu())
    return acc


class Model(pl.LightningModule):
    def __init__(self, model_config):
        super().__init__()
        dataset = load_dataset("commonsense_qa")
        tokenizer = BertTokenizer.from_pretrained(config["pretrained_model_name"])
        self.x_train, self.y_train = preprocess(dataset["validation"], tokenizer, config["to_trim"])
        self.x_val, self.y_val = preprocess(dataset["validation"], tokenizer, config["to_trim"])
        self.pretrained_model = BertForMultipleChoice.from_pretrained(config["pretrained_model_name"], return_dict=True)
        self.accuracy = pl.metrics.Accuracy()
        self.model_config = model_config
        self.batch_size = 32

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        outputs = self.forward(**batch)
        loss = outputs.loss
        logits = outputs.logits
        self.log(f'train_loss', loss)
        acc = acc_from_logits_and_labels(logits, labels, self.accuracy)
        self.log(f'train_acc', acc)
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
        acc = acc_from_logits_and_labels(reshaped_logits, batch["labels"], self.accuracy)
        self.log(f"val_step_acc", acc)
        self.log(f"val_step_loss", loss)
        return {
            "val_step_loss": loss,
            "val_step_acc": acc,
        }

    def validation_epoch_end(self, outputs):
        # print(outputs)
        avg_loss = torch.tensor(
            [x["val_step_loss"] for x in outputs]).mean()
        avg_acc = torch.tensor(
            [x["val_step_acc"] for x in outputs]).mean()
        self.log("val_epoch_loss", avg_loss)
        self.log("val_epoch_acc", avg_acc)

    # def testing_step(self, batch, batch_idx):
    #     labels = batch["labels"]
    #     batch.pop("labels", None)
    #
    #     outputs = self.forward(
    #         **batch
    #     )
    #
    #     loss = outputs.loss
    #     reshaped_logits = outputs.logits.view(-1, NUM_CHOICES)
    #     acc = acc_from_logits_and_labels(reshaped_logits, labels, self.accuracy)
    #
    #     return {
    #         "loss": loss,
    #         "acc": acc
    #     }
    #
    # def testing_epoch_end(self, outputs):
    #     acc_li = [output["acc"] for output in outputs]
    #     self.log("test acc", sum(acc_li) / len(acc_li))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.model_config.get("lr", 2e-5))
        return optimizer

    def train_dataloader(self):
        return DataLoader(DictDataset(self.x_train, self.y_train), batch_size=self.batch_size,
                          num_workers=8)

    def val_dataloader(self):
        return DataLoader(DictDataset(self.x_val, self.y_val), batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == "__main__":
    early_stop_callback = EarlyStopping(
        monitor="val_epoch_loss",
        min_delta=0.0,
        patience=3,
        verbose=True,
        mode="min"
    )

    if config["use_gpu"]:
        ray.init(num_gpus=1)

    rt_config = {
        # "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32])
    }

    callback = TuneReportCallback(
        {
            "loss": "val_epoch_loss",
            "acc": "val_epoch_acc"
        },
        on="validation_end")


    def train_tune(config, epochs=3, gpus=1):
        model = Model(config)
        trainer = pl.Trainer(
            max_epochs=epochs,
            gpus=gpus,
            # auto_scale_batch_size="power",
            # progress_bar_refresh_rate=20,
            callbacks=[callback])
        trainer.fit(model)


    analysis = tune.run(
        partial(
            train_tune, epochs=1, gpus=config["use_gpu"],
        ),
        config=rt_config,
        num_samples=1,
        resources_per_trial={
            "cpu": 10,
            "gpu": config["use_gpu"]
        },
        metric="acc",
        mode="max",
    )

    print("Best config: ", analysis.best_config)
    # trainer.test(test_dataloaders=DataLoader(DictDataset(x_test, y_test), batch_size=config["test_batch_size"]))
