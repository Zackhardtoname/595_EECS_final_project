import os
from functools import partial
from glob import glob

import hjson
import pytorch_lightning as pl
import ray
import ray.tune as tune
import torch
from datasets import load_dataset
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMultipleChoice, PretrainedConfig, AlbertTokenizer, \
    AlbertForMultipleChoice

NUM_CHOICES = 5
with open("config.hjson") as f:
    config = hjson.load(f)

pl.seed_everything(config["seed"])
train_step = 0


def preprocess(data, tokenizer, max_length, trim=False):
    q = data["question"]
    rep_q = [item for item in q for _ in range(5)]
    c = data["choices"]
    expanded_c = [e for ele in c for e in ele["text"]]
    x = tokenizer(rep_q, expanded_c, return_tensors='pt', padding='max_length', truncation=True,
                  max_length=max_length).data
    end = 200 if trim else len(x["input_ids"])
    x = {k: v.view(-1, NUM_CHOICES, max_length)[:end] for k, v in x.items()}
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
        self.model_config = model_config
        dataset = load_dataset("commonsense_qa")
        tokenizer = self.model_config["architecture"]["tokenizer"].from_pretrained(
            self.model_config["architecture"]["pretrained_model_name"])
        training_data = dataset["train"][config["test_size"]:]
        test_data = dataset["train"][:config["test_size"]]
        self.x_train, self.y_train = preprocess(training_data, tokenizer, self.model_config["max_seq_length"],
                                                config["to_trim"])
        # self.example_input_array = self.val_dataloader()
        self.x_test, self.y_test = preprocess(test_data, tokenizer, self.model_config["max_seq_length"],
                                              config["to_trim"])
        self.x_val, self.y_val = preprocess(dataset["validation"], tokenizer, self.model_config["max_seq_length"],
                                            config["to_trim"])
        self.pretrained_model = self.model_config["architecture"]["model"].from_pretrained(
            self.model_config["architecture"]["pretrained_model_name"],
            hidden_act=self.model_config[
                "hidden_act"],
            hidden_dropout_prob=self.model_config[
                "hidden_dropout_prob"], return_dict=True)
        self.accuracy = pl.metrics.Accuracy()
        self.batch_size = self.model_config.get("batch_size", 32)

    def log_each_step(self, name, val, on_step=True, on_epoch=True):
        self.log(name, val, prog_bar=True, on_step=on_step, on_epoch=on_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.model_config.get("lr", 2e-5))
        return optimizer

    def train_dataloader(self):
        return DataLoader(DictDataset(self.x_train, self.y_train), batch_size=self.batch_size,
                          num_workers=8)

    def val_dataloader(self):
        return DataLoader(DictDataset(self.x_val, self.y_val), batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(DictDataset(self.x_test, self.y_test), batch_size=self.batch_size, num_workers=8)

    def forward(self, **x):
        # in lightning, forward defines the prediction/inference actions
        outputs = self.pretrained_model(**x)
        return outputs

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        outputs = self.forward(**batch)
        loss = outputs.loss
        logits = outputs.logits
        self.log_each_step(f'train_loss', loss)
        acc = acc_from_logits_and_labels(logits, labels, self.accuracy)
        self.log_each_step(f'train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        loss = outputs.loss
        reshaped_logits = outputs.logits.view(-1, NUM_CHOICES)
        acc = acc_from_logits_and_labels(reshaped_logits, batch["labels"], self.accuracy)
        self.log_each_step(f"val_step_acc", acc)
        self.log_each_step(f"val_step_loss", loss)

        return {
            "val_step_loss": loss,
            "val_step_acc": acc,
        }

    def validation_epoch_end(self, outputs):
        print("\n\nvalidation_epoch_end\n\n")
        avg_loss = torch.tensor(
            [x["val_step_loss"] for x in outputs]).mean()
        avg_acc = torch.tensor(
            [x["val_step_acc"] for x in outputs]).mean()
        self.log_each_step("val_epoch_loss", avg_loss, on_step=False)
        self.log_each_step("val_epoch_acc", avg_acc, on_step=False)

    def test_step(self, batch, batch_idx):
        labels = batch["labels"]
        batch.pop("labels", None)
        outputs = self.forward(**batch)
        loss = outputs.loss
        reshaped_logits = outputs.logits.view(-1, NUM_CHOICES)
        pred = torch.argmax(reshaped_logits, dim=1)
        print(pred)
        print(pred == labels)
        acc = acc_from_logits_and_labels(reshaped_logits, labels, self.accuracy)

        return {
            "loss": loss,
            "acc": acc
        }

    def test_epoch_end(self, outputs):
        acc_li = [output["acc"] for output in outputs]
        self.log_each_step("test acc", sum(acc_li) / len(acc_li), on_step=False)


def get_trainer(logger, epochs=3, gpus=1):
    trainer = pl.Trainer(
        default_root_dir="./pl_logs",
        logger=logger,
        log_every_n_steps=1,
        max_epochs=epochs,
        gpus=gpus,
        # auto_scale_batch_size="power",
        # progress_bar_refresh_rate=20,
        callbacks=[
            ray_tune_callback,
            early_stop_callback,
            checkpoint_callback,
        ])

    return trainer


def train_tune(config, logger, epochs=3, gpus=1):
    trainer = get_trainer(logger, epochs, gpus)
    model = Model(config)
    trainer.fit(model)


if __name__ == "__main__":
    early_stop_callback = EarlyStopping(
        monitor="val_epoch_loss",
        min_delta=0.0,
        patience=3,
        verbose=True,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_step_loss',
        dirpath=config["ckpt_dir"],
        filename='{epoch:02d}-{val_step_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    if config["use_gpu"]:
        ray.init(num_gpus=1)

    rt_config = {
        # "lr": tune.loguniform(1e-6, 1e-4),

        # the following values are the results of fine-tuning
        "lr": tune.choice([1.57513e-05]),
        "batch_size": tune.choice([32]),
        "max_seq_length": tune.choice([48]),
        "hidden_dropout_prob": tune.choice([.1]),
        "hidden_act": tune.choice(["gelu"]),
        "architecture": {
            "tokenizer": BertTokenizer,
            "model": BertForMultipleChoice,
            "pretrained_model_name": "bert-base-uncased"
        }
    }

    ray_tune_callback = TuneReportCallback(
        {
            "loss": "val_epoch_loss",
            "acc": "val_epoch_acc"
        },
        on="validation_end")

    logger = TensorBoardLogger('tb_logs/', name='csqa')
    # tr = train_tune(config, logger, epochs=1, gpus=1)

    analysis = tune.run(
        partial(
            train_tune, logger=logger, epochs=3, gpus=config["use_gpu"],
        ),
        config=rt_config,
        num_samples=1,
        resources_per_trial={
            "cpu": 10,
            "gpu": config["use_gpu"],
        },
        metric="acc",
        mode="max",
        log_to_file=True,
        local_dir="./ray_results",
        name="csqa",
        fail_fast=True
    )

    print("Best config: ", analysis.best_config)

    # testing with the latest ckpt file
    glob_pattern = os.path.join(checkpoint_callback.dirpath, '*')
    ckpt_files = sorted(glob(glob_pattern), key=os.path.getctime)
    model = Model.load_from_checkpoint(ckpt_files[-1], model_config=analysis.best_config)
    trainer = get_trainer(logger=logger, epochs=3, gpus=config["use_gpu"])
    test_res = trainer.test(model)
