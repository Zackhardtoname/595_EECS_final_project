# for ele in c:
#     if len(ele["text"]) != 5:
#         print(ele["text"])

# lens = [len(e) for e in rep_q]
# max_len = max(lens)
# max_len_idx = lens.index(max_len)
# print(max_len, max_len_idx)
# to_decode = tokenizer(rep_q[max_len_idx], rep_q[max_len_idx], return_tensors='pt', padding=True, truncation=True, max_length=32)["input_ids"][0]
# print(tokenizer.decode(to_decode))
import torch
import pytorch_lightning as pl

a = torch.tensor([1, 2, 3])
a = a.cuda()
b = a.clone()
b = b.cuda()
accuracy = pl.metrics.Accuracy()
print(accuracy(a, b))

# logger = TensorBoardLogger('lightning_logs', name='my_model')

# trainer = pl.Trainer(
#     logger=logger,
# log_every_n_steps=5,  # each batch is a step
# gpus=1,
# max_epochs=config["max_epochs"]
# )
# model = Model()

# x_test, y_test = preprocess(dataset["test"])
# trainer.test(test_dataloaders=DataLoader(DictDataset(x_test, y_test), batch_size=config["test_batch_size"]))
# analysis.get_best_trial().last_result["acc"]
# analysis.get_best_trial().config["batch_size"]
# print(f"test_acc", acc)
# print(f"test_loss", loss)

# trainer.fit(
#     model,
#     DataLoader(DictDataset(x_train, y_train), batch_size=config["train_batch_size"], num_workers=8),
#     DataLoader(DictDataset(x_val, y_val), batch_size=config["val_batch_size"], num_workers=8),
# )

# pred = torch.argmax(logits, dim=1)
# labels.to(pred.device)
# # acc = CrossEntropyLoss(logits, labels)
# acc = labels.eq(pred).sum() / labels.shape[0]

# with open(ckpt_path) as f:
#     model_state, optimizer_state = torch.load(f)
#
# model.load_state_dict(model_state)
# optimizer.load_state_dict(optimizer_state)
# ouputs = model(model.x_test)
# print(test_res)
