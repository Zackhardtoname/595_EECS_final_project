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
