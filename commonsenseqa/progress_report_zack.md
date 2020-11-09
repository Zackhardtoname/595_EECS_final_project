I am mainly working the commonsenseqa paper. My overall goals are reimplementing (in PyTorch versus the Tensorflow 1 based code the authors used) the Bert model the authored reported the best results on. Then if I have more time, I would try to add some improvements (such as trying out GPT-2; the authors also mention the possibility of adding gradient accumulation and gradient checkpointing).

I have done the following tasks.

1. I first read and annotated carefully the following papers and learned a great deal about the inner details of transformers, GPT1, Bert, and GPT2.

   COMMONSENSEQA
   illustrated-gpt2
   BERT: Pre-training of Deep Bidirectional Transformers for?Language Understanding?
   Attention Is All You Need
   The Illustrated Transformer
   Language Models are Few-Shot Learners
   Recent Advances in Natural Language Inference
   Using the Output Embedding to Improve Language Models
   illustrated-bert

2. I attended office hours to ask my remaining conceptual questions on the aforementioned models and communicated my plan for the final project with the TAs during office hours.

3. I have tried to evaluate the validation set based on the authors' code on BERT Large for the best performance. However, the default parameters straight from the paper are too large (max sequence length 128, batch size 16, adam optimizer) for me to train either on my local machine or on Colab. I later found on the authors' repo that all "experiments in the paper were fine-tuned on a Cloud TPU, which has 64GB of device RAM." Thus, I had to drastically reduce parameters (max_seq_length=32, train_batch_size=4) just to get the model to run. 

4. After some experimentations and hyper-parameter tuning (the authors did not perform), I arrived at the validation accuracy of 61.83456%. It is quite a bit higher than the 55.9% in the paper.

5. Our team also met and communicated each others' progress and plan for the remaining deliverables.







