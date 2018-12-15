### **optional arguments**:
* **-rnn_size [500]** 
Size of LSTM hidden states

* **-emb_size [200]** 
Size of embedding layer for pretrain this should match with pretrained
embeddings

* **-rnn_type [LSTM]** 
The gate type to use in the RNNs

* **-layers [1]** 
Number of layers in enc/dec.

* **-dropout []** 
Dropout probability; applied in LSTM stacks.

* **-temp [0.1]** 
SM temperature

* **-weight_mlp [0.3]** 
weight_mlp

* **-weight_ce [0.7]** 
weight_ce

* **-save_model []** 
model file base name

* **-vocab_file []** 
vocab file containing word to id mapping

* **-embeddings []** 
path of glove embedding file

* **-train_embedding []** 
Train embeddings.

* **-num_epoch_pretrain [100]** 
Number of epochs to pretrain

* **-save_pre_trained_filename [pretrained]** 
File to save pretrained encoder and decoder

* **-pre_trained_model [pretrained]** 
File to load pretrained encoder and decoder

* **-start_epoch [1]** 
The epoch from which to start

* **-resume []** 
If training from a checkpoint then this is the path to the trained model's
state_dict.

* **-decoder [rnn]** 
Type of decoder rnn or attn-rnn

* **-data_reader [nmt]** 
Type of data reader nmt or graph

* **-save_every [10]** 
Save model every

* **-generate_every [10]** 
generate valid senetences model every these many epochs

* **-batch_size [64]** 
Maximum batch size

* **-learning_rate [0.001]** 
Starting learning rate. If adagrad/adadelta/adam is used, then this is the
global learning rate. Recommended settings: sgd = 1, adagrad = 0.1, adadelta =
1, adam = 0.001

* **-epochs [100]** 
Number of training epochs

* **-pretrain_decoder []** 
pretrain encoder.

* **-load_pretrained []** 
load pretrain .

* **-pretrain []** 
pretrain encoder.

* **-add_grad_noise []** 
Add some noise to the gradients

* **-use_cuda []** 
Train on gpu.

* **-seed [-1]** 
Random seed used for the experiments reproducibility.

* **-debug []** 
debug prints.

* **-pretrain_src []** 
Path to train source file

* **-train_src []** 
Path to train source file

* **-train_tgt []** 
Path to train target file

* **-valid_src []** 
Path to valid source file

* **-valid_tgt []** 
Path to valid target file

* **-src_vocab_size [100000]** 
Size of the source vocabulary

* **-tgt_vocab_size [100000]** 
Size of the target vocabulary

* **-src_words_min_frequency []** 

* **-tgt_words_min_frequency []** 

* **-src_seq_length [50]** 
Maximum source sequence length

* **-tgt_seq_length [50]** 
Maximum target sequence length to keep.

* **-merge_vocab []** 
Megre source and target vocab

* **-shuffle [True]** 
Shuffle dataset.

* **-fw_w [1]** 
readability score weight

* **-lm_w [1]** 
language model score weight

* **-len_w [1]** 
length penalty weight

* **-c_w [1]** 
class score weight

* **-dc_w [1]** 
document similarity weight

* **-baseline []** 
Initial baseline for reinforce reward

* **-lm_snapshot []** 
language model snapshot directory

* **-use_vector []** 
use vector based sampling and loss calculation

* **-readaility**
Readability type to use

Dependencies: Spacy, NLTK, KenLM, Pytorch, Readability, gensim, Tensorflow, marian-nmt
Readability : https://pypi.org/project/readability/ Version 0.2
Spacy: https://spacy.io/ Version: 2.0.10
Pytorch: 0.4
gensim: 3.4


### Dataset:
* For training language model and vocab creation
** http://www.statmt.org/europarl/
** http://bailando.sims.berkeley.edu/enron email.html
** http://ota.ox.ac.uk/desc/2077
** http://csmining.org/index.php/spam-assassin-datasets.html
** http://www.mykidsway.com/essays/

-------------------------------

### Sample CPU command

python train.py -rnn_size 250 -train_src data/formal_informal_v1/sentences100.txt -train_tgt data/formal_informal_v1/controls100.txt -valid_src data/formal_informal_v1/sentences100.txt -valid_tgt data/formal_informal_v1/controls100.txt -src_words_min_frequency 1 -tgt_words_min_frequency 1 -batch_size 64 -save_every 3 -epochs 2000 -learning_rate 0.01 -src_seq_length 10 -tgt_seq_length 10 -decoder attn-rnn -data_reader nmt  -pretrain -num_epoch_pretrain 1 -pre_trained_encoder_file preTrainedEncoder  -vocab_file data/vocab_mc5.txt  -lm_snapshot ../snapshot/ -pretrain_decoder -fw_w 1 -lm_w 1 -len_w 1  -save_model attnptbs64fw1lm1len1 -embeddings glove100.dat -emb_size 100 -generate_every 1

#### Train Neural Language model
* In word language model directory
* mkdir snapshot
* python train_nlm.py --data ../../data_for_lm/ --cuda
* mv model.pt snapshop
* mv snapshot snapshot_gpu
* mv snapshot_gpu ../../
* move the folder outside(contains model.pt corpus.p and arg.p file), this will be accessed during training


#### Kenlm Language model
Code: https://github.com/kpu/kenlm
Dataset: Europarl monolingual English

### Spacy
* https://spacy.io/usage/



### pretraining example:

python train.py -emb_size 300 -rnn_size 250 -pretrain_src ../dataset/large.txt -train_src ../dataset/sentences100.txt -valid_src data/sentences100.txt -src_words_min_frequency 1 -batch_size 32 -save_every 1 -epochs 20 -learning_rate 0.001 -src_seq_length 17 -decoder attn-rnn -data_reader nmt -layers 2 -pretrain -num_epoch_pretrain 1 -save_pre_trained_filename pretrained_model -src_words_min_frequency 0 -vocab_file data/vocab_mc5.txt  -pretrain_decoder -fw_w 10 -len_w 0.2 -dc_w 5  -lm_w 5  -generate_every 1 -debug -save_model model -baseline 0.7


### pretraining with glove embeddings

python train.py -rnn_size 250 -pretrain_src ../dataset/large.txt -train_src ../dataset/sentences100.txt -valid_src ../dataset/sentences100.txt -src_words_min_frequency 1 -batch_size 32 -save_every 1 -epochs 20 -learning_rate 0.001 -src_seq_length 17 -decoder attn-rnn -data_reader nmt -layers 2 -pretrain -num_epoch_pretrain 1 -save_pre_trained_filename pretrained_model -src_words_min_frequency 0 -vocab_file data/vocab_mc5.txt  -pretrain_decoder -fw_w 10 -len_w 0.2 -dc_w 5 -lm_w 5  -generate_every 1 -debug -save_model model -baseline 0.7 -embeddings ../controllable_gen/glove100.dat -emb_size 100

### load pretrained

python train.py -emb_size 300 -rnn_size 250 -train_src data/formal_informal_v1/sentences100.txt -valid_src data/formal_informal_v1/sentences100.txt -src_words_min_frequency 1 -batch_size 32 -save_every 1 -epochs 20 -learning_rate 0.001 -src_seq_length 17 -decoder attn-rnn -data_reader nmt -layers 2 -src_words_min_frequency 0 -vocab_file data/vocab_mc5.txt -fw_w 10 -len_w 0.2 -dc_w 5  -lm_w 5  -generate_every 1 -debug -save_model model -baseline 0.7 -load_pretrained -pre_trained_model pretrained_model


### Train
python train.py -emb_size 300 -rnn_size 250  -train_src ../dataset/train.input -valid_src ../dataset/valid.input -src_words_min_frequency 1 -batch_size 64 -save_every 1 -epochs 100 -learning_rate 0.001 -src_seq_length 17 -decoder rnn -data_reader nmt -layers 2 -src_words_min_frequency 0  -vocab_file ../dataset/vocab_more.txt -fw_w 15 -len_w 0.2 -dc_w 5 -c_w 0.1 -lm_w 5  -generate_every 1 -save_model ../model -baseline 0.7 -num_epoch_pretrain 50 -load_pretrained -pre_trained_model pretrained_model-debug -weight_mlp 0.1 -weight_ce 0.9 -temp 0.01

#### Train without Attention
python train.py -rnn_size 250 -pretrain_src data/formal_informal_v1/sentences100.txt -train_src data/formal_informal_v1/sentences100.txt -valid_src data/formal_informal_v1/sentences100.txt -src_words_min_frequency 1 -batch_size 32 -save_every 1 -epochs 20 -learning_rate 0.001 -src_seq_length 17 -decoder rnn -data_reader nmt -layers 2 -pretrain -num_epoch_pretrain 1 -save_pre_trained_filename pretrained_model -src_words_min_frequency 0 -vocab_file data/vocab_mc5.txt  -pretrain_decoder -fw_w 10 -len_w 0.2 -dc_w 5 -lm_w 5  -generate_every 1 -debug -save_model model -baseline 0.7 -embeddings ../controllable_gen/glove100.dat -emb_size 100


### Simple NMT Experiment for simple to complex task
https://marian-nmt.github.io/