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
