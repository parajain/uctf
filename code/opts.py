import argparse

def model_opts(parser):
    parser.add_argument('-rnn_size', type=int, default=500,
                        help='Size of LSTM hidden states')
    parser.add_argument('-emb_size', type=int, default=200,
                        help='Size of embedding layer for pretrain this should match with pretrained embeddings')
    parser.add_argument('-rnn_type', type=str, default='LSTM',
                        choices=['LSTM', 'GRU'],
                        help="""The gate type to use in the RNNs""")
    parser.add_argument('-layers', type=int, default=1,
                        help='Number of layers in enc/dec.')
    parser.add_argument('-dropout', type=float, default=0,
                        help="Dropout probability; applied in LSTM stacks.")
    parser.add_argument('-temp', type=float, default=0.1,
                        help="SM temperature")
    parser.add_argument('-weight_mlp', type=float, default=0.3,
                        help="weight_mlp")
    parser.add_argument('-weight_ce', type=float, default=0.7,
                        help="weight_ce")
    #parser.add_argument('-bidirectional', action="store_false", default=True)


def train_opts(parser):
    parser.add_argument('-save_model',  required=True, help='model file base name')
    parser.add_argument('-vocab_file',  required=True, help='vocab file containing word to id mapping')
    parser.add_argument('-embeddings', default='', help='path of glove embedding file')
    parser.add_argument('-train_embedding', action="store_true", default=False,
                        help='Train embeddings.')
    parser.add_argument('-num_epoch_pretrain', type=int, default=100,
                        help='Number of epochs to pretrain')
    parser.add_argument('-save_pre_trained_filename', default='pretrained', type=str,
                        help="""File to save pretrained encoder and decoder""")
    parser.add_argument('-pre_trained_model', default='pretrained', type=str,
                        help="""File to load pretrained encoder and decoder""")
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')
    parser.add_argument('-resume', default='', type=str,
                        help="""If training from a checkpoint then this is the
                        path to the trained model's state_dict.""")
    parser.add_argument('-decoder', default='rnn', type=str,
                        help="""Type of decoder rnn or attn-rnn""")
    parser.add_argument('-data_reader', default='nmt', type=str,
                        help="""Type of data reader nmt or graph""")
    parser.add_argument('-save_every', type=int, default=10,
                        help='Save model every')
    parser.add_argument('-generate_every', type=int, default=10,
                        help='generate valid senetences model every these many epochs')


    # Optimization options
    parser.add_argument('-batch_size', type=int, default=64,
                        help='Maximum batch size')
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Starting learning rate. If adagrad/adadelta/adam
                            is used, then this is the global learning rate.
                            Recommended settings: sgd = 1, adagrad = 0.1,
                            adadelta = 1, adam = 0.001""")
    parser.add_argument('-epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('-pretrain_decoder', action="store_true", default=False,
                        help='pretrain encoder.')
    parser.add_argument('-load_pretrained', action="store_true", default=False,
                        help='load pretrain .')
    parser.add_argument('-pretrain', action="store_true", default=False,
                        help='pretrain encoder.')
    parser.add_argument('-add_grad_noise', action="store_true", default=False,
                        help='Add some noise to the gradients')
    # GPU
    parser.add_argument('-use_cuda', action="store_true", default=False,
                        help='Train on gpu.')
    parser.add_argument('-seed', type=int, default=-1,
                        help="""Random seed used for the experiments
                            reproducibility.""")

    parser.add_argument('-debug', action="store_true", default=False,
                        help='debug prints.')



def data_opts(parser):
    #data path
    parser.add_argument('-pretrain_src', required=False,
                        help='Path to train source file')
    parser.add_argument('-train_src', required=True,
                        help='Path to train source file')
    parser.add_argument('-train_tgt', required=False,
                        help='Path to train target file')
    parser.add_argument('-valid_src', required=False,
                        help='Path to valid source file')
    parser.add_argument('-valid_tgt', required=False,
                        help='Path to valid target file')

    # Dictionary Options
    parser.add_argument('-src_vocab_size', type=int, default=100000,
                        help="Size of the source vocabulary")
    parser.add_argument('-tgt_vocab_size', type=int, default=100000,
                        help="Size of the target vocabulary")

    parser.add_argument('-src_words_min_frequency', type=int, default=0)
    parser.add_argument('-tgt_words_min_frequency', type=int, default=0)

    # Truncation options
    parser.add_argument('-src_seq_length', type=int, default=50,
                        help="Maximum source sequence length")
    parser.add_argument('-tgt_seq_length', type=int, default=50,
                        help="Maximum target sequence length to keep.")
    parser.add_argument('-merge_vocab', action="store_true",default=False,
                        help='Megre source and target vocab')

    # Data processing options
    parser.add_argument('-shuffle', action="store_false", default=True,
                        help='Shuffle dataset.')

def score_opts(parser):
    parser.add_argument('-fw_w', type=float, default=1, help='readability score weight')
    parser.add_argument('-lm_w', type=float, default=1, help='language model score weight')
    parser.add_argument('-len_w', type=float, default=1, help='length penalty weight')
    parser.add_argument('-c_w', type=float, default=1, help='class score weight')
    parser.add_argument('-dc_w', type=float, default=1, help=' document similarity weight')
    parser.add_argument('-baseline', type=float, default=0, help='Initial baseline for reinforce reward')
    parser.add_argument('-lm_snapshot',  required=False, help='language model snapshot directory')
    parser.add_argument('-use_vector', action="store_true", default=False, help='use vector based sampling and loss calculation')
    parser.add_argument('-readaility',  required=False, help='Which readability implementation to use')





