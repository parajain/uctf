from modules.ControllableModel import *
import torch
import torch.nn as nn
from torch.autograd import Variable
from modules.ScoreModelDataset import *
import torch.utils.data as data_utils
from modules.Vocab import *

def _cat_directions(h, bidirectional_encoder):
    if bidirectional_encoder:
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
    return h


class ScoreSentenceEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, emb_size, embedding, n_layers=1):
        super(ScoreSentenceEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = embedding
        self.gru = nn.GRU(emb_size, hidden_size, n_layers, bidirectional=False)
        # if bidirectional:
        #    self.bidiout = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input_seqs, hidden, flag):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        if flag:
            embedded = self.embedding(input_seqs)
        else:
            embedded = torch.matmul(input_seqs, self.embedding.weight)
        # print(embedded)
        # print('input_lengths', input_lengths)
        #packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        #outputs, hidden = self.gru(packed, hidden)
        outputs, hidden = self.gru(embedded, hidden)
        #outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        # outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        # if self.bidirectional:
        #    outputs = F.tanh(self.bidiout(outputs))
        return outputs, hidden


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class ScoreModel(nn.Module):
    def __init__(self, input_size, rnn_hidden_size, net_hidden_size, emb_size, embedding, num_of_controls, n_layers=1):
        super(ScoreModel, self).__init__()
        self.encoder = ScoreSentenceEncoder(input_size, rnn_hidden_size, emb_size, embedding, n_layers)
        self.net = Net(rnn_hidden_size * 2, net_hidden_size, num_of_controls)

    def forward(self, input_seqs, generated_seqs, flag, hidden=None):
        outputs, hidden = self.encoder(input_seqs, hidden, True)
        ########################### TO TEST MADE TRUE
        goutputs, ghidden = self.encoder(generated_seqs, hidden, flag)
        # print('ghidden ', ghidden)
        combined_hidden = torch.cat((hidden, ghidden), 2)
        combined_hidden = combined_hidden.squeeze(0)
        # print(combined_hidden)
        out = self.net(combined_hidden)
        # print(out)
        return out

def init_score_model(vocab_file):
    learning_rate = 0.001
    emb_size = 100
    rnn_size = 200
    net_hidden_size = 100
    num_layers = 1
    num_of_controls = 4

    vocab_src = Vocab('model_vocab')
    vocab_src.load_vocab(vocab_file)
    vocab_tgt = vocab_src

    embedding = nn.Embedding(vocab_src.get_n_words, emb_size)
    model = ScoreModel(vocab_src.get_n_words, rnn_size, net_hidden_size, emb_size, embedding, num_of_controls,
                       num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, criterion, optimizer

def train_score_model(sentences, gsentences, controls, vocab_src, vocab_tgt, model, criterion, optimizer, options):
    num_epochs = options.num_epoch_pretrain
    #vocab_file = '../data/vocab_more.txt'

    #sentences = open('score_model_test_data/sampled.txt', 'r').readlines()
    #controls = open('score_model_test_data/control.txt', 'r').readlines()
    #controls = [[float(c.rstrip())] for c in controls]

    dataset = ScoreModelDataset(sentences, gsentences, controls, 'test',
                                                     vocab_src, vocab_tgt, 15, 15)
    dataloader = data_utils.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)



    for epoch in range(num_epochs):
        batch_num = 0
        epoch_loss = 0
        for sample_batched in dataloader:
            # print(sample_batched)
            batch_num = batch_num + 1
            input_batches = Variable(sample_batched['src']).transpose(0, 1)  # will give seq_len x batch_size
            input_lengths = sample_batched['src_len']

            output_batches = Variable(sample_batched['tgt']).transpose(0, 1)  # will give seq_len x batch_size
            output_lengths = sample_batched['tgt_len']
            control_batches = Variable(sample_batched['control_tensor'])  # this is  bs x control_len

            input_lengths, perm_idx = input_lengths.sort(0, descending=True)
            input_batches = input_batches[:, perm_idx]
            output_lengths = output_lengths[perm_idx]
            output_batches = output_batches[:, perm_idx]
            input_lengths = [x for x in input_lengths]
            output_lengths = [x for x in output_lengths]
            control_batches = control_batches[perm_idx, :]

            if options.use_cuda:
                input_batches = input_batches.cuda()
                output_batches = output_batches.cuda()
                control_batches = control_batches.cuda()

            model_outputs = model(input_batches, output_batches, True)
            control_batches = control_batches.long().squeeze(1)
            # print(control_batches)
            # print(model_outputs)

            loss = criterion(model_outputs, control_batches - 1)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data[0]
        print ('Epoch %d, Loss: %.4f' % (epoch + 1, epoch_loss))


def main():
    train()


if __name__ == '__main__':
    main()








