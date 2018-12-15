import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.autograd import Variable
import sys

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, emb_size ,embedding, n_layers=1, dropout=0, bidirectional=False):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = embedding
        self.bidirectional = bidirectional
        self.gru = nn.GRU(emb_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        #if bidirectional:
        #    self.bidiout = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input_seqs, input_lengths, hidden, opts):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        #outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        #if self.bidirectional:
        #    outputs = F.tanh(self.bidiout(outputs))
        return outputs, hidden

class ControllableDecoder(nn.Module):
    def __init__(self, hidden_size, output_size,emb_size,  embedding, context_dim, n_layers=1,dropout_p = 0, teacher_forcing=True):
        super(ControllableDecoder, self).__init__()
        print('Init ControllableDecoder length of control vector ', context_dim)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.embedding = embedding
        self.context_dim = context_dim
        if teacher_forcing:
            self.gru = nn.GRU(emb_size, hidden_size, n_layers, dropout=dropout_p)
        else:
            self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        #self.gru = gru
        self.transform = nn.Linear(emb_size + context_dim, emb_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, emb_input, hidden, encoder_outputs,control_input, opts, pretrain=False, teacher_forcing=True):
        """
        :param emb_input: 1 x bs x hidden_size if not pretrain otherwise (batch_size,)
        :param hidden:
        :param encoder_outputs:
        :param control_input:
        :param opts:
        :return:
        """
        #print('emb_input', emb_input.size())
        #print('control_input', control_input.size())
        if pretrain or teacher_forcing:
            batch_size = emb_input.size()[0]
            emb_input = self.embedding(emb_input)
            emb_input = emb_input.view(1, batch_size, -1)
            #print('emb_input', emb_input.size())
        else:
            batch_size = emb_input.size()[1]
        #emb_input = emb_input.view(1, batch_size, -1)
        #control_input = control_input.view(1, batch_size, -1)
        print('control input', control_input)
        control_input = control_input.unsqueeze(0)
        rnn_input = torch.cat((emb_input, control_input), 2)
        rnn_input = F.tanh(self.transform(rnn_input))
        output, hidden = self.gru(rnn_input, hidden)
        output_projected = self.out(output[0])
        return output, output_projected, hidden, None

    def initInput(self, opts, batch_size, vocab):
        go = vocab.GO
        result = Variable(self.embedding(go))
        if opts.use_cuda:
            return result.cuda()
        else:
            return result

    def fakeControl(self, opts, batch_size):
        result = Variable(torch.zeros(batch_size, self.context_dim))
        if opts.use_cuda:
            return result.cuda()
        else:
            return result

