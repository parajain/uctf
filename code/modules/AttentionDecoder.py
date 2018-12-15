import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        self.softmax = nn.Softmax()

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score
        return self.softmax(attn_energies).unsqueeze(1) # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = self.attn(torch.cat([hidden, encoder_outputs], 2)) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]

class ControllableAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, embed_size, embedding , context_dim, n_layers=1, dropout_p=0, teacher_forcing=True):
        super(ControllableAttnDecoderRNN, self).__init__()
        print('Initializing batch attention ')
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.context_dim = context_dim
        self.dropout_p = dropout_p
        #self.embedding = nn.Embedding(output_size, embed_size)
        self.embedding = embedding
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', hidden_size)
        #self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers)
        self.transform = nn.Linear(hidden_size + embed_size + context_dim, embed_size)
        #self.attn_combine = nn.Linear(hidden_size + embed_size, hidden_size)
        if teacher_forcing:
            self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout_p)
        else:
            self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
        #self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, word_embedded, last_hidden, encoder_outputs, control_input, opts, pretrain=False, teacher_forcing=True):
        '''
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop
            to process the whole sequence
        '''
        # Get the embedding of the current input word (last output word)
        if pretrain or teacher_forcing:
            batch_size = word_embedded.size()[0]
            word_embedded = self.embedding(word_embedded).view(1, batch_size, -1)
        else:
            batch_size = word_embedded.size()[1]
            word_embedded = word_embedded.unsqueeze(0)


        #word_embedded = self.embedding(word_input).view(1, word_input.data.shape[0], -1) # (1,B,N)
        word_embedded = self.dropout(word_embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        #rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
        control_input = control_input.unsqueeze(0)
        rnn_input = torch.cat((rnn_input, control_input), 2)
        # doing this transformation so make the size consistent, due to concat size doubles + control_input size
        rnn_input = F.tanh(self.transform(rnn_input))
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N)->(B,N)
        #output = F.log_softmax(self.out(torch.cat((output, context), 1)))
        # Return final output, hidden state
        output_projected = self.out(output)
        return output, output_projected, hidden, attn_weights