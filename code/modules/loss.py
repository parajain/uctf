import torch
from torch.nn import functional
from torch.autograd import Variable


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand



def BCELossLogitsUsingVector(logits, vector, target_lengths, opts):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        vector: (batch_size,  max_length , num_classes)
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: Binary cross entropy loss
    """
    target_lengths = Variable(torch.LongTensor(target_lengths))
    if opts.use_cuda:
        target_lengths = target_lengths.cuda()

    batch_size, max_len, num_classes = logits.size()
    vector = vector.view(-1, logits.size(-1)) # (bs * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1)) # (bs * max_len, num_classes)

    weights = sequence_mask(sequence_length=target_lengths, max_len=max_len).float()
    weights = weights.view(-1, 1).repeat(1, num_classes)
    logits_flat = functional.softmax(logits_flat)
    y_ = Variable(vector.float())
    loss_f = torch.nn.BCELoss(weights.data)
    loss = loss_f(logits_flat, y_)
    return loss


def BCELossLogitsUsingTargets(logits, target, target_lengths, opts):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: Binary cross entropy loss
    """
    target_lengths = Variable(torch.LongTensor(target_lengths))
    if opts.use_cuda:
        target_lengths = target_lengths.cuda()

    logits_flat = logits.view(-1, logits.size(-1)) # (bs * max_len, num_classes)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)

    weights = sequence_mask(sequence_length=target_lengths, max_len=target.size(1)).float()
    batch_size, max_len, num_classes = logits.size()
    y_onehot = torch.FloatTensor(batch_size * max_len, num_classes)
    if opts.use_cuda:
        y_onehot = y_onehot.cuda()
    weights = weights.view(-1, 1).repeat(1, num_classes)
    y_onehot.zero_()
    y_onehot.scatter_(1, target_flat.data, 1)
    #print(target_flat)
    #print(y_onehot)
    #sigmoid = torch.nn.Sigmoid()
    #ogits_flat = sigmoid(logits_flat)
    logits_flat = functional.softmax(logits_flat)
    #print(logits_flat)
    y_onehot = Variable(y_onehot)
    #print(weights)
    loss_f = torch.nn.BCELoss(weights.data)
    loss = loss_f(logits_flat, y_onehot)
    return loss





def masked_cross_entropy(logits, target, length, opts):
    length = Variable(torch.LongTensor(length))
    if opts.use_cuda:
        length = length.cuda()

    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """
    #print(' cross entropy ')
    #print(logits.size())
    batch_max_len = logits.size()[1]
    #print('batch max len', batch_max_len)
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    logits_flat = logits_flat + 0.01
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat)
    #in logits max length is max length of a batch to optimize computation during decoding
    # and not overall max sequence length
    #However, targets is padded till overall max sequence length so chop that too first
    target = target[:, :batch_max_len].contiguous()
    #print('target size ', target.size())

    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss