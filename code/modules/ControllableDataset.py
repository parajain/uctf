# import  torch.utils.data.Dataset
from modules.Vocab import *
import torch.utils.data as data_utils
import torch



class ControllableUnsupervisedDatasetFromArray():
    def __init__(self, src_lines, controls, control_len, name, vocab_src, src_msl):
        self.src_lines = src_lines
        if controls is None:
            print('Creating default control vector')
            self.controls = []
            for i in range(len(src_lines)):
                x = [1] * control_len
                self.controls.append(x)
        else:
            self.controls = controls
        self.vocab_src = vocab_src
        self.src_msl = src_msl
        assert len(self.src_lines) == len(self.controls), 'Number of source and target instance should match: ' + str(
            len(self.src_lines)) + ' ' + str(len(self.controls))

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_line = self.src_lines[idx]
        src_sentence = src_line.rstrip()

        control_list = self.controls[idx]
        control_tensor = self.get_control(control_list)

        src_int, src_len = self.vocab_src.sentence_to_word_ids(src_sentence, max_sequence_length=self.src_msl,
                                                               prependGO=False, eos=True)
        src_tensor = torch.LongTensor(src_int)

        example = {'src': src_tensor, 'src_len': src_len, 'control_tensor': control_tensor}
        return example

    def get_control(self, s):
        #slist = s.split(' ')
        control_list = [float(x) for x in s]
        control_list = np.asarray(control_list, dtype='float32')
        control_tensor = torch.from_numpy(control_list)
        return control_tensor


class ControllableSupervisedDatasetFromArray():
    """Dataset class which defines basic Controllable dataset"""

    def __init__(self, src_lines, tgt_lines, controls, name, vocab_src, vocab_tgt, src_msl, tgt_msl):
        """
        Each line of input file is of format:  sentence TAB <list of control numbers separated by SPACE >
        :param src_data_file: source data file, one line per instance
        :param tgt_data_file: target data file, one line per instance
        :param src_msl: max sequence length
        :param tgt_msl: max sequence length
        """
        #sf = open(src_data_file, 'r')
        #tf = open(tgt_data_file, 'r')
        #self.src_lines = sf.readlines()
        #self.tgt_lines = tf.readlines()
        self.src_lines = src_lines
        self.tgt_lines = tgt_lines
        self.controls = controls

        self.num_instances = len(self.src_lines)
        assert len(self.src_lines) == len(self.tgt_lines), 'Number of source and target instance should match: ' + str(
            len(self.src_lines)) + ' ' + str(len(self.tgt_lines))
        self.vocab_tgt = vocab_tgt
        self.vocab_src = vocab_src
        self.src_msl = src_msl
        self.tgt_msl = tgt_msl

    def __len__(self):
        return len(self.tgt_lines)

    def __getitem__(self, idx):
        src_line = self.src_lines[idx].rstrip()
        tgt_line = self.tgt_lines[idx].rstrip()
        #src_sentence = src_line.split('    ')[0].strip('\r\n')
        #control_list = src_line.split('    ')[1].strip('\r\n')
        src_sentence = src_line
        tgt_sentence = tgt_line

        control_list = self.controls[idx]
        control_tensor = self.get_control(control_list)

        src_int, src_len = self.vocab_src.sentence_to_word_ids(src_sentence, max_sequence_length=self.src_msl,
                                                               prependGO=False, eos=True)
        src_tensor = torch.LongTensor(src_int)

        tgt_int, tgt_len = self.vocab_src.sentence_to_word_ids(tgt_sentence, max_sequence_length=self.src_msl,
                                                               prependGO=False, eos=True)
        tgt_tensor = torch.LongTensor(tgt_int)

        example = {'src': src_tensor, 'src_len': src_len, 'tgt': tgt_tensor,'tgt_len': tgt_len,
                   'control_tensor': control_tensor}
        return example

    def get_control(self, s):
        #slist = s.split(' ')
        control_list = [float(x) for x in s]
        control_list = np.asarray(control_list, dtype='float32')
        control_tensor = torch.from_numpy(control_list)
        return control_tensor

    def get_num_instances(self):
        return self.num_instances

class SupervizedLoader():
    """Dataset class which defines basic NMT dataset"""
    def __init__(self, src_data_list, tgt_data_list, name, vocab_src, vocab_tgt, src_msl, tgt_msl):
        """
        :param src_data_list: source data list
        :param tgt_data_list: target data list
        :param src_msl: max sequence length
        :param tgt_msl: max sequence length
        """
        
        self.src_lines = src_data_list
        self.tgt_lines = tgt_data_list
        assert  len(self.src_lines) == len(self.tgt_lines), 'Number of source and target instance should match: ' + str(len(self.src_lines)) + ' ' + str(len(self.tgt_lines))
        self.vocab_tgt = vocab_tgt
        self.vocab_src = vocab_src
        self.src_msl = src_msl
        self.tgt_msl = tgt_msl
        self.num_instances = len(self.src_lines)
    def __len__(self):
        return len(self.tgt_lines)
    def get_num_instances(self):
        return self.num_instances

    def __getitem__(self, idx):
        #print('get item', idx)
        src = self.src_lines[idx].rstrip()
        tgt = self.tgt_lines[idx].rstrip()

        src_int, src_len = self.vocab_src.sentence_to_word_ids(src, max_sequence_length = self.src_msl, prependGO = False, eos = True)

        #Removing append <GO> from here, as during creating training batch it is done.
        tgt_int, tgt_len = self.vocab_tgt.sentence_to_word_ids(tgt, max_sequence_length = self.tgt_msl, prependGO = False, eos = True)

        #print('src_int len', len(src_int))
        #print('tgt_int len', len(tgt_int))
        #print('tgt_len ', tgt_len)
        #print('src_len ', src_len)

        src_tensor = torch.LongTensor(src_int)
        tgt_tensor = torch.LongTensor(tgt_int)


        #print('src_tensor ', src_tensor.size())
        #print('tgt_tensor ', tgt_tensor.size())
        example = {'src' : src_tensor, 'tgt' : tgt_tensor, 'src_len' : src_len, 'tgt_len' : tgt_len}
        return example

class UnsupervisedDataset():

    def __init__(self, src_data_file, name, vocab_src, src_msl):
        """
        Each line of input file is of format:  sentence TAB <list of control numbers separated by SPACE >
        :param src_data_file: source data file, one line per instance
        :param src_msl: max sequence length
        """
        sf = open(src_data_file, 'r')
        self.src_lines = sf.readlines()
        self.num_instances = len(self.src_lines)
        self.vocab_src = vocab_src
        self.src_msl = src_msl

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_line = self.src_lines[idx].rstrip()
        src_sentence = src_line
        src_int, src_len = self.vocab_src.sentence_to_word_ids(src_sentence, max_sequence_length=self.src_msl,
                                                               prependGO=False, eos=True)
        src_tensor = torch.LongTensor(src_int)
        example = {'src': src_tensor, 'src_len': src_len}
        return example

    def get_num_instances(self):
        return self.num_instances


def test():
    vocab = Vocab(['../small_test_data/train.src', '../small_test_data/train.tgt'])
    dataset = ControllableDataset('../small_test_data/train.src', '../small_test_data/train.tgt', 'unit_test', vocab,
                                  vocab, 15, 15)
    # print(dataset)
    vocab.print_vocab()
    # for i in range(len(dataset)):
    #    sample = dataset[i]
    #    print(i, " ", sample)
    dataloader = data_utils.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1)
    for sample_batched in dataloader:
        print('#################')
        print('src', sample_batched['src'])
        print(sample_batched['src_len'])
        print(sample_batched['control_tensor'])
        print(sample_batched['tgt'])

        break




if __name__ == "__main__":
    test()


