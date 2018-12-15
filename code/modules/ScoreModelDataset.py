import torch
import numpy as np

class ScoreModelDataset():
    """Dataset class which defines basic Controllable dataset"""

    def __init__(self, src_lines, tgt_lines, controls, name, vocab_src, vocab_tgt, src_msl, tgt_msl):
        """
        Each line of input file is of format:  sentence TAB <list of control numbers separated by SPACE >
        :param src_data_file: source data file, one line per instance
        :param tgt_data_file: target data file, one line per instance
        :param src_msl: max sequence length
        :param tgt_msl: max sequence length
        """
        # sf = open(src_data_file, 'r')
        # tf = open(tgt_data_file, 'r')
        # self.src_lines = sf.readlines()
        # self.tgt_lines = tf.readlines()
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
        # src_sentence = src_line.split('    ')[0].strip('\r\n')
        # control_list = src_line.split('    ')[1].strip('\r\n')
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

        example = {'src': src_tensor, 'src_len': src_len, 'tgt': tgt_tensor, 'tgt_len': tgt_len,
                   'control_tensor': control_tensor}
        return example

    def get_control(self, s):
        # slist = s.split(' ')
        control_list = [long(x) for x in s]
        control_list = np.asarray(control_list, dtype='float32')
        control_tensor = torch.from_numpy(control_list)
        return control_tensor

    def get_num_instances(self):
        return self.num_instances