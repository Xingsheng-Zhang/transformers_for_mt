
from tqdm import tqdm, trange
import numpy as np
import torch
import tokenize
from torch.utils.data.dataset import TensorDataset
from .vocab import Vocab

class Preprocess(object):
    def __init__(self, max_seq_len, src_vocab_path, trg_vocab_path, bos_token='[BOS]', eos_token='[EOS]', pad_token='[PAD]', unk_token='[UNK]'):
        self.max_seq_len = max_seq_len
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.src_vocab = Vocab(src_vocab_path)
        self.trg_vocab = Vocab(trg_vocab_path)

    def load_dataset(self, src_file_path, trg_file_path):
        src_ori_data = self.load_data_from_file(src_file_path)
        trg_ori_data = self.load_data_from_file(trg_file_path)
        # 去除eos位置的mask
        # trg_trui_matrix = self.subsequent_mask(self.max_seq_len - 1)  # [1, max_seq_len, max_seq_len]
        # trg_trui_matrix = torch.from_numpy(trg_trui_matrix)  # [1, max_seq_len, max_seq_len]
        src_token_ids_list = []
        src_mask_ids_list = []
        trg_token_ids_list = []
        trg_mask_ids_list = []
        for idx, (src_ex, trg_ex) in enumerate(tqdm(zip(src_ori_data, trg_ori_data), desc='Data loading', total=len(src_ori_data))):
            src_ex_token_ids, src_ex_mask_ids = self.pad_seq(src_ex, lang='src')
            trg_ex_token_ids, trg_ex_mask_ids = self.pad_seq(trg_ex, include_bos=True, include_eos=True, lang='trg')
            trg_ex_mask_ids = trg_ex_mask_ids[:-1] # 去除最后一个位置eos的mask
            src_token_ids_list.append(src_ex_token_ids)
            src_mask_ids_list.append(src_ex_mask_ids)
            trg_token_ids_list.append(trg_ex_token_ids)
            trg_mask_ids_list.append(trg_ex_mask_ids)
        # src_token_ids = np.array(src_token_ids_list) # [num_examples, max_seq_len]
        # src_mask_ids = np.array(src_mask_ids_list) # [num_examples, max_seq_len]
        # trg_token_ids = np.array(trg_token_ids_list) # [num_examples, max_seq_len]
        # trg_mask_ids = torch.tensor(trg_mask_ids_list).unsqueeze(-2) # [num_examples, max_seq_len]
        # trg_mask_ids = trg_mask_ids & trg_trui_matrix # [num_examples, max_seq_len, max_seq_len]

        t_src_token_ids = torch.tensor(src_token_ids_list, dtype=torch.long)
        t_src_mask_ids = torch.tensor(src_mask_ids_list, dtype=torch.uint8)
        t_trg_token_ids = torch.tensor(trg_token_ids_list, dtype=torch.long)
        t_trg_mask_ids = torch.tensor(trg_mask_ids_list, dtype=torch.uint8)
        tensor_dataset = TensorDataset(t_src_token_ids, t_src_mask_ids, t_trg_token_ids, t_trg_mask_ids)
        return tensor_dataset

    def pad_seq(self, seq_str_data, include_bos=False, include_eos=False, lang='src'):
        """
        :param seq_str_data:
        :param init_token: str '<b>'
        :param eos_token: str '<e>'
        :param pad_token:
        :return:
        """
        v_max_seq_len = self.max_seq_len
        if include_bos:
            v_max_seq_len -= 1
        if include_eos:
            v_max_seq_len -= 1
        tokens = seq_str_data.split(' ')
        seq_len = len(tokens)
        if seq_len >= v_max_seq_len:
            tokens = tokens[:v_max_seq_len]
        if include_bos:
            tokens.insert(0, self.bos_token)
        if include_eos:
            tokens.append(self.eos_token)
        mask_ids = [1] * len(tokens)
        while len(tokens) < self.max_seq_len:
            tokens.append(self.pad_token)
            mask_ids.append(0)
        # print('len(tokens):{}'.format(len(tokens)))
        if lang == 'src':
            token_ids = self.src_vocab.convert_tokens_to_ids(tokens)
        elif lang == 'trg':
            token_ids = self.trg_vocab.convert_tokens_to_ids(tokens)
        else:
            raise Exception('Unsupport the lang')
        # print('len(token_ids):{}'.format(len(token_ids)))
        assert len(token_ids) == len(mask_ids) == self.max_seq_len
        return token_ids, mask_ids

    def load_data_from_file(self, file_path, encoding="utf-8", **kwargs):
        dict_list = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                value = line[:-1]  # each line is endding by '\n'
                dict_list.append(value)
        return dict_list

    def subsequent_mask(self, size):
        """Mask out subsequent positions."""
        attn_shape = (1, size, size)
        subseq_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return (subseq_mask == 0) + 0

    def decode_sequence(self, seq_ids, lang='trg'):
        """ convert from seq_ids to seq words

        seq_ids: a list of id"""
        if lang == 'src':
            seq_words = self.src_vocab.convert_ids_to_tokens(seq_ids)
        elif lang == 'trg':
            seq_words = self.trg_vocab.convert_ids_to_tokens(seq_ids)
        else:
            raise Exception('Unsupport the lang')
        seq_str = ' '.join(seq_words)
        return seq_str








if __name__ == '__main__':

    src_data_file_path = '../data/train.en.tok'
    trg_data_file_path = '../data/train.zh.tok'
    vocab_file_path = '../vocab.txt'
    max_seq_len = 80
    preprocess = Preprocess(max_seq_len, bos_token='[BOS]', eos_token='[EOS]', pad_token='[PAD]')
    # test_data = preprocess.load_data_from_file(src_data_file_path)
    src_word2idx, src_idx2word = preprocess.build_vocab(src_data_file_path)
    trg_word2idx, trg_idx2word = preprocess.build_vocab(trg_data_file_path)

    # print(word2idx)