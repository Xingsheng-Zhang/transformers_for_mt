import json

def build_vocab(data_file_path, bos_token='[BOS]', eos_token='[EOS]', pad_token='[PAD]', unk_token='[UNK]', max_vocab=None):
    dataset = []
    with open(data_file_path, "r", encoding="utf-8") as f:
        for line in f:
            value = line[:-1]  # each line is endding by '\n'
            dataset.append(value)
    special_word = 0
    idx2word = []
    vocabulary = {} # count the frequency
    if unk_token is not None:
        vocabulary[unk_token] = 1
        special_word += 1
        idx2word.append(unk_token)
    if bos_token is not None:
        vocabulary[bos_token] = 1
        special_word += 1
        idx2word.append(bos_token)
    if eos_token is not None:
        vocabulary[eos_token] = 1
        special_word += 1
        idx2word.append(eos_token)
    if pad_token is not None:
        vocabulary[pad_token] = 1
        special_word += 1
        idx2word.append(pad_token)
    for data_str in dataset:
        data_list = data_str.split(' ')
        for word in data_list:
            vocabulary[word] = vocabulary.get(word, 0) + 1
    if max_vocab is None:
        idx2word.extend(sorted(vocabulary, key=vocabulary.get, reverse=True))
    else:
        idx2word.extend(sorted(vocabulary, key=vocabulary.get, reverse=True)[:max_vocab - special_word])

    word2idx = {idx2word[idx]: idx for idx, _ in enumerate(idx2word)}
    return word2idx, idx2word


def load_vocab(json_file_path, encoding='utf-8', **kwargs):
    with open(json_file_path, 'r', encoding=encoding) as fin:
        tmp_json = json.load(fin, **kwargs)
    return tmp_json

def dump_vocab(obj, json_file_path, encoding='utf-8', ensure_ascii=False, indent=2, **kwargs):
    with open(json_file_path, 'w', encoding=encoding) as fout:
        json.dump(obj, fout,
                  ensure_ascii=ensure_ascii,
                  indent=indent,
                  **kwargs)

def gen_vocab(dataset_file_path, vocab_file_path, max_vocab=None):
    word2idx, idx2word = build_vocab(dataset_file_path, max_vocab=max_vocab)
    dump_vocab(idx2word, vocab_file_path)


class Vocab(object):
    def __init__(self, vocab_file):
        vocab = self.load_vocab(vocab_file)
        self.idx2word = vocab
        self.word2idx = {self.idx2word[idx]: idx for idx, _ in enumerate(self.idx2word)}
    def load_vocab(self, vocab_file, encoding='utf-8', **kwargs):
        with open(vocab_file, 'r', encoding=encoding) as fin:
            vocab = json.load(fin, **kwargs)
        return vocab

    def convert_tokens_to_ids(self, tokens):
        """

        :param tokens: the list of tokens or a str
        :return: ids the list of idx
        """
        assert self.word2idx is not None
        if isinstance(tokens, str):
            tokens = tokens.split(' ')
        ids = []
        for token in tokens:
            if token in self.word2idx:
                idx = self.word2idx[token]
                ids.append(idx)
            else:
                idx = self.word2idx['[UNK]']
                ids.append(idx)
        return ids

    def convert_ids_to_tokens(self, ids):
        """

        :param ids: the list of idx
        :return: words the list of word
        """
        assert self.idx2word is not None
        words = []
        for idx in ids:
            if idx < len(self.idx2word):
                word = self.idx2word[idx]
            else:
                word = '[UNK]'
            words.append(word)
        return words

if __name__ == '__main__':
    en_data_file_path = '../data/train.en.tok'
    zh_data_file_path = '../data/train.zh.tok'
    vocab_file_path = '../vocab.txt'
    max_seq_len = 80
    # test_data = preprocess.load_data_from_file(src_data_file_path)
    en_vocab_file_path = '../vocab/en_vocab.json'
    zh_vocab_file_path = '../vocab/zh_vocab.json'
    gen_vocab(en_data_file_path, en_vocab_file_path, max_vocab=None)
    gen_vocab(zh_data_file_path, zh_vocab_file_path, max_vocab=None)
    # en_vocab = Vocab(en_vocab_file_path)
    # zh_vocab = Vocab(zh_vocab_file_path)
    # test_str = 'i'
    # print(en_vocab.convert_tokens_to_ids(test_str))
