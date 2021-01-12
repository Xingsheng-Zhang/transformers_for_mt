import sacrebleu


def compute_bleu(preds, labels):
    """

    :param preds: list []
    :param labels: list example_idx -> [sent_len]
    :return:
    """

    bleu = sacrebleu.corpus_bleu(labels, preds)
    return bleu.score

def test():
    # preds = [['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'],
    #     ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.']]
    preds = [['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.']]
    labels = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']
    print(compute_bleu(preds, labels))
if __name__ == '__main__':
    test()