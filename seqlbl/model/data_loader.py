import numpy as np

from seqlbl.data.vocab import Vocab


def _convert_to_ids(sentences, vocab, level=0):
    if level == 0:
        return vocab[sentences]
    elif level == 1:
        return [vocab[item] for item in sentences]
    elif level == 2:
        return [[vocab[item] for item in sentence] for sentence in sentences]
    elif level == 3:
        return [[[vocab[item] for item in items] for items in sentence] for sentence in sentences]
    else:
        raise Exception('List depth not supported')


class DataLoader(object):
    def __init__(self):
        self.word_vocab = None
        self.word_pt_vocab = None
        self.tag_vocab = None
        self.tag_pt_vocab = None
        self.label_vocab = None

        self.word_cutoff_threshold = 0
        self.lowercase = False

        self.NONE = '-NONE-'
        self.UNKNOWN = '-UNKNOWN-'

    def init_pretrained_vocab(self, vocab_name, words):
        str2id = dict()
        id2str = list()
        id = 0

        for str in [self.NONE, self.UNKNOWN]:
            str2id[str] = id
            id2str.append(str)
            id += 1

        for word in words:
            str2id[word] = id
            id2str.append(word)
            id += 1

        vocab = Vocab()
        vocab.str2id = str2id
        vocab.id2str = id2str
        vocab.unk_str = self.UNKNOWN
        vocab.unk_id = 1
        self.__setattr__(vocab_name, vocab)

    def load_embeddings(self, embeddings, have_unk=False):
        if have_unk:
            embeddings = np.pad(embeddings, ((1, 0), (0, 0)), 'constant', constant_values=0)
        else:
            embeddings = np.pad(embeddings, ((2, 0), (0, 0)), 'constant', constant_values=0)
        return embeddings

    def init_vocabs(self, raw_data):
        words, tags, labels = raw_data

        self.word_vocab = Vocab()
        self.tag_vocab = Vocab()
        self.label_vocab = Vocab()

        self.word_vocab.init(words, unk_str=self.UNKNOWN,
                             special_strs=[self.NONE],
                             cutoff_threshold=self.word_cutoff_threshold)
        self.tag_vocab.init(tags, unk_str=self.UNKNOWN,
                            special_strs=[self.NONE])
        self.label_vocab.init(labels, unk_str=self.UNKNOWN,
                              special_strs=[self.NONE])

    def load(self, raw_data):
        words, tags, labels = raw_data
        results = list()
        results.append(_convert_to_ids(words, self.word_vocab, level=2))
        if self.word_pt_vocab is not None:
            results.append(_convert_to_ids(words, self.word_pt_vocab, level=2))
        else:
            results.append(None)
        results.append(_convert_to_ids(tags, self.tag_vocab, level=2))
        if self.tag_pt_vocab is not None:
            results.append(_convert_to_ids(tags, self.tag_pt_vocab, level=2))
        else:
            results.append(None)
        results.append(_convert_to_ids(labels, self.label_vocab, level=2))
        return results
