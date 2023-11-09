from seqlbl.model.conll import CoNLLFile


def read(fp, cols=(0, 1, 2)):
    with open(fp, 'r') as f:
        all_words = list()
        all_tags = list()
        all_labels = list()

        conll_file = CoNLLFile(f)
        for idx, sentence in enumerate(conll_file):
            words = list()
            tags = list()
            labels = list()
            for line in sentence:
                if line.startswith('#'):
                    continue
                items = line.rstrip().split('\t')
                words.append(items[cols[0]])
                tags.append(items[cols[1]])
                labels.append(items[cols[2]])
            all_words.append(words)
            all_tags.append(tags)
            all_labels.append(labels)
        return all_words, all_tags, all_labels
