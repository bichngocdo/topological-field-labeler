def write(fp, raw_data, results):
    with open(fp, 'w') as f:
        all_words, all_tags, _ = raw_data
        for words, tags, labels in zip(all_words, all_tags, results):
            for word, tag, label in zip(words, tags, labels):
                f.write('%s\t%s\t%s\n' % (word, tag, label))
            f.write('\n')
