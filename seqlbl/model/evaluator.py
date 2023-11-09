from collections import OrderedDict


class Stats(object):
    def __init__(self, name, main_labels, ignore_labels):
        self.name = name
        self.main_labels = set(main_labels)
        self.ignore_labels = set(ignore_labels)

        self.loss = 0.
        self.time = 0.
        self.num_iterations = 0

        # Sentence accuracy
        self.num_correct_sentences = 0
        self.num_sentences = 0

        # Label accuracy
        self.num_correct_tokens = 0
        self.num_tokens = 0

        # Precision / Recall
        self.num_correct_labels = 0
        self.num_gold_labels = 0
        self.num_retrieved_labels = 0

    def reset(self):
        self.loss = 0.
        self.time = 0.
        self.num_iterations = 0

        self.num_correct_sentences = 0
        self.num_sentences = 0

        self.num_correct_tokens = 0
        self.num_tokens = 0

        self.num_correct_labels = 0
        self.num_gold_labels = 0
        self.num_retrieved_labels = 0

    def update(self, loss, time, gold, pred):
        self.loss += loss
        self.time += time
        self.num_iterations += 1

        for gold_sentence, pred_sentence in zip(gold, pred):
            num_incorrect_tokens = 0
            for gold_label, pred_label in zip(gold_sentence, pred_sentence):
                if gold_label not in self.ignore_labels:
                    self.num_tokens += 1
                    if gold_label == pred_label:
                        self.num_correct_tokens += 1
                    else:
                        num_incorrect_tokens += 1
                    if gold_label in self.main_labels:
                        self.num_gold_labels += 1
                    if pred_label in self.main_labels:
                        self.num_retrieved_labels += 1
                    if gold_label == pred_label and gold_label in self.main_labels:
                        self.num_correct_labels += 1
            self.num_sentences += 1
            self.num_correct_sentences += num_incorrect_tokens == 0

    def aggregate(self):
        results = OrderedDict()

        sent_acc = 1. * self.num_correct_sentences / self.num_sentences if self.num_sentences > 0 \
            else float('NaN')
        label_acc = 1. * self.num_correct_tokens / self.num_tokens if self.num_tokens > 0 \
            else float('NaN')
        p = 1. * self.num_correct_labels / self.num_retrieved_labels if self.num_retrieved_labels > 0 \
            else float('NaN')
        r = 1. * self.num_correct_labels / self.num_gold_labels if self.num_gold_labels > 0 \
            else float('NaN')
        f1 = 2. * p * r / (p + r) if p + r > 0 \
            else float('NaN')

        results['%s_loss' % self.name] = self.loss / self.num_iterations
        results['%s_rate' % self.name] = self.num_sentences / self.time
        results['%s_p' % self.name] = p
        results['%s_r' % self.name] = r
        results['%s_f1' % self.name] = f1
        results['%s_acc' % self.name] = label_acc
        results['%s_sent_acc' % self.name] = sent_acc

        self.reset()

        return results
