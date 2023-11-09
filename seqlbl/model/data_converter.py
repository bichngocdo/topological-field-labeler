import numpy as np

from seqlbl.data.data_converters import AbstractDataConverter


def convert_to_tensor(sequences, type='int32'):
    return np.asarray(sequences).astype(type)


def convert_to_tensor_padded(sequences, type='int32', value=0):
    lengths = [len(s) for s in sequences]
    batch_size = len(sequences)
    max_length = max(lengths)
    tensor = np.zeros((batch_size, max_length)).astype(type) + value
    for i, s in enumerate(sequences):
        tensor[i, :lengths[i]] = s
    return tensor


class DataConverter(AbstractDataConverter):
    def __init__(self):
        super(AbstractDataConverter, self).__init__()

    def convert(self, batch):
        results = list()
        for sequence in batch:
            if sequence is not None:
                results.append(convert_to_tensor_padded(sequence))
            else:
                results.append(None)
        return results
