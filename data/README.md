# Data

The data to train topological field labelers are from the TüBa-D/Z corpus version 11.0.
* The data are preprocessed to match the format in the German dataset from the SPMRL Shared Task data:
  * Open (`(`, `[`, `»`) and closing brackets (`)`, `]`, `«`) are replaced with `“` and `”`
  * Preposition contractions (which have been split into definite articles and the prepositions)
    are rejoined into single tokens.
* The POS tags are assigned by the MarMoT tagger trained on the training set of the SPMRL dataset.
* The topological field tag of each word is the nearest topological field tag from the tree hierarchy
  from the original data.

The data are then split into train/dev/test sets:
* `split1024328679` contains train/dev/test sets with *gold* POS tags from the TüBa-D/Z corpus.
* `split1024328679_marmot` is exactly the same split as above but with *predicted* POS tags previously described.
