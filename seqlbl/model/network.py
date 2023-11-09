import tensorflow as tf


def sequence_lengths(x):
    mask = tf.greater(x, 0)
    return tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)


class SequenceLabeler(object):
    def __init__(self, args):
        self._build_embeddings(args)

        self.train = self._build_train_function(args)
        self.eval = self._build_eval_function(args)

        self.make_train_summary = self._build_train_summary_function()
        self.make_dev_summary = self._build_dev_summary_function()

    def initialize_global_variables(self, session):
        feed_dict = dict()
        if self.word_pt_embeddings is not None:
            feed_dict[self.word_pt_embeddings_ph] = self._word_pt_embeddings
        if self.tag_pt_embeddings is not None:
            feed_dict[self.tag_pt_embeddings_ph] = self._tag_pt_embeddings
        session.run(tf.global_variables_initializer(), feed_dict=feed_dict)

    def _build_train_summary_function(self):
        with tf.variable_scope('train_summary/'):
            x_f1 = tf.placeholder(tf.float32,
                                  shape=None,
                                  name='x_clf_acc')
            x_acc = tf.placeholder(tf.float32,
                                   shape=None,
                                   name='x_att_acc')

            tf.summary.scalar('train_f1', x_f1, collections=['train_summary'])
            tf.summary.scalar('train_acc', x_acc, collections=['train_summary'])

            summary = tf.summary.merge_all(key='train_summary')

        def f(session, clf_acc, att_acc):
            feed_dict = {
                x_f1: clf_acc,
                x_acc: att_acc,
            }
            return session.run(summary, feed_dict=feed_dict)

        return f

    def _build_dev_summary_function(self):
        with tf.variable_scope('dev_summary/'):
            x_loss = tf.placeholder(tf.float32,
                                    shape=None,
                                    name='x_loss')
            x_f1 = tf.placeholder(tf.float32,
                                  shape=None,
                                  name='x_clf_acc')
            x_acc = tf.placeholder(tf.float32,
                                   shape=None,
                                   name='x_att_acc')

            tf.summary.scalar('dev_loss', x_loss, collections=['dev_summary'])
            tf.summary.scalar('dev_f1', x_f1, collections=['dev_summary'])
            tf.summary.scalar('dev_acc', x_acc, collections=['dev_summary'])

            summary = tf.summary.merge_all(key='dev_summary')

        def f(session, loss, clf_acc, att_acc):
            feed_dict = {
                x_loss: loss,
                x_f1: clf_acc,
                x_acc: att_acc,
            }
            return session.run(summary, feed_dict=feed_dict)

        return f

    def _build_placeholders(self):
        x_word = tf.placeholder(tf.int32,
                                shape=(None, None),
                                name='x_word')
        if self.word_pt_embeddings is not None:
            x_pt_word = tf.placeholder(tf.int32,
                                       shape=(None, None),
                                       name='x_pt_word')
        else:
            x_pt_word = None
        x_tag = tf.placeholder(tf.int32,
                               shape=(None, None),
                               name='x_tag')
        if self.tag_pt_embeddings is not None:
            x_pt_tag = tf.placeholder(tf.int32,
                                      shape=(None, None),
                                      name='x_pt_tag')
        else:
            x_pt_tag = None
        y_label = tf.placeholder(tf.int32,
                                 shape=(None, None),
                                 name='y_label')
        return [x_word, x_pt_word, x_tag, x_pt_tag, y_label]

    def _build_embeddings(self, args):
        with tf.variable_scope('embeddings'):
            if args.word_embeddings is not None:
                self.word_embeddings = tf.get_variable('word_embeddings',
                                                       shape=(args.no_words, args.word_dim),
                                                       dtype=tf.float32,
                                                       initializer=tf.zeros_initializer,
                                                       regularizer=tf.contrib.layers.l2_regularizer(args.word_l2))
                self.word_pt_embeddings_ph = tf.placeholder(tf.float32,
                                                            shape=args.word_embeddings.shape,
                                                            name='word_pt_embeddings_ph')
                self.word_pt_embeddings = tf.Variable(self.word_pt_embeddings_ph,
                                                      name='word_pt_embeddings',
                                                      trainable=False)
                self._word_pt_embeddings = args.word_embeddings
            else:
                self.word_embeddings = tf.get_variable('word_embeddings',
                                                       shape=(args.no_words, args.word_dim),
                                                       dtype=tf.float32,
                                                       initializer=tf.variance_scaling_initializer(scale=3.,
                                                                                                   distribution='uniform',
                                                                                                   mode='fan_in'),
                                                       regularizer=tf.contrib.layers.l2_regularizer(args.word_l2))
                self.word_pt_embeddings = None

            if args.tag_embeddings is not None:
                self.tag_embeddings = tf.get_variable('tag_embeddings',
                                                      shape=(args.no_tags, args.tag_dim),
                                                      dtype=tf.float32,
                                                      initializer=tf.zeros_initializer,
                                                      regularizer=tf.contrib.layers.l2_regularizer(args.tag_l2))
                self.tag_pt_embeddings_ph = tf.placeholder(tf.float32,
                                                           shape=args.tag_embeddings.shape,
                                                           name='tag_pt_embeddings_ph')
                self.tag_pt_embeddings = tf.Variable(self.tag_pt_embeddings_ph,
                                                     name='tag_pt_embeddings',
                                                     trainable=False)
                self._tag_pt_embeddings = args.tag_embeddings
            else:
                self.tag_embeddings = tf.get_variable('tag_embeddings',
                                                      shape=(args.no_tags, args.word_dim),
                                                      dtype=tf.float32,
                                                      initializer=tf.variance_scaling_initializer(scale=3.,
                                                                                                  distribution='uniform',
                                                                                                  mode='fan_in'),
                                                      regularizer=tf.contrib.layers.l2_regularizer(args.tag_l2))
                self.tag_pt_embeddings = None

    def _build_input_layers(self, args, is_training):
        def f(x_word, x_pt_word, x_tag, x_pt_tag):
            Ew = tf.nn.embedding_lookup(self.word_embeddings, x_word)
            if self.word_pt_embeddings is not None:
                Ew_pt = tf.nn.embedding_lookup(self.word_pt_embeddings, x_pt_word)
                Ew += Ew_pt
            Et = tf.nn.embedding_lookup(self.tag_embeddings, x_tag)
            if self.tag_pt_embeddings is not None:
                Et_pt = tf.nn.embedding_lookup(self.tag_pt_embeddings, x_pt_tag)
                Et += Et_pt
            input = tf.concat([Ew, Et], axis=-1)
            if is_training:
                input = tf.nn.dropout(input,
                                      keep_prob=1 - args.input_dropout)
            return input

        return f

    def _build_hidden_layers(self, args, is_training):
        def f(input, lengths):
            hidden = input
            with tf.variable_scope('lstms'):
                for i in range(args.num_lstms):
                    fw_cell = tf.nn.rnn_cell.LSTMCell(args.hidden_dim)
                    if is_training:
                        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell,
                                                                state_keep_prob=1 - args.recurrent_dropout,
                                                                output_keep_prob=1 - args.dropout,
                                                                variational_recurrent=True,
                                                                dtype=tf.float32)

                    bw_cell = tf.nn.rnn_cell.LSTMCell(args.hidden_dim)
                    if is_training:
                        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell,
                                                                state_keep_prob=1 - args.recurrent_dropout,
                                                                output_keep_prob=1 - args.dropout,
                                                                variational_recurrent=True,
                                                                dtype=tf.float32)
                    with tf.variable_scope('lstm%d' % i):
                        (fw, bw), (fw_s, bw_s) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, hidden,
                                                                                 sequence_length=lengths,
                                                                                 dtype=tf.float32)
                    hidden = tf.concat([fw, bw], axis=-1)

            with tf.variable_scope('mlp'):
                for _ in range(args.num_mlps):
                    hidden = tf.layers.dense(hidden,
                                             units=args.mlp_dim,
                                             activation=tf.nn.leaky_relu,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(args.l2))
                    if is_training:
                        hidden = tf.nn.dropout(hidden, keep_prob=1 - args.dropout)

            return hidden

        return f

    def _build(self, args, is_training):
        with tf.variable_scope('placeholders'):
            x_word, x_pt_word, x_tag, x_pt_tag, y_label = self._build_placeholders()

        input_layers = self._build_input_layers(args, is_training)
        hidden_layers = self._build_hidden_layers(args, is_training)

        with tf.variable_scope('input_layers', reuse=tf.AUTO_REUSE):
            input = input_layers(x_word, x_pt_word, x_tag, x_pt_tag)
            mask = tf.greater(x_word, 0)
            lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)

        with tf.variable_scope('hidden_layers', reuse=tf.AUTO_REUSE):
            hidden = hidden_layers(input, lengths)

        with tf.variable_scope('output_layers', reuse=tf.AUTO_REUSE):
            logit = tf.layers.dense(hidden,
                                    units=args.no_labels,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(args.l2))

            if args.use_crf:
                log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(logit, y_label, lengths)
                loss = -tf.reduce_sum(log_likelihood) / tf.cast(tf.reduce_sum(lengths), tf.float32)
                prediction, probability = tf.contrib.crf.crf_decode(logit, transition_params, lengths)
                prediction = tf.where(mask, prediction, tf.zeros_like(prediction) - 1)
            else:
                mask_float = tf.to_float(mask)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label, logits=logit)
                cross_entropy *= mask_float
                loss = tf.reduce_sum(cross_entropy) / tf.reduce_sum(mask_float)

                probability = tf.nn.softmax(logit)
                probability *= tf.expand_dims(mask_float, -1)

                prediction = tf.to_int32(tf.argmax(probability, axis=-1))
                prediction = tf.where(mask, prediction, tf.zeros_like(prediction) - 1)

        inputs = [x_word, x_pt_word, x_tag, x_pt_tag, y_label]
        outputs = {
            'logit': logit,
            'probability': probability,
            'prediction': prediction
        }

        return inputs, outputs, loss

    def _build_train_function(self, args):
        with tf.name_scope('train'):
            inputs, outputs, loss = self._build(args, is_training=True)

            self.iteration = tf.Variable(-1, name='iteration', trainable=False)
            with tf.variable_scope('learning_rate'):
                self.learning_rate = tf.train.exponential_decay(learning_rate=args.learning_rate,
                                                                global_step=self.iteration,
                                                                decay_steps=args.decay_step,
                                                                decay_rate=args.decay_rate)

            with tf.variable_scope('train_summary/'):
                tf.summary.scalar('lr', self.learning_rate, collections=['train_instant_summary'])
                tf.summary.scalar('train_loss', loss, collections=['train_instant_summary'])
                summary = tf.summary.merge_all(key='train_instant_summary')

            if args.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif args.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            elif args.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif args.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            elif args.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            else:
                raise Exception('Unknown optimizer:', args.optimizer)

            gradients_vars = optimizer.compute_gradients(loss, tf.trainable_variables())
            if args.max_norm is not None:
                with tf.variable_scope('max_norm'):
                    gradients = [gv[0] for gv in gradients_vars]
                    gradients, _ = tf.clip_by_global_norm(gradients, args.max_norm)
                    gradients_vars = [(g, gv[1]) for g, gv in zip(gradients, gradients_vars)]

            with tf.variable_scope('optimizer'):
                train_step = optimizer.apply_gradients(gradients_vars, global_step=self.iteration)

        def f(vars, session):
            feed_dict = dict()
            for input, var in zip(inputs, vars):
                if input is not None:
                    feed_dict[input] = var
            _, output_, loss_, summary_ = session.run([train_step, outputs, loss, summary],
                                                      feed_dict=feed_dict)
            return output_, loss_, summary_

        return f

    def _build_eval_function(self, args):
        with tf.name_scope('eval'):
            inputs, outputs, loss = self._build(args, is_training=False)

        def f(vars, session):
            feed_dict = dict()
            for input, var in zip(inputs, vars):
                if input is not None:
                    feed_dict[input] = var
            return session.run([outputs, loss], feed_dict=feed_dict)

        return f
