
import numpy as np
import tensorflow as tf
import pandas as pd
import tokenization
from nltk.tokenize import TweetTokenizer
import collections
import ast
import time
import json


def read_movie_data():

    columns = ['character1', 'character2', 'movietitle', 'conversation']

    movie_conver = []
    with open('data/movie_conversations.txt', 'r') as file:
        for l in file:
            lines = l.split('+++$+++')
            line_content = eval(lines[-1].rstrip())
            lines = [x.strip() for x in lines[:-1]]
            lines.append(line_content)
            movie_conver.append(lines)
    movie_conver = pd.DataFrame(movie_conver, columns=columns)

    columns = ['line', 'character_id', 'movietitle', 'character_name', 'text']

    movie_lines = []
    with open('data/movie_lines.txt', 'r', errors='ignore') as file:
        for l in file:
            lines = l.split('+++$+++')
            lines[-1] = lines[-1].rstrip()
            lines = [x.strip() for x in lines]
            movie_lines.append(lines)
    movie_lines = pd.DataFrame(movie_lines, columns=columns)
    # %%
    assert len(movie_lines['line'].unique()) == len(movie_lines['line'])
    movie_lines.set_index('line', inplace=True)

    return movie_conver, movie_lines

def select_movie_by_title(movie_conver, movie_lines, number_movies = 300):

    movie_title = movie_conver['movietitle'].unique()
    select_title = movie_title[:number_movies]

    select_movie_conver = movie_conver[np.isin(movie_conver['movietitle'], select_title)]
    select_movie_lines = movie_lines[np.isin(movie_lines['movietitle'], select_title)]

    return select_movie_conver, select_movie_lines


class Example():

    def __init__(self, input_sentences, output_sentences):

        self.input_sentences = input_sentences

        self.output_sentences = output_sentences


def convert_toIndex(tokens, word_dict):
    index_features = []
    for token in tokens:
        if not token in word_dict:
            word_dict[token] = len(word_dict)

        index_features.append(word_dict[token])
    return index_features


def truncate_sequence(text_a, text_b, max_len):
    while len(text_a) + len(text_b) + 3 >= max_len:
        if len(text_b) > len(text_a):
            text_b.pop()
        else:
            text_a.pop()


def convert_toExample(num_i, conversation, tokenizer, max_input_seq, max_output_seq, word_dict):
    #     convert text into examples

    num_convers = len(conversation)
    for i, _ in enumerate(conversation):

        if i < num_convers - 1:
            num_i += 1

            input_texts = tokenizer.tokenize(conversation[i])
            response = tokenizer.tokenize(conversation[i + 1])

            if len(input_texts) == 0 or len(response) == 0:
                continue

            if len(input_texts) > max_input_seq or len(response) > max_output_seq - 1:
                continue

            response = response + ['[END]']

            input_id = convert_toIndex(input_texts, word_dict)
            output_id = convert_toIndex(response, word_dict)

            if len(input_id) < max_input_seq:
                input_id.extend([0] * (max_input_seq - len(input_id)))

            if len(output_id) < max_output_seq:
                output_id.extend([0] * (max_output_seq - len(output_id)))

            assert len(input_id) == max_input_seq
            assert len(output_id) == max_output_seq

            if num_i % 5000 == 0:
                tf.logging.info('input text: %s' % ' '.join(input_texts))
                tf.logging.info('input id: %s' % ' '.join([str(x) for x in input_id]))
                tf.logging.info('output text: %s' % ' '.join(response))
                tf.logging.info('output id: %s' % ' '.join([str(x) for x in output_id]))

            yield num_i, Example(input_id, output_id)


def build_dataSet(movie_conver, movie_lines, max_input_seq, max_output_seq, tokenizer):

    word_dict = collections.OrderedDict()
    word_dict['#UNTOKEN#'] = 0
    word_dict['[START]'] = 1
    num_i = 0

    examples = []
    for index, row in movie_conver.iterrows():
        line_index = row['conversation']
        conversation = movie_lines.loc[line_index, 'text']

        if len(conversation) > 2:
            for num_i, example in convert_toExample(num_i, conversation, tokenizer, max_input_seq, max_output_seq,
                                                    word_dict):
                examples.append(example)
    return word_dict, examples


def save_data(examples, file_path):

    writer = tf.python_io.TFRecordWriter(file_path)

    for example in examples:
        features = collections.OrderedDict()
        features["input_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=example.input_sentences))
        features["output_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=example.output_sentences))

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()

def process_data(select_movie_conver, select_movie_lines):

    tokenizer = TweetTokenizer()

    max_input_seq = 10
    max_output_seq = 10

    word_dict, examples = build_dataSet(select_movie_conver, select_movie_lines, max_input_seq, max_output_seq, tokenizer)
    print(len(examples))

    from sklearn.model_selection import train_test_split

    train_example, val_examples = train_test_split(examples, test_size=0.2, random_state=3)
    save_data(train_example, 'data/train')
    save_data(val_examples, 'data/val')

    with open('data/vocab.json', 'w') as fp:
        json.dump(word_dict, fp)


def encoder(input_ids, embeding, vocab_size, embeding_size):
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        input_embeding = tf.nn.embedding_lookup(embeding, input_ids)

        rnn_layers = [tf.nn.rnn_cell.GRUCell(size) for size in [512, 256]]

        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        sequence_length = tf.reduce_sum(tf.cast(input_ids > 0, dtype=tf.int32), axis=-1)

        outputs, state = tf.nn.dynamic_rnn(multi_rnn_cell, input_embeding, dtype=tf.float32,
                                           sequence_length=sequence_length)

    return outputs, state, sequence_length


def decoder(output_ids, embeding, vocab_size, attention_define, padding_size=10, training=True):
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):

        batch = tf.shape(output_ids)[0]

        rnn_layers = [tf.nn.rnn_cell.GRUCell(size) for size in [512, 256]]

        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        encoder_output, encoder_state, encoder_length, attention_unit = attention_define

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(attention_unit, encoder_output, encoder_length)

        attention_cell = tf.contrib.seq2seq.AttentionWrapper(multi_rnn_cell, attention_mechanism,
                                                             attention_layer_size=None)

        attention_cell = tf.contrib.rnn.OutputProjectionWrapper(attention_cell, vocab_size, reuse=tf.AUTO_REUSE)

        initial_state = attention_cell.zero_state(dtype=tf.float32, batch_size=batch)
        initial_state = initial_state.clone(cell_state=encoder_state)

        if training:
            input_embeding = tf.nn.embedding_lookup(embeding, output_ids)
            sequence_length = tf.count_nonzero(output_ids, axis=-1, dtype=tf.int32)
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=input_embeding,
                sequence_length=sequence_length,
            )
        else:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embeding, start_tokens=tf.zeros([batch], dtype=tf.int32) + word_dict['[START]'],
                end_token=word_dict['[END]'])

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=attention_cell,
            helper=helper,
            initial_state=initial_state)

        decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=output_length)

        padding_need = padding_size - tf.shape(decoder_outputs.rnn_output)[1]
        padding_zeros = tf.zeros([batch, padding_need, vocab_size])
        decoder_rnn = tf.concat([decoder_outputs.rnn_output, padding_zeros], axis=1)

    return decoder_rnn


def sequence_loss(prediction, label, decoder_length):

    one_hot_label = tf.one_hot(label, depth=tf.shape(prediction)[-1])
    per_example_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_label, logits=prediction)
    per_example_loss = per_example_loss * decoder_length

    #     per_example_loss = tf.where(decoder_length < 1, tf.stop_gradient(per_example_loss), per_example_loss)
    per_example_loss = tf.reduce_sum(per_example_loss, axis=-1) / tf.cast(tf.reduce_sum(decoder_length, axis=-1),
                                                                          dtype=tf.float32)
    return per_example_loss


def inferQA(questions, index2word, word_dict, tokenizer, max_input_seq=10):
    input_texts = tokenizer.tokenize(questions)
    input_ids = [word_dict[i] for i in input_texts if i in word_dict]

    if len(input_ids) > max_input_seq:
        raise 'sentence is too long'

    if len(input_ids) < max_input_seq:
        input_ids.extend([0] * (max_input_seq - len(input_ids)))

    input_ids = np.array(input_ids).reshape(-1, max_input_seq).astype(np.int64)

    print('question: %s' % ' '.join(questions))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        full_path = tf.train.latest_checkpoint('model')
        if not full_path:
            raise 'no model file exist'

        saver.restore(sess, full_path)
        sess.run('test_init_op', feed_dict={'input_plh:0': input_ids})
        prob = sess.run('prediction:0')[0]
        words = [index2word[i] for i in prob]
    print('answer: %s' % ' '.join(words))


class trainHook(tf.train.SessionRunHook):

    def __init__(self, metrics_op):
        #       dict of {name: metrics}
        self.tensor_name = list(metrics_op.keys())
        self.tensor_eval = list(metrics_op.values())
        self.epoch = 0
        super().__init__()

    def before_run(self, run_context):

        return tf.train.SessionRunArgs(self.tensor_eval)

    def after_run(self, run_context, run_values):

        eval_result = run_values.results
        self.epoch += 1
        if self.epoch % 100 == 0:
            for i, tensor in enumerate(eval_result):
                if isinstance(tensor,tuple):
                    eval_result, _ = tensor
                else:
                    eval_result = tensor
                print('%s is %s' % (self.tensor_name[i], eval_result))


def build_input_fn(input_file, input_length, output_length, mode):
    feature_description = {
        "input_ids": tf.FixedLenFeature([input_length], tf.int64),
        "output_ids": tf.FixedLenFeature([output_length], tf.int64),
    }

    def _parse_function(record):
        return tf.parse_single_example(record, feature_description)

    def input_fn(params):

        batch_size = params['batch_size']
        epoch = params['epoch']

        if not mode == 'test':
            data = tf.data.TFRecordDataset(input_file)
            data = data.map(lambda x: _parse_function(x))

            if mode == 'train':
                data = data.repeat(epoch)
                data = data.shuffle(buffer_size=1000)

            data = data.batch(batch_size)
            iter_ = data.make_one_shot_iterator()
            features = iter_.get_next()
            return features["input_ids"], features["output_ids"]

        else:
            inputs = tf.placeholder(shape=[None, input_length], dtype=tf.int64, name='input_plh')
            data = tf.data.Dataset.from_tensor_slices(inputs)
            data = data.batch(batch_size=tf.shape(inputs)[0])
            return data

    return input_fn


def model_fn(features, labels, mode, params):

    word_dict = params['word_dict']
    attention_unit = params['attention_unit']
    vocab_size = len(word_dict)
    embeding_size = params['embeding_size']

    if mode == tf.estimator.ModeKeys.PREDICT:
        train_output_ids = tf.zeros(shape=tf.shape(features))
    else:
        start_tokens = tf.zeros([tf.shape(labels)[0]], dtype=tf.int64) + word_dict['[START]']
        train_output_ids = tf.concat([tf.expand_dims(start_tokens, 1), labels], 1)

    with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
        embeding = tf.get_variable(name='word_embedding', shape=[vocab_size, embeding_size], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer())

    encoder_output, encoder_state, encoder_length = encoder(features, embeding, vocab_size, embeding_size)

    training = (mode == tf.estimator.ModeKeys.TRAIN)

    attention_define = (encoder_output, encoder_state, encoder_length, attention_unit)

    decoder_output = decoder(train_output_ids, embeding, vocab_size, attention_define, padding_size=output_length,
                             training=training)
    prediction = tf.argmax(decoder_output, axis=-1, name='prediction')

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=prediction)

    decoder_length_mask = tf.cast(labels > 0, dtype=tf.float32)

    per_example_loss = sequence_loss(decoder_output, labels, decoder_length_mask)
    loss = tf.metrics.mean(values=per_example_loss)
    total_loss = tf.reduce_mean(per_example_loss)

    accuracy = tf.metrics.accuracy(prediction, labels, weights=decoder_length_mask)

    if mode == tf.estimator.ModeKeys.TRAIN:

        logging_hook_train = trainHook({'total_loss': loss})
        step = tf.train.get_global_step()
        train_op = tf.train.AdamOptimizer().minimize(total_loss, global_step=step)
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op, training_hooks=[logging_hook_train],
                                          eval_metric_ops={'accuracy': accuracy, 'loss': loss})
    else:
        logging_hook_eval = trainHook({'accuracy_eval': accuracy})
        return tf.estimator.EstimatorSpec(mode, loss=total_loss,
                                          eval_metric_ops={'accuracy_eval': accuracy, 'loss_eval': loss},
                                          evaluation_hooks = [logging_hook_eval])


if __name__ == '__main__':

    with open('data/vocab.json', 'r') as fp:
        word_dict = json.load(fp)
    index2word = [word for word, i in word_dict.items()]

    input_file_train = 'data/train'
    input_file_eval = 'data/val'
    input_length, output_length = 10, 10
    input_fn_train = build_input_fn(input_file_train, input_length, output_length, mode='train')
    input_fn_eval = build_input_fn(input_file_eval, input_length, output_length, mode='eval')

    params = {'batch_size': 32,
              'word_dict': word_dict,
              'attention_unit': 32,
              'embeding_size': 300,
              'epoch': 200}

    estimator = tf.estimator.Estimator(model_fn, params=params, model_dir='model')

    train_spec =tf.estimator.TrainSpec(input_fn=input_fn_train)

    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_eval, throttle_secs = 30)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)



