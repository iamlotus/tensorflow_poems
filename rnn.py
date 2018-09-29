
import tensorflow as tf
import operator
import numpy as np
import collections
import math


def process_corpus(file_name,treat_corpus_as_byte):
    # contents -> list of word/byte
    contents = []

    buffer_size=1000
    with open(file_name, "rb") if treat_corpus_as_byte else open(file_name, "r", encoding='utf-8') as f:
        content=f.read(buffer_size)
        while content :
            contents+=content
            content = f.read(buffer_size)

    counter = collections.Counter(contents)
    # sort by value then key, to guarantee count_pairs is stable
    count_pairs = sorted(counter.items(), key=lambda kv: (kv[1],kv[0]), reverse=True)
    words, _ = zip(*count_pairs)

    word_int_map = dict(zip(words, range(len(words))))
    contents_vector = list(map(word_int_map.get,contents))

    print("## content size %d , %d words totally"%(len(contents_vector),len(word_int_map)))
    return contents_vector, word_int_map, words


class DataProvider:
    def __init__(self,content_vector, batch_size,seq_len,validate_ratio):
        x=np.copy(content_vector)
        y=np.zeros(x.shape,dtype=x.dtype)
        y[:-1]=x[1:]
        y[-1]=x[0]

        total_batch_num = len(content_vector) // (batch_size*seq_len)

        x_batches = []
        y_batches = []
        for i in range(total_batch_num):
            start_index = i * batch_size*seq_len
            end_index = start_index + batch_size*seq_len
            x_data= x[start_index:end_index].reshape(batch_size,seq_len)
            y_data = y[start_index:end_index].reshape(batch_size, seq_len)

            """
            x_data             y_data
            [6,2,4,6,9]       [2,4,6,9,9]
            [1,4,2,8,5]       [4,2,8,5,5]
            """
            x_batches.append(x_data)
            y_batches.append(y_data)

        x_batches=np.array(x_batches)
        y_batches=np.array(y_batches)

        validate_size=math.floor(total_batch_num * validate_ratio)
        train_size=total_batch_num-validate_size

        if validate_size==0:
            validate_size=1
            train_size-=1

        if train_size<=0:
            raise ValueError('total_batch_num %d is too small'%total_batch_num)


        validate_indices= np.random.choice(total_batch_num,validate_size,replace=False)
        train_indices=np.array(list(set(range(total_batch_num))-set(validate_indices)))

        self._x_train,self._x_validate=x_batches[train_indices],x_batches[validate_indices]
        self._y_train, self._y_validate = y_batches[train_indices], y_batches[validate_indices]

    @property
    def train_batch_num(self):
        return len(self._x_train)

    @property
    def validate_batch_num(self):
        return len(self._x_validate)

    def train_batch(self, train_batch_id):
        return self._x_train[train_batch_id], self._y_train[train_batch_id]

    def validate_batch(self, validate_batch_id):
        return self._x_validate[validate_batch_id], self._y_validate[validate_batch_id]


def rnn_model(cell_type, input_data, output_data, vocab_size, rnn_size, num_layers, batch_size,
              learning_rate):
    """
    construct rnn seq2seq model.
    :param cell_type: cell_type class
    :param input_data: input data placeholder
    :param output_data: output data placeholder
    :param vocab_size:
    :param rnn_size:
    :param num_layers:
    :param batch_size:
    :param learning_rate:
    :return:
    """
    end_points = {}

    if cell_type== 'rnn':
        cell_fun=tf.nn.rnn_cell.BasicRNNCell
    elif cell_type == 'gru':
        cell_fun=tf.nn.rnn_cell.GRUCell
    elif cell_type == 'lstm':
        cell_fun =tf.nn.rnn_cell.LSTMCell

    cell = cell_fun(rnn_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    if output_data is not None:
        initial_state = cell.zero_state(batch_size, tf.float32)
    else:
        initial_state = cell.zero_state(1, tf.float32)

    with tf.device("/cpu:0"),tf.name_scope("hidden"):
        embedding = tf.get_variable('embedding', initializer=tf.random_uniform(
            [vocab_size + 1, rnn_size], -1.0, 1.0))
        inputs = tf.nn.embedding_lookup(embedding, input_data)
        tf.summary.histogram("embedding", embedding)

    # [batch_size, ?, rnn_size] = [64, ?, 128]
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
    output = tf.reshape(outputs, [-1, rnn_size])

    weights = tf.Variable(tf.truncated_normal([rnn_size, vocab_size]))
    bias = tf.Variable(tf.zeros(shape=[vocab_size]))
    logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)
    # [?, vocab_size+1]

    if output_data is not None:
        # output_data must be one-hot encode
        labels = tf.one_hot(tf.reshape(output_data, [-1]), depth=vocab_size)
        # should be [?, vocab_size+1]

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        # loss shape should be [?, vocab_size+1]
        total_loss = tf.reduce_mean(loss)

        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('learning_rate', learning_rate)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        end_points['initial_state'] = initial_state
        end_points['output'] = output
        end_points['train_op'] = train_op
        end_points['total_loss'] = total_loss
        end_points['loss'] = loss
        end_points['last_state'] = last_state
    else:
        prediction = tf.nn.softmax(logits)

        end_points['initial_state'] = initial_state
        end_points['last_state'] = last_state
        end_points['prediction'] = prediction

    return end_points


def vocabulary_summary(vocabularies):
    print("## top ten vocabularies: %s" % str(vocabularies[:10]))
    print("## tail ten vocabularies: %s" % str(vocabularies[-10:]))


def corpus_summary(data_provider):
    x_train_0,y_train_0=data_provider.train_batch(0)
    print("## x_train[0][0][:10]: %s" % x_train_0[0][:10])
    print("## y_train[0][0][:10]: %s" % y_train_0[0][:10])

    x_validate_0, y_validate_0 = data_provider.validate_batch(0)
    print("## x_validate[0][0][:10]: %s" % x_validate_0[0][:10])
    print("## y_validate[0][0][:10]: %s" % y_validate_0[0][:10])

def print_args(flags):
    print('=' * 100)
    print('[FLAGS]')
    for k, v in sorted(flags.flag_values_dict().items(), key=operator.itemgetter(0)):
        if k in ['h', 'help', 'helpfull', 'helpshort']:
            continue

        if isinstance(v, str):
            print('%s = "%s"' % (k, v))
        else:
            print('%s = %s' % (k, v))
    print('=' * 100, flush=True)




