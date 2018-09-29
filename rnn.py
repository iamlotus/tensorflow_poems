import os
import tensorflow as tf
import time
import operator
import numpy as np
import collections
import sys
import math

tf.app.flags.DEFINE_string('mode', 'train', 'train or compose')
tf.app.flags.DEFINE_bool('treat_corpus_as_byte', True, 'treat corpus as byte or word, default(True) will treat input'
                                                       ' as byte stream, set to False will treat input as text(with '
                                                       'utf encode)')
tf.app.flags.DEFINE_integer('random_seed', 12345, 'random seed')
tf.app.flags.DEFINE_string('cell_type', 'lstm', 'lstm or rnn or gru')
tf.app.flags.DEFINE_integer('rnn_size', 128, 'rnn size.')
tf.app.flags.DEFINE_integer('num_layers', 2, 'layer num.')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate.')
tf.app.flags.DEFINE_float('validate_ratio', 0.05, 'how many data are used as validate set.')
tf.app.flags.DEFINE_string('input_name', 'small_poems', 'name of data(.txt)/model dir/model prefix')
tf.app.flags.DEFINE_integer('epochs', 50, 'train how many epochs.')
tf.app.flags.DEFINE_integer('train_sequence_len', 50, 'length of train sequence')
tf.app.flags.DEFINE_integer('print_every_steps', 100, '''save model every steps''')
tf.app.flags.DEFINE_integer('save_every_epoch', 1, '''save model every epoch''')
tf.app.flags.DEFINE_integer('gen_sequence_len', 500, 'length of gen sequence')
tf.app.flags.DEFINE_string('cuda_visible_devices', '2', '''[Train] visible GPU ''')


FLAGS = tf.app.flags.FLAGS

model_dir=os.path.join('model',FLAGS.input_name)
log_dir=os.path.join('logs',FLAGS.input_name)
model_file=os.path.join(model_dir,FLAGS.input_name)
corpus_path=os.path.join('data', FLAGS.input_name + ".txt")


def process_corpus(file_name):
    # contents -> list of word/byte
    contents = []

    buffer_size=1000
    with open(file_name, "rb") if FLAGS.treat_corpus_as_byte else open(file_name, "r", encoding='utf-8') as f:
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
    def __init__(self,content_vector, batch_size,seq_len):
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

        validate_size=math.floor(total_batch_num * FLAGS.validate_ratio)
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


def run_training():
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    poems_vector, word_to_int, vocabularies = process_corpus(corpus_path)
    data_provider=DataProvider(poems_vector, FLAGS.batch_size, FLAGS.train_sequence_len)

    print("## top ten vocabularies: %s" % str(vocabularies[:10]))
    print("## tail ten vocabularies: %s" % str(vocabularies[-10:]))
    x_train_0,y_train_0=data_provider.train_batch(0)
    print("## x_train[0][0][:10]: %s" % x_train_0[0][:10])
    print("## y_train[0][0][:10]: %s" % y_train_0[0][:10])

    x_validate_0, y_validate_0 = data_provider.validate_batch(0)
    print("## x_validate[0][0][:10]: %s" % x_validate_0[0][:10])
    print("## y_validate[0][0][:10]: %s" % y_validate_0[0][:10])

    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

    end_points = rnn_model(cell_type=FLAGS.cell_type, input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=FLAGS.rnn_size, num_layers=FLAGS.num_layers, batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    summary_op=tf.summary.merge_all()
    with tf.Session() as sess:

        train_writer = tf.summary.FileWriter(os.path.join(log_dir, "train"), sess.graph)
        validate_writer = tf.summary.FileWriter(os.path.join(log_dir, "validate"), sess.graph)

        sess.run(init_op)

        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(model_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print('## restore from the checkpoint %s, continue training...'%checkpoint,flush=True)
            start_epoch += int(checkpoint.split('-')[-1])+1
        else:
            print('## can not find checkpoint from "%s", restart training... '%model_dir,flush=True)


        for epoch in range(start_epoch, FLAGS.epochs):
            for train_batch_id in range(data_provider.train_batch_num):
                global_step = epoch * data_provider.train_batch_num + train_batch_id
                if global_step % FLAGS.print_every_steps==0:
                    x_train, y_train=data_provider.train_batch(train_batch_id)

                    train_loss, _, _,train_summary = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op'],
                        summary_op
                    ], feed_dict={input_data: x_train, output_targets: y_train})

                    validate_batch_id=global_step%data_provider.validate_batch_num
                    x_validate,y_validate=data_provider.validate_batch(validate_batch_id)
                    validate_loss, _, _, validate_summary = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op'],
                        summary_op
                    ], feed_dict={input_data: x_validate, output_targets: y_validate})

                    train_writer.add_summary(train_summary,global_step=global_step)
                    validate_writer.add_summary(validate_summary, global_step=global_step)
                    print('[%s] Global step: %d, Epoch: %d, Batch: %d, Train loss: %.8f, Validate loss: %.8f' %
                          (time.strftime('%Y-%m-%d %H:%M:%S'),global_step,epoch, train_batch_id, train_loss,
                           validate_loss), flush=True)
                else:
                    x_train, y_train = data_provider.train_batch(train_batch_id)
                    _, _ = sess.run([
                        end_points['last_state'],
                        end_points['train_op']
                    ], feed_dict={input_data: x_train, output_targets: y_train})
                global_step += 1
            if epoch % FLAGS.save_every_epoch == 0:
                saver.save(sess, model_file, global_step=epoch)
                print("[%s] Saving checkpoint for epoch %d"%(time.strftime('%Y-%m-%d %H:%M:%S'), epoch),flush=True)



def print_args():
    print('=' * 100)
    print('[FLAGS]')
    for k, v in sorted(FLAGS.flag_values_dict().items(), key=operator.itemgetter(0)):
        if k in ['h', 'help', 'helpfull', 'helpshort']:
            continue

        if isinstance(v, str):
            print('%s = "%s"' % (k, v))
        else:
            print('%s = %s' % (k, v))
    print('=' * 100, flush=True)


def to_word(predict, vocabs):
    predict = predict[0]
    predict /= np.sum(predict)
    sample = np.random.choice(np.arange(len(predict)), p=predict)

    return vocabs[sample]


def run_compose():
    batch_size = 1

    poems_vector, word_int_map, vocabularies = process_corpus(corpus_path)

    input_data = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(cell_type=FLAGS.cell_type, input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=FLAGS.rnn_size, num_layers=FLAGS.num_layers, batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint(model_dir)
        print('## loading corpus from %s' % checkpoint)
        saver.restore(sess, checkpoint)

        while True:

            input('## print any key to compose new sentence')

            # pick first word randomly
            word = vocabularies[np.random.randint(len(vocabularies))]
            output= [word]

            x = np.array([[word_int_map[word]]])
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x})
            # second word
            word = to_word(predict, vocabularies)

            i = 1
            while i < FLAGS.gen_sequence_len:
                output.append(word)
                i += 1
                x = np.zeros((1, 1))
                x[0, 0] = word_int_map[word]
                [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                                 feed_dict={input_data: x, end_points['initial_state']: last_state})
                word = to_word(predict, vocabularies)

            if FLAGS.treat_corpus_as_byte:
                sys.stdout.buffer.write(bytes(output))
                sys.stdout.buffer.write(b'\n')
                sys.stdout.buffer.write(b'\n')
                sys.stdout.flush()
            else:
                print("".join(output),end='\n',flush=True)

def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_visible_devices  # set GPU visibility in multiple-GPU environment
    print_args()

    np.random.seed(FLAGS.random_seed)

    if FLAGS.mode=='train':
        run_training()
    elif FLAGS.mode =='compose':
        run_compose()
    else:
        raise ValueError("unknown mode %s"%FLAGS.mode)


if __name__ == '__main__':
    tf.app.run()