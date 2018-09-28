import os
import tensorflow as tf
import time
import operator
import numpy as np
import collections
import sys

tf.app.flags.DEFINE_string('mode', 'train', 'train or compose')
tf.app.flags.DEFINE_bool('treat_corpus_as_byte', True, 'treat corpus as byte or word, default(True) will treat input'
                                                       ' as byte stream, set to False will treat input as text(with '
                                                       'utf encode)')
tf.app.flags.DEFINE_integer('rnn_size', 128, 'rnn size.')
tf.app.flags.DEFINE_integer('num_layers', 2, 'layer num.')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate.')
tf.app.flags.DEFINE_string('input_name', 'small_poems', 'name of data(.txt)/model dir/model prefix')
tf.app.flags.DEFINE_integer('epochs', 50, 'train how many epochs.')
tf.app.flags.DEFINE_integer('train_sequence_len', 50, 'length of train sequence')
tf.app.flags.DEFINE_integer('print_every_steps', 100, '''save model every steps''')
tf.app.flags.DEFINE_integer('save_every_epoch', 1, '''save model every epoch''')
tf.app.flags.DEFINE_integer('gen_sequence_len', 100, 'length of gen sequence')
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

    print("content size = %d , %d words totally"%(len(contents_vector),len(word_int_map)))
    return contents_vector, word_int_map, words


def generate_batch(content_vector, batch_size,seq_len):
    x=np.copy(content_vector)
    y=np.zeros(x.shape,dtype=x.dtype)
    y[:-1]=x[1:]
    y[-1]=x[0]

    n_chunk = len(content_vector) // (batch_size*seq_len)
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
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
    return x_batches, y_batches


def rnn_model(model, input_data, output_data, vocab_size, rnn_size=128, num_layers=2, batch_size=64,
              learning_rate=0.01):
    """
    construct rnn seq2seq model.
    :param model: model class
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

    if model=='rnn':
        cell_fun=tf.nn.rnn_cell.BasicRNNCell
    elif model == 'gru':
        cell_fun=tf.nn.rnn_cell.GRUCell
    elif model == 'lstm':
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

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
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
    batches_inputs, batches_outputs = generate_batch(poems_vector,FLAGS.batch_size, FLAGS.train_sequence_len)

    print("## top ten vocabularies: %s" % str(vocabularies[:10]))
    print("## tail ten vocabularies: %s" % str(vocabularies[-10:]))
    print("## poems_vector[:10]: %s" % poems_vector[:10])
    print("## poems_vector[-10:]: %s" % poems_vector[-10:])

    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    summary_op=tf.summary.merge_all()
    with tf.Session() as sess:

        train_writer = tf.summary.FileWriter(os.path.join(log_dir, "train"), sess.graph)

        sess.run(init_op)

        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(model_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("## restore from the checkpoint {0}".format(checkpoint),flush=True)
            start_epoch += int(checkpoint.split('-')[-1])+1
        print('## start training...',flush=True)

        n_chunk = len(batches_inputs)

        try:
            for epoch in range(start_epoch, FLAGS.epochs):
                n = 0

                for batch in range(n_chunk):
                    step = epoch * n_chunk+batch
                    if step % FLAGS.print_every_steps==0:
                        loss, _, _,train_summary = sess.run([
                            end_points['total_loss'],
                            end_points['last_state'],
                            end_points['train_op'],
                            summary_op
                        ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                        train_writer.add_summary(train_summary,global_step=step)
                        print('[%s] Step: %d, Epoch: %d, batch: %d, training loss: %.6f' % (time.strftime('%Y-%m-%d %H:%M:%S'),step,epoch, batch, loss), flush=True)
                    else:
                         _, _ = sess.run([
                            end_points['last_state'],
                            end_points['train_op']
                        ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                    n += 1
                    step += 1
                if epoch % FLAGS.save_every_epoch == 0:
                    saver.save(sess, model_file, global_step=epoch)
                    print("[%s] Saving checkpoint for epoch %d"%(time.strftime('%Y-%m-%d %H:%M:%S'), epoch),flush=True)
        except KeyboardInterrupt:
            print('## Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, model_file, global_step=epoch)
            print('## Last epoch were saved, next time will start from epoch {}.'.format(epoch), flush=True)


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

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=FLAGS.learning_rate)

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
                sys.stdout.flush()

                # print("="*100)
                # print(output,end='\n',flush=True)

            else:
                print("".join(output),end='\n',flush=True)






def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_visible_devices  # set GPU visibility in multiple-GPU environment
    print_args()
    if FLAGS.mode=='train':
        run_training()
    elif FLAGS.mode =='compose':
        run_compose()
    else:
        raise ValueError("unknown mode %s"%FLAGS.mode)


if __name__ == '__main__':
    tf.app.run()