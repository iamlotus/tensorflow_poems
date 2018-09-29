import tensorflow as tf
import rnn
import os
import pickle
import numpy as np
import sys


tf.app.flags.DEFINE_string('input_name', 'small_poems', 'name of data(.txt)/model dir/model prefix')
tf.app.flags.DEFINE_string('epoch', None, 'checkpoint epoch, None mean use up-to-date checkpoint')
tf.app.flags.DEFINE_integer('seq_len', 500, 'length of gen sequence')
tf.app.flags.DEFINE_string('cuda_visible_devices', '2', '''[Train] visible GPU ''')

FLAGS=tf.app.flags.FLAGS

model_dir=os.path.join('model',FLAGS.input_name)
corpus_path=os.path.join('data', FLAGS.input_name + ".txt")


def to_word(predict, vocabs):
    predict = predict[0]
    predict /= np.sum(predict)
    sample = np.random.choice(np.arange(len(predict)), p=predict)

    return vocabs[sample]


def run_compose():
    batch_size = 1
    input_data = tf.placeholder(tf.int32, [batch_size, None])

    checkpoint = tf.train.latest_checkpoint(model_dir) if FLAGS.epoch is None else os.path.join(model_dir,"%s-%d"%(FLAGS.input_name,FLAGS.epoch))
    param_path=os.path.join(model_dir,"%s"%FLAGS.input_name+".param")
    
    if not os.path.isfile(param_path):
        raise ValueError('can not find "%s"'%param_path)
    
    with open(param_path,"rb") as f:
        param=pickle.load(f)

    # set random seed before process_corpus
    np.random.seed(param["random_seed"])
    
    _, word_int_map, vocabularies = rnn.process_corpus(corpus_path,param["treat_corpus_as_byte"])
    rnn.vocabulary_summary(vocabularies)
    
    end_points = rnn.rnn_model(cell_type=param["cell_type"], input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=param["rnn_size"], num_layers=param["num_layers"], batch_size=batch_size, learning_rate=1)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
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
            while i < FLAGS.seq_len:
                output.append(word)
                i += 1
                x = np.zeros((1, 1))
                x[0, 0] = word_int_map[word]
                [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                                 feed_dict={input_data: x, end_points['initial_state']: last_state})
                word = to_word(predict, vocabularies)

            if param["treat_corpus_as_byte"]:
                sys.stdout.buffer.write(bytes(output))
                sys.stdout.buffer.write(b'\n')
                sys.stdout.buffer.write(b'\n')
                sys.stdout.flush()
            else:
                print("".join(output),end='\n',flush=True)


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_visible_devices  # set GPU visibility in multiple-GPU environment
    rnn.print_args(FLAGS)
    run_compose()


if __name__ == '__main__':
    tf.app.run()