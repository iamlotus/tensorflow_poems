import os
import tensorflow as tf
import time
import pickle
import rnn
import numpy as np

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
tf.app.flags.DEFINE_string('cuda_visible_devices', '2', '''[Train] visible GPU ''')


FLAGS = tf.app.flags.FLAGS

model_dir=os.path.join('model',FLAGS.input_name)
log_dir=os.path.join('logs',FLAGS.input_name)
model_file=os.path.join(model_dir,FLAGS.input_name)
corpus_path=os.path.join('data', FLAGS.input_name + ".txt")


def run_training():
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # params to rebuild training graph
    compose_params = {
        "cell_type": FLAGS.cell_type,
        "rnn_size": FLAGS.rnn_size,
        "num_layers": FLAGS.num_layers,
        "batch_size": FLAGS.batch_size,
        "random_seed": FLAGS.random_seed,
        "treat_corpus_as_byte": FLAGS.treat_corpus_as_byte,
        "validate_ratio":FLAGS.validate_ratio
    }

    # save param that compose can rebuild graph
    with open(os.path.join(model_dir, "%s.param" % FLAGS.input_name), "wb") as f:
        pickle.dump(compose_params, f)

    # set random seed before process_corpus
    np.random.seed(FLAGS.random_seed)

    poems_vector, word_to_int, vocabularies = rnn.process_corpus(corpus_path,FLAGS.treat_corpus_as_byte)
    data_provider=rnn.DataProvider(poems_vector, FLAGS.batch_size, FLAGS.train_sequence_len,FLAGS.validate_ratio)
    rnn.vocabulary_summary(vocabularies)
    rnn.corpus_summary(data_provider)

    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

    end_points = rnn.rnn_model(cell_type=FLAGS.cell_type, input_data=input_data, output_data=output_targets, vocab_size=len(
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


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_visible_devices  # set GPU visibility in multiple-GPU environment
    rnn.print_args(FLAGS)
    run_training()

if __name__ == '__main__':
    tf.app.run()
