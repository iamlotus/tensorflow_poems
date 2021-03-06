# -*- coding: utf-8 -*-
# file: main.py
# author: JinTian
# time: 11/03/2017 9:53 AM
# Copyright 2017 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
import os
import numpy as np
import tensorflow as tf
from poems.model import rnn_model
from poems.poems import process_poems, generate_batch
import time

tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate.')
tf.app.flags.DEFINE_string('input_name', 'small_poems', 'name of data(.txt)/model dir/model prefix')
tf.app.flags.DEFINE_integer('epochs', 50, 'train how many epochs.')
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

    poems_vector, word_to_int, vocabularies = process_poems(corpus_path)
    batches_inputs, batches_outputs = generate_batch(FLAGS.batch_size, poems_vector, word_to_int)

    print("## top ten vocabularies: %s" % str(vocabularies[:10]))
    print("## tail ten vocabularies: %s" % str(vocabularies[-10:]))
    print("## len(first vector)=%d, first vector[:50]: %s"% (len(poems_vector[0]),poems_vector[0][:50]))
    print("## len(last vector)=%d, second vector[:50]: %s" % (len(poems_vector[-1]), poems_vector[-1][:50]))

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

        n_chunk = len(poems_vector) // FLAGS.batch_size

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


def main(_):
    run_training()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_visible_devices  # set GPU visibility in multiple-GPU environment
    tf.app.run()