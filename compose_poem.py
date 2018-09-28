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
import tensorflow as tf
from poems.model import rnn_model
from poems.poems import process_poems
import numpy as np
import os

tf.app.flags.DEFINE_string('input_name', 'small_poems', 'name of data(.txt)/model dir/model prefix')
tf.app.flags.DEFINE_integer('gen_sequence_len', 500, 'length of gen sequence')
tf.app.flags.DEFINE_string('cuda_visible_devices', '1', '''[Train] visible GPU ''')

FLAGS=tf.app.flags.FLAGS


start_token = 'B'
end_token = 'E'
model_dir = os.path.join('model',FLAGS.input_name)
corpus_path = os.path.join('data', FLAGS.input_name + '.txt')

lr = 0.0002


def to_word(predict, vocabs):
    predict = predict[0]       
    predict /= np.sum(predict)
    sample = np.random.choice(np.arange(len(predict)), p=predict)
    if sample > len(vocabs):
        return vocabs[-1]
    else:
        return vocabs[sample]


def gen_poem():
    batch_size = 1
    print('## loading corpus from %s' % model_dir)
    poems_vector, word_int_map, vocabularies = process_poems(corpus_path)

    input_data = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=lr)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, checkpoint)



        while True:
            x = np.array([list(map(word_int_map.get, start_token))])

            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x})
            begin_word = input('## please input the first character:')
            if begin_word and begin_word in vocabularies:
                word = begin_word
            else:
                print('## begin word not in vocabularies, use random:')
                word = to_word(predict, vocabularies)
            poem_ = ''

            i = 0
            while i < FLAGS.gen_sequence_len:
                poem_ += word
                i += 1
                x = np.zeros((1, 1))
                x[0, 0] = word_int_map[word]
                [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                                 feed_dict={input_data: x, end_points['initial_state']: last_state})
                word = to_word(predict, vocabularies)

            print(poem_,flush=True)





if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_visible_devices  # set GPU visibility in multiple-GPU environment
    gen_poem()
