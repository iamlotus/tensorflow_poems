# -*- coding: utf-8 -*-
# file: poems.py
# author: JinTian
# time: 08/03/2017 7:39 PM
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
import collections
import numpy as np


def process_poems(file_name):


    # contents -> list of word
    contents = []

    buffer_size=1000
    with open(file_name, "r", encoding='utf-8') as f:
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


    # flatten poem_vec

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
