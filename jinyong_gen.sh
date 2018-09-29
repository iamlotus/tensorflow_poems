#!/bin/sh

python3 rnn.py --mode=compose --input_name=jinyong --rnn_size=256 --num_layers=3 --gen_sequence_len=2000 --cuda_visible_devices=3


