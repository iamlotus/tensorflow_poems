#!/bin/sh

./jinyong_end_train.sh && rm -rf logs/jinyong && rm -rf logs/jinyong.out && rm -rf model/jinyong && ./jinyong_begin_train.sh


