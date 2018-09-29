#!/bin/sh

# multiple command in one line
if [ -f ".jinyongpid" ]; then
    if [ -d /proc/`cat .jinyongpid` ]; then
        echo found running pid `cat .jinyongpid`
    else
        rm .jinyongpid \
        && echo [remove dead pid `cat .jinyongpid`] \
        && nohup python3 train.py --cuda_visible_devices=0 --input_name=jinyong --rnn_size=256 --num_layers=3 --learning_rate=0.0002 --epochs=1000 >logs/jinyong.out 2>&1 & echo $! > .jinyongpid \
        && echo [train started] \
        && busybox tail -f logs/jinyong.out
    fi
else
    nohup python3 train.py --cuda_visible_devices=0 --input_name=jinyong --rnn_size=256 --num_layers=3 --learning_rate=0.0002 --epochs=1000 >logs/jinyong.out 2>&1 & echo $! > .jinyongpid \
    && echo [train started] \
    && busybox tail -f logs/jinyong.out
fi

