#!/bin/sh

# multiple command in one line
if [ -f ".dpcqpid" ]; then
    if [ -d /proc/`cat .dpcqpid` ]; then
        echo found running pid `cat .dpcqpid`
    else
        rm .dpcqpid \
        && echo [remove dead pid `cat .dpcqpid`] \
        && nohup python3 train.py --cuda_visible_devices=0 --input_name=dpcq --rnn_size=256 --num_layers=3 --learning_rate=0.0002 --epochs=1000 >logs/dpcq.out 2>&1 & echo $! > .dpcqpid \
        && echo [train started] \
        && busybox tail -f logs/dpcq.out
    fi
else
    nohup python3 train.py --cuda_visible_devices=0 --input_name=dpcq --rnn_size=256 --num_layers=3 --learning_rate=0.0002 --epochs=1000 >logs/dpcq.out 2>&1 & echo $! > .dpcqpid \
    && echo [train started] \
    && busybox tail -f logs/dpcq.out
fi

