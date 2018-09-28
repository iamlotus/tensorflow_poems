#!/bin/sh

# multiple command in one line
if [ -f ".sptrainpid" ]; then
    if [ -d /proc/`cat .sptrainpid` ]; then
        echo found running pid `cat .sptrainpid`
    else
        rm .sptrainpid \
        && echo [remove dead pid `cat .sptrainpid`] \
        && nohup python3 train.py --cuda_visible_devices=1 --input_name=shakespeare --learning_rate=0.001 --epochs=1000 --input_name=poems >logs/train.out 2>&1 & echo $! > .sptrainpid \
        && echo [train started] \
        && busybox tail -f logs/sptrain.out
    fi
else
    nohup python3 train.py --cuda_visible_devices=1 --input_name=shakespeare --learning_rate=0.001 --epochs=1000 --input_name=poems >logs/train.out 2>&1 & echo $! > .sptrainpid \
    && echo [train started] \
    && busybox tail -f logs/sptrain.out
fi

