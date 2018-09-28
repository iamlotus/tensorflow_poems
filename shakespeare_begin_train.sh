#!/bin/sh

# multiple command in one line
if [ -f ".shakespearepid" ]; then
    if [ -d /proc/`cat .shakespearepid` ]; then
        echo found running pid `cat .shakespearepid`
    else
        rm .shakespearepid \
        && echo [remove dead pid `cat .shakespearepid`] \
        && nohup python3 rnn.py --cuda_visible_devices=0 --mode=train --input_name=shakespeare --learning_rate=0.001 --epochs=1000 >logs/shakespeare.out 2>&1 & echo $! > .shakespearepid \
        && echo [train started] \
        && busybox tail -f logs/shakespeare.out
    fi
else
    nohup python3 rnn.py --cuda_visible_devices=0 --mode=train --input_name=shakespeare --learning_rate=0.001 --epochs=1000 >logs/shakespeare.out 2>&1 & echo $! > .shakespearepid \
    && echo [train started] \
    && busybox tail -f logs/shakespeare.out
fi

