#!/bin/sh

# multiple command in one line
if [ -f ".trainpid" ]; then
    if [ -d /proc/`cat .trainpid` ]; then
        echo found running pid `cat .trainpid`
    else
        rm .trainpid \
        && echo [remove dead pid `cat .trainpid`] \
        && nohup python3 train.py --learning_rate=0.001 --epochs=1000 >logs/train.out 2>&1 & echo $! > .trainpid \
        && echo [train started] \
        && busybox tail -f logs/train.out
    fi
else
    nohup python3 train.py --learning_rate=0.001 --epochs=1000 >logs/train.out 2>&1 & echo $! > .trainpid \
    && echo [train started] \
    && busybox tail -f logs/train.out
fi



