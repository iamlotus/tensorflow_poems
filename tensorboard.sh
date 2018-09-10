#!/bin/sh
if [ -f ".tensorboardpid" ]; then
    if [ -d /proc/`cat .tensorboardpid` ]; then
        echo found running tensorboard pid `cat .tensorboardpid`
    else
        echo [remove dead pid `cat .tensorboard`] \
        && rm .tensorboard \
        && nohup tensorboard --port 10087 --logdir=logs/ >logs/tensorboard.out 2>&1 & echo $! > .tensorboardpid \
        && echo [tensorboard started] \
        && busybox tail -f logs/tensorboard.out
    fi
else
    nohup tensorboard --port 10087 --logdir=logs/ >logs/tensorboard.out 2>&1 & echo $! > .tensorboardpid \
    && echo [tensorboard started] \
    && busybox tail -f logs/train.out
fi