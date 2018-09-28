#!/bin/sh
if [ -f ".tensorboardpid" ]; then
    if [ -d /proc/`cat .tensorboardpid` ]; then
        echo [kill live pid `cat .tensorboardpid`] \
        && kill `cat .tensorboardpid`
    else
        echo [remove dead pid `cat .tensorboardpid`] \
        && rm .tensorboardpid
    fi
fi

nohup tensorboard --port 10086 --logdir=logs/ >logs/tensorboard.out 2>&1 & echo $! > .tensorboardpid \
    && echo [start tensorboard on port 10086]
