#!/bin/sh

# multiple command in one line
if [ -f ".poemspid" ]; then
    if [ -d /proc/`cat .poemspid` ]; then
        echo found running pid `cat .poemspid`
    else
        rm .poemspid \
        && echo [remove dead pid `cat .poemspid`] \
        && nohup python3 train.py --cuda_visible_devices=2 --input_name=poems --treat_corpus_as_byte=False --rnn_size=256 --num_layers=3 --learning_rate=0.0002 --epochs=1000 >logs/poems.out 2>&1 & echo $! > .poemspid \
        && echo [train started] \
        && busybox tail -f logs/poems.out
    fi
else
    nohup python3 train.py --cuda_visible_devices=2 --input_name=poems --treat_corpus_as_byte=False --rnn_size=256 --num_layers=3 --learning_rate=0.0002 --epochs=1000 >logs/poems.out 2>&1 & echo $! > .poemspid \
    && echo [train started] \
    && busybox tail -f logs/poems.out
fi



