#!/bin/sh

./poem_end_train.sh && rm -rf logs/poems && rm -rf logs/poems.out && rm -rf model/poems && ./poem_begin_train.sh


