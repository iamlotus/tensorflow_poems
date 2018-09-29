#!/bin/sh

./dpcq_end_train.sh && rm -rf logs/dpcq && rm -rf logs/dpcq.out && rm -rf model/dpcq && ./dpcq_begin_train.sh


