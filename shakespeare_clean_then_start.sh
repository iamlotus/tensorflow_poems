#!/bin/sh

./shakespeare_end_train.sh && rm -rf logs/shakespeare && rm -rf logs/shakespeare.out && rm -rf model/shakespeare && ./shakespeare_begin_train.sh


