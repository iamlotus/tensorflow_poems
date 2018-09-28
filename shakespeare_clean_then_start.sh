#!/bin/sh

rm -rf logs/shakespeare/ model/shakespeare/
./shakespeare_end_train.sh
./shakespeare_begin_train.sh


