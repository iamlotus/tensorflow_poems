#!/bin/sh

if [ -f ".sptrainpid" ]; then
    if [ -d /proc/`cat .sptrainpid` ]; then
      echo stop `cat .sptrainpid` && kill `cat .sptrainpid` && rm .sptrainpid
  else
     echo remove dead pid `cat .sptrainpid` && rm .sptrainpid
  fi
else
    echo nothing to stop, can not find .sptrainpid file
fi