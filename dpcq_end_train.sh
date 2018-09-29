#!/bin/sh

if [ -f ".dpcqpid" ]; then
    if [ -d /proc/`cat .dpcqpid` ]; then
      echo stop `cat .dpcqpid` && kill `cat .dpcqpid` && rm .dpcqpid
  else
     echo remove dead pid `cat .dpcqpid` && rm .dpcqpid
  fi
else
    echo nothing to stop, can not find .dpcqpid file
fi