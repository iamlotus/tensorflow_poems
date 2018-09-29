#!/bin/sh

if [ -f ".jinyongpid" ]; then
    if [ -d /proc/`cat .jinyongpid` ]; then
      echo stop `cat .jinyongpid` && kill `cat .jinyongpid` && rm .jinyongpid
  else
     echo remove dead pid `cat .jinyongpid` && rm .jinyongpid
  fi
else
    echo nothing to stop, can not find .jinyongpid file
fi