#!/bin/sh

if [ -f ".shakespearepid" ]; then
    if [ -d /proc/`cat .shakespearepid` ]; then
      echo stop `cat .shakespearepid` && kill `cat .shakespearepid` && rm .shakespearepid
  else
     echo remove dead pid `cat .shakespearepid` && rm .shakespearepid
  fi
else
    echo nothing to stop, can not find .shakespearepid file
fi