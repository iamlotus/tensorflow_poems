#!/bin/sh

if [ -f ".poemspid" ]; then
    if [ -d /proc/`cat .poemspid` ]; then
      echo stop `cat .poemspid` && kill `cat .poemspid` && rm .poemspid
  else
     echo remove dead pid `cat .poemspid` && rm .poemspid
  fi
else
    echo nothing to stop, can not find .poemspid file
fi