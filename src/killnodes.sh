#!/bin/bash

SCRIPTS=$HOME/tensorflow
NODEFILE=$SCRIPTS/nodefile.txt
CERT=$HOME/.ssh/id_rsa
LOGIN=xslamp01

while read LINE
do
    ssh -i $CERT $LOGIN@$LINE "pkill ld-2.17.so" &
done < $NODEFILE

#exit
exit
