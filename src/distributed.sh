#!/bin/bash

#directories
SCRDIR="$HOME/scratch_"`date +%Y-%m-%d_%H-%M-%S`
PYLIBS=$HOME/pylibs
SCRIPTS=$HOME/tensorflow
NODEFILE=$SCRIPTS/nodefile.txt

#net settings
NET=$SCRIPTS/squeezenet.py
DATA=/tmp/caltech256
SERVER=distributed
SYNC=none
PROFILE="--profile"
ACCURACY= #"--accuracy"

#python and ssh settings
#PYTHON=python3
PYTHON="LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH:$HOME/my_libc_env/lib/x86_64-linux-gnu/:$HOME/my_libc_env/usr/lib64/\" $HOME/my_libc_env/lib/x86_64-linux-gnu/ld-2.17.so `which python3.6`"
LOGIN=xslamp01
CERT=$HOME/.ssh/id_rsa

#make scratch directory
rm -rf $SCRDIR
mkdir -p $SCRDIR

# change to scratch directory, exit on failure
cd $SCRDIR || exit

ITER=0
while read LINE
do
    LOGDIR=$SCRDIR/"node_"$ITER
    mkdir -p $LOGDIR
    STDOUT=$LOGDIR/stdout.txt
    STDERR=$LOGDIR/stderr.txt
     
    ssh -i $CERT $LOGIN@$LINE "$PYTHON $SCRIPTS/dist.py --net=$NET --logdir=$LOGDIR --datadir=$DATA --nodefile=$NODEFILE --server=$SERVER --index=$ITER --sync=$SYNC $PROFILE $ACCURACY >$STDOUT 2>$STDERR" &
    ITER=$(($ITER+1))
done < $NODEFILE

#exit
exit
