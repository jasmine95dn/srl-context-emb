#!/bin/bash

filename=$1

python rewrite.py $1 ;

sent='_sent'
substr='.txt'
new_add='_new.txt'

inp_=${1%$substr}$new_add

#echo $inp_

new_=${1%$substr}$sent$substr
#echo $new_

grep -Ev "^#" $inp_ | cut -f 4  > $new_

python make_sents.py $new_

