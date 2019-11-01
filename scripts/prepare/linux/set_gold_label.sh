#!/usr/bin/bash

sed -E "s/ +/\t/g" $1 | grep -v '^#' | rev | cut -f2- | rev | cut -f 7,12- > test_label.txt

