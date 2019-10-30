#-*- coding utf-8 -*-

import re
from sys import argv, stdout
from os.path import splitext

with open(argv[1], 'r') as file2read, open(splitext(argv[1])[0]+'_new.txt','w') as file2write:
	for line in file2read.readlines():
		new_line = re.sub(' +', '\t', line)
		file2write.write(new_line)
	stdout.write('Work done\n')