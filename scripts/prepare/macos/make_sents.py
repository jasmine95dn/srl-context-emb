# -*- coding utf8 -*-

from sys import argv, stdout
from os.path import splitext

filename = argv[1]

with open(filename, 'r') as file2read, open(splitext(filename)[0]+'_new.txt', 'w') as file2write:
	sent = []

	for line in file2read.readlines():
		word = line.strip()
		if not word == '': 
			sent.append(word)
		else:
			file2write.write(' '.join(sent)+'\n')
			sent = []
	stdout.write('Work done.\n')
