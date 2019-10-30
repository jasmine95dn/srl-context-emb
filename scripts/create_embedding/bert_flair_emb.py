from flair.data import Sentence
from flair.embeddings import BertEmbeddings, FlairEmbeddings, StackedEmbeddings
from os.path import splitext
from h5py import File, special_dtype
from json import dumps
from sys import stdout
from time import time, ctime
import numpy as np
import argparse
import logging

def arguments():
	parser = argparse.ArgumentParser(description='''A wrap-up data file to transform sentences in a text file 
					into its corresponding BERT/Flair embeddings''')
	parser.add_argument('-f','--filename', type=str, help='text file, each sentence per line is recommended')
	parser.add_argument('-t','--type', type=str, default='bert', help='type of embeddings, BERT/Flair, default=bert')
	parser.add_argument('-s','--stacked', action='store_true', help='''call this flag if the output embeddings are a combination of other embeddings, 
					temporarily this will be the combination of BERT and Flair embeddings''')
	parser.add_argument('-m','--model', type=str, default='bert-large-uncased-whole-word-masking', help='model of BERT embeddings as output (default: bert-large-uncased-whole-word-masking)')
	parser.add_argument('-n','--n_layers', type=int, default=3, help='''number of last layers of BERT embeddings (default: last 3 layers)''')
	args = parser.parse_args()
	return args


# create object of embedding type for later use
def out_embedding(type_, model, n_layers, stacked=False):
	'''
	Create object of embedding type for later purpose
	:param:
		:type_: (str) type of embedding (currently there are only BERT or Flair embeddings)
		:model: (str) pretrained model of BERT embedding
		:n_layers: (int) number of last layers of trained BERT embeddings to be chosen
		:stacked: (bool) if this embedding is a combination of more embeddings (True/False)
	:return:
		:embedding: (BertEmbeddings / StackedEmbeddings) embedding object
	'''
	out_layers = ','.join([str(-i) for i in range(1,n_layers+1)])
	if not stacked:
		if type_.lower() == 'bert':
			embedding = BertEmbeddings(bert_model_or_path=model, layers=out_layers)
			return embedding
		else:
			emb = WordEmbeddings('glove')
	else:
		emb = BertEmbeddings(bert_model_or_path=model, layers=out_layers)

	flair_forward = FlairEmbeddings('news-forward-fast')
	flair_backward = FlairEmbeddings('news-backward-fast')
	embedding = StackedEmbeddings(embeddings=[flair_forward, flair_backward, emb])

	return embedding
		

def embed_file(filename, embfile, embedding):
	'''
	Write trained embedding into *.hdf5 file
	:param:
		:filename:(str) text file as input to train embedding
		:embfile: (str) *.hdf5 file to save results
		:embedding: (BertEmbeddings / StackedEmbeddings) embedding object
	'''
	sentence_to_index = {}
	with open(filename, 'r') as file2read, File(embfile,'w') as file2write:
		for idx, line in enumerate(file2read.readlines()):
			sentence = Sentence(line.strip())
			sentence_to_index[line.strip()] = str(idx)

			embedding.embed(sentence)

			output = []

			for token in sentence:
				# in case in some servers, trained embeddings for all tokens are not saved in CPU but GPU
				try:
					embedding_ = np.array(token.embedding)
				except TypeError:
					embedding_ = token.embedding.cpu().numpy()

				# this is used to define the case that each type of embeddings fed into this stacked embedding has the dimension of (1024,) 
				# Flair embeddings that stack with GloVe Embedding (100,) must be reimplemented, 
				# as well as other type of embeddings from Flair and BERT that don't have dimension of (1024,) 
				if len(embedding_) // 1024 > 3:
					part1 = embedding_[:2048] 
					part2 = np.average(embedding_[2048:].reshape((len(embedding_[2048:])//1024,1024)),axis=0)
					embedding_ = np.concatenate((part1,part2))

				output.append(embedding_.reshape((3,1024)))
			output = np.array(output).swapaxes(0,1)
			file2write.create_dataset(str(idx), output.shape, dtype='float32', data=output)

		file2write.create_dataset('sentence_to_index', (1,), dtype=special_dtype(vlen=str), data=dumps(sentence_to_index))

def main():
	args = arguments()

	filename = args.filename
	if not args.stacked:
		stdout.write(ctime(time()) + ' - Proceed %s Embedding....\n'%args.type.lower().title())
		embfile = splitext(filename)[0] + '_%s.hdf5'%args.type.lower()
	else: 
		stdout.write(ctime(time()) + ' - Proceed Stacked Embedding....\n')
		embfile = splitext(filename)[0] + '_stacked.hdf5'

	embedding = out_embedding(args.type, args.model, args.n_layers, args.stacked)

	embed_file(filename, embfile, embedding)
	stdout.write(ctime(time()) + ' - Work done \n Check file %s\n'%embfile)

if __name__ == '__main__':
	main()
	



