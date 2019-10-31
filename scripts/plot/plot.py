import sys
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser
from sklearn.decomposition import IncrementalPCA, PCA, KernelPCA
import eval
from ordered_set import OrderedSet
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import pandas as pd
import logging
import pickle
from time import time, ctime


def arguments():
	parser = ArgumentParser(description='Visualize some results')
	parser.add_argument('file', nargs='+', help='file to plot')
	parser.add_argument('-l', '--label', help='label file in each output folder with their corresponding indices')
	
	parser.add_argument('-cl', '--choose_label', nargs='+', help='''space-separated ; if this is specified, than only plot those chosen labels, 
																check file label_ids.txt for right name of labels,
																this can be used for scatter plot and confusion matrix''')
	parser.add_argument('-rs', '--size', type=int, nargs=2, default=[10,10], help='new size of figure instead of default' )
	
	# label embeddings
	parser.add_argument('-v', '--vector', action='store_true', help='call this flag to visualize learnt label embeddings in scatterplot')
	parser.add_argument('--icr', action='store_true', help='call this flag for IncrementalPCA instead of normal one, default is PCA')
	parser.add_argument('--kernel', action='store_true', help='call this flag for KernelPCA')

	# confusion matrix
	parser.add_argument('-m', '--matrix', action='store_true', help='call this flag to visualize confusion matrix for labeling errors of this model. Each cell shows the percentage of predicted labels for each gold label.')
	parser.add_argument('--error', action='store_true', help='call this flag to choose what kind of labeling results to be plotted (errors/correctness)')
	
	# line progress
	parser.add_argument('-p', '--plot', action='store_true', help='call this flag to visualize the progress of training process on validating/development data sets based on their reported F1 scores, call flag --valid or --train to see progress on training/development data sets')
	parser.add_argument('--valid', action='store_true', help='progress on development/validating set')
	parser.add_argument('--train', action='store_true', help='progress on training set')
	
	parser.add_argument('-e', '--embeddings', default='unknown', nargs='+', help='types of embeddings, should correspond to the given input files')
	parser.add_argument('-s', '--save', action='store_true', help='whether to save this figure')
	parser.add_argument('--output_dir', help='where to save figure')

	args = parser.parse_args()
	return args

def load_labels(file):
	'''
	Load all labels from texts
	:param:
		:file: (str) input should be the output label file 'label_ids*.txt' from training process of Ouchi's span-based model
	:param:
		:labels:(list) list of all labels from file
	'''
	with open(file) as file_labels:
		labels = [line.rstrip().split()[0] for line in file_labels]
	return labels


# load each type of inputs for each visualization
def load_vectors(file):
	'''
	Load learnt label embeddings 
	:param:
		:file: (str) input is a 'param.*.pkl.gz' file that is included as output of each trained model
	:return: a matrix in which each element represents vector of a label
	'''
	assert type(file) == str
	assert file.endswith('pkl.gz')
	import gzip as gz
	with gz.open(file, 'rb') as f:
		p = pickle.load(f)
	return p[-2].T

def load_matrix(files):
	'''
	Load datas for confusion matrix, 
	:param:
		:files: (list of str) inputs are 2 files, first with true labels, second with predicted labels 
	:return: 2 lists 
		:sents1: (list) one for loaded gold labels 
		:sents2: (list) one for predicted labels
	'''
	assert type(files) == list
	assert len(files) == 2
	assert all([type(f) == str for f in files])

	sents1 = [list(zip(*sent)) for sent in eval.load(files[0])]
	sents2 = [list(zip(*sent)) for sent in eval.load(files[1])]
	return sents1, sents2

def load_progress(file, train=True, valid=True):
	''' 
	load datas for visualization of training/validation process through epoch
	:param:
		:file: (str) input is a file report that contains the progress of training
		:train: (bool) this report has progress on training datas 
		:valid: (bool) this report has progress on development datas
	:return: (np.array) a object that saved result of each epoch as an element in an 1-D array
	'''
	import re
	import random
	assert type(file) == str

	if train and valid:
		sys.stderr.write('Only choose one so as not to be messed up\n')
		sys.exit(1)

	regex = r'[\d]*\.?[\d]+'
	prog = [None] * 101
	
	with open(file) as f:
		# this file starts with line F1 HISTORY, 
		# should contain only epochs and f-scores, 
		# that may not be continuously recorded
		if valid:
			old_ep = 1
			for line in f.readlines()[1:]:
				#print(re.findall(regex, line))
				ep, score = re.findall(regex, line)
				ep = int(ep)
				score = float(score)
				if ep - old_ep > 1:
					for idx in range(old_ep+1, ep):
						# in normal output from Ouchi's model, 
						# only F1 score on validating set that is higher than the former one will be saved
						# so the epochs in between will have the score randomly in the range between this score and its previous one
						#prog[idx] = round(random.uniform(prog[old_ep], score))

						# instead of randomly choose between ranges, 
						# epochs that has no reported F1 Score will be
						# the same one with the latest epoch (not the actual one)
						prog[idx] = prog[old_ep]
				prog[ep] = score
				old_ep = ep

			# if last epoch in report with higher F-Score is not epoch 100
			if old_ep < len(prog) - 1:
				for idx in range(old_ep+1, len(prog)):
					#prog[idx] = round(random.uniform(prog[old_ep]-1.00, prog[old_ep]))
					prog[idx] = prog[old_ep]

		# this file has other format, 
		# just read the normal report file that was extracted directly from stdout
		# there is hard coding here as this is set to run with the exact output of Ouchi's model
		elif train:
			lines = f.readlines()
			for i, line in enumerate(lines):
				if line.startsswith('TRAIN'):
					ep = int(re.search(regex,lines[i-1]).group())
					score = float(re.search(regex, lines[i+4]).group())
					prog[ep] = score
	return np.array(prog[1:])

# dimension reduction
def reduce_dim(vectors, args):
	'''
	Dimension reduction
	:param:
		:vectors: (np.array/np.matrix/np.ndarray) 2-dimensional matrix mxn in which each element is a vector
	:return: 2-dimensional matrix mx2 of the original one
	'''
	if args.icr:
		model = IncrementalPCA(n_components=2)
	elif args.kernel:
		model = KernelPCA(n_components=2)
	else:
		model = PCA(n_components=2)
	return model.fit_transform(vectors)

def scatterplot(vectors, labels, size, emb='', save=False, outdir='./'):
	'''
	Scatter plot for learnt label embeddings
	:param:
		:vectors: (np.array) data for scatter plot
		:labels: (list) list of loaded labels
		:size: (tuple) resize this figure
		:emb: (str) type of embedding
		:save: (bool) whether to save this figure
		:outdir: (str) where to save output
	'''
	import random as rd

	# set colors for each label
	cm = [*list('abcdef'), *[str(i) for i in range(10)]]
	colors = ['#%s'%(''.join([rd.choice(cm) for i in range(6)])) for x in range(len(labels))]

	# set figure size
	fig = plt.figure(figsize=size, dpi=80)

	# scatter
	ax = fig.add_subplot()

	# set scatter for each point
	for v, label, color in zip(vectors, labels, colors):
		ax.scatter(v[0], v[1], color=color, alpha=0.5, label=label)
		ax.annotate(xy=(v[0]+0.001, v[1]+0.001), s=label, weight='bold')

	ax.legend(loc='best')
	ax.set_title('%s embedding'%emb, weight='bold')
	plt.axis('on')
	plt.tight_layout()
	name = None
	if save:
		name = '%slabel_distribution_%s.png'%(outdir,emb)
		fig.savefig(name, dpi='figure')
	else:
		plt.show()
	plt.close()
	print('Done plotting label embeddings for {0} at {1}'.format(emb, ctime(time())))
	if name:
		sys.stdout.write('Check %s\n'%name)


# helper function
def respan(spans):
	'''
	Reorganize the spans and sort them into 3 lists: 
	one for labels, one for starts of a span, one for ends of a span
	:param:
		:spans: (list) contain the argument spans of a sentence, each span is in form [label, start, end]
	:return: 
		:new_spans: (list) contain only 3 elements as 3 lists/sets in this order : [labels, starts, ends]
	'''
	new_spans = [[], OrderedSet(), OrderedSet()]
	for span in spans:
		starts = [s for i,s in enumerate(span[1:]) if i%2 == 0]
		ends = [s for i,s in enumerate(span[1:]) if i%2 != 0]

		assert len(starts) == len(ends)

		labels = span[:1] * len(starts)
		new_spans[0] += labels
		new_spans[1].update(starts)
		new_spans[2].update(ends)
	return new_spans

def create_data(y_true, y_pred):
	'''
	Create data of true and predicted labels for confusion matrix
	:param:
		:y_true: (list) true labels with spans
		:y_pred: (list) predicted labels with spans
	:return: 2 flattened lists
		:true: (list) gold labels
		:predict: (list) corresponding predicted labels
	'''
	true, predict = [], []

	assert len(y_true) == len(y_pred)
	for y_true_i, y_pred_i in zip(y_true, y_pred):
		assert len(y_true_i) == len(y_pred_i)
		for y_true_j, y_pred_j in zip(y_true_i[1:], y_pred_i[1:]):
			assert len(y_true_j) == len(y_pred_j)
			y_true_spans = respan(eval.get_labeled_spans(y_true_j))
			y_pred_spans = respan(eval.get_labeled_spans(y_pred_j))
			
			labels_true, starts_true, ends_true = y_true_spans
			labels_pred, starts_pred, ends_pred = y_pred_spans

			# if starts of the predicted and true overlap, check if they represent the same labels
			overlap_start = starts_pred & starts_true
			for s in overlap_start:
				idx_true = starts_true.index(s)
				idx_pred = starts_pred.index(s)
				true.append(labels_true[idx_true])
				predict.append(labels_pred[idx_pred])

			# if ends of predicted and starts of true overlap, this means label of true is mistaken with other label
			start_true_diff = starts_true - starts_pred
			if len(start_true_diff) != 0:
				overlap_end_start = start_true_diff & ends_pred
				for s in overlap_end_start:
					idx_true = starts_true.index(s)
					idx_pred = ends_pred.index(s)
					true.append(labels_true[idx_true])
					predict.append(labels_pred[idx_pred])

			# if starts of predicted and ends of true overlap, this means label of true is mistaken with other label
			start_pred_diff = starts_pred - starts_true
			if len(start_pred_diff) != 0:
				overlap_start_end = start_pred_diff & ends_true
				for s in overlap_start_end:
					idx_true = ends_true.index(s)
					idx_pred = starts_pred.index(s)
					true.append(labels_true[idx_true])
					predict.append(labels_pred[idx_pred])

	return true, predict

def cf_matrix(y_true, y_predict, labels, error=True):
	'''
	Recalculate the confusion matrix for labeling error, 
	right predicted error is set to 0, 
	the others are calculated based on percentages of the given numbers of gold labels 
	except for the matched predicted labels
	:param:
		:y_true: (list) true labels
		:y_predict: (list) predicted labels
		:labels: list of given labels/classes

	:return: (np.array) new confusion matrix (transposed), with gold labels on x axis, predicted labels on y axis 
				(normal confusion matrices are reversed)
	'''
	matrix = confusion_matrix(y_true, y_predict, labels)
	label_sum = np.sum(matrix, axis=1)
	
	if error:
		for i in range(len(label_sum)):
			label_sum[i] -= matrix[i, i]
			matrix[i,i] = 0

	for i, (vec, lsum) in enumerate(zip(matrix, label_sum)):
		if lsum != 0:
			matrix[i] = np.around(vec * 100 / lsum)

	return matrix.T

def cf_matrix_plot(matrix, labels, size, emb='', save=False, outdir='./', err=True):
	'''
	Confusion matrix plot for labeling error
	:param:
		:matrix: (np.array) datas for confusion matrix (in form of 2-dimensional list mxm)
		:labels: (list) list of labels
		:size: (tuple) resize this figure
		:emb: (str) type of embedding
		:save: (bool) whether to save this figure
		:outdir: (str) where to save output
	'''

	# set x axis on top
	plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
	plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True


	# set figure size
	plt.figure(figsize=size)

	# plot confusion matrix
	plt.imshow(matrix, interpolation='nearest', cmap=plt.get_cmap('plasma'), alpha=0.9)

	# add label to x and y axis
	tick_marks = np.arange(len(labels))
	plt.xticks(tick_marks, labels)
	plt.yticks(tick_marks, labels)

	# add value into each cell
	for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
		plt.text(j, i, '%d'%matrix[i, j], horizontalalignment='center', color='black', size=10)

	plt.tight_layout()
	plt.suptitle('gold /\npredict', x=0.02, y=0.9, horizontalalignment='left', size=9, weight='bold', verticalalignment='bottom')
	plt.xlabel('%s embedding'%emb, weight='bold')

	name = None
	if save:
		error = '_error' if err else ''
		name = '%slabel_confusion_matrix_%s%s.png'%(outdir, emb, error)
		plt.savefig(name, dpi='figure')
	else:
		plt.show()
	plt.close()
	print('Done plotting label embeddings for {0} at {1}'.format(emb, ctime(time())))
	if name:
		sys.stdout.write('Check %s\n'%name)

# helper function
def check_prime(num):
	for i in range(2, num//2):
		if (num%i) == 0:
			return False
	return True

def line(datas, size, epochs=range(1,101), dataset='', save=False, outdir='./'):
	'''
	Line plot for progress of 4 types of embeddings on datas 
	(from the results I got now, I prefered that on development data sets)
	with help from : https://python-graph-gallery.com/125-small-multiples-for-line-chart/
	:param:
		:datas: (dict) should be a dictionary in which key is name of embedding 
				and value is the progress in numpy data
		:size: (tuple) resize this figure
		:epochs: (range/list) labels for x axis
		:dataset: (str) type of data set (train/development...)
		:save: (bool) whether to save this figure
		:outdir: (str) where to save output

	++++ Note: this is specifically set for plot of 2x2 (4 types of embeddings), 
	some hyper- und parameters should be adapted again for later use of this for another dimension, 
	lines that need changes will be marked at the end with ***
	'''

	assert len(datas) == 4

	# make data frame
	df = pd.DataFrame({'x': epochs, **datas})

	# initialize the figure
	plt.style.use('seaborn')

	# create a color palette
	palette = plt.get_cmap('Set1')

	## set subplots
	fig, ax = plt.subplots(2,2, sharex='col', sharey='row', figsize=size, dpi=80) # ***

	subplots = [(i,j) for i in range(2) for j in range(2)] # ***

	## multiple line plot
	for data, (i, j), num in zip(df.drop('x', axis=1), subplots, range(1, len(datas)+1)):

		## plot every groups but discrete
		for v in df.drop('x', axis=1):
			ax[i, j].plot(df['x'], df[v], marker='', color='grey', linewidth=0.6, alpha=0.3)

		## plot the line plot
		ax[i, j].plot(df['x'], df[data], marker='', color=palette(num), linewidth=2.4, alpha=0.9, label=data)

		## Same limits for everybody
		ax[i, j].set_xlim(0,101)
		ax[i, j].set_ylim(60,100)

		# Add title
		ax[i, j].set_title(data, loc='center', fontsize=15, fontweight=2, color=palette(num))

	## add title
	fig.suptitle('How well is the model trained on development set with each embedding?', horizontalalignment='center', x=0.5, y=0.95, fontsize=20, fontweight=5, color='black')#, y=1.02)

	## set common labels
	fig.text(0.5, 0.04, 'Epoch', ha='center', va='center',fontsize=15, fontweight=3)
	fig.text(0.06, 0.5, 'F1 Score', ha='center', va='center', rotation='vertical', fontsize=15, fontweight=3)

	name = None
	if save:
		name = '%sprogress_%s_data.png'%(outdir, dataset)
		plt.savefig(name, dpi='figure')
	else:
		plt.show()
	plt.close()
	print('Done plotting progress at {}'.format(ctime(time())))

	if name:
		sys.stdout.write('Check %s\n'%name)

# main function
def main():
	args = arguments()

	if args.label:
		labels = load_labels(args.label)

	if all([args.vector, args.matrix, args.plot, args.train, args.valid]):
		sys.stderr.write('Choose only one type of plotting\n')
		sys.exit(1)

	files = args.file
	embeddings = args.embeddings

	size = tuple(args.size)

	outdir = './'
	if args.output_dir:
		outdir = args.output_dir
	if not outdir.endswith('/'): outdir += '/'

	# plot learnt label embeddings
	if args.vector:
		
		if len(embeddings) == 1 and 'unknown' in embeddings:
			embeddings = ['unknown%s'%i for i in range(len(files))]

		sys.stdout.write(ctime(time()) + ' - Start plotting learnt label embeddings\n')

		for file, emb in zip(files, embeddings):
			sys.stdout.write(ctime(time()) + ' - Plot for %s embedding\n'%emb)
			# load file
			vectors = load_vectors(file)

			# reduce dimension
			vectors = reduce_dim(vectors, args)

			# choose only some labels to plot
			if args.choose_label:
				indices = [labels.index(l) for l in args.choose_label]
				labels = [l.replace('ARGM-','').replace('RG','') for l in args.choose_label]
				vectors = [vectors[idx] for idx in indices]

			# plot
			scatterplot(vectors, labels, emb=emb, save=args.save, size=size, outdir=outdir)

	# plot confusion matrix
	elif args.matrix:
		if len(files)//len(embeddings) != 2:
			sys.stderr.write('Not enough files for embeddings!!\n \t*** For each embedding we need a pair of (true,predicted) files ***\n')
			sys.exit(2)

		sys.stdout.write(ctime(time()) + ' - Start plotting confusion matrix\n')

		files = [[files[i], files[i+1]] for i in range(0,len(files),2)]
		for emb, label_files in zip(embeddings, files):
			sys.stdout.write(ctime(time()) + ' - Plot for %s embedding\n'%emb)
			# load file
			y_true, y_pred = load_matrix(label_files)

			# create datas for true and predicted labels
			true_labels, pred_labels = create_data(y_true, y_pred)

			# calculate confusion matrix for labeling results (errors / correctness)
			matrix = cf_matrix(true_labels, pred_labels, labels, error=args.error)

			# choose only some labels to plot
			if args.choose_label:
				indices = [labels.index(l) for l in args.choose_label]
				labels = [l.replace('ARGM-','').replace('RG','') for l in args.choose_label]
				matrix = matrix[:,indices][indices,:]

			# plot
			cf_matrix_plot(matrix, labels, emb=emb, save=args.save, size=size, outdir=outdir, err=args.error)

	# plot line progress
	elif args.plot:
		sys.stdout.write(ctime(time()) + ' - Start plotting progress of training process\n')
		# prepare datas for whole line plot
		datas_prog = {}

		for emb, file in zip(embeddings, files):
			# load file
			datas_prog[emb] = load_progress(file, train=args.train, valid=args.valid)

		# set data set
		if args.train: 
			dataset = 'train'
		elif args.valid:
			dataset = 'valid'

		# plot
		line(datas_prog, size=size, dataset=dataset, save=args.save, outdir=outdir)

if __name__ == '__main__':
	# display progress logs on stdout
	logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s %(mesagge)s')
	main()











