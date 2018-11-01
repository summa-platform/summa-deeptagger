
from __future__ import print_function

import os
os.environ['THEANO_FLAGS'] = "mode=FAST_COMPILE,device=gpu,lib.cnmem=1,floatX=float32"
import sys
import time
import theano
import theano.tensor as T
import numpy as np
import json
from processing import load_data, load_word_vectors, load_vecs_multilabel, compute_label_indexes
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import merge,  Merge, Permute, RepeatVector, Flatten, TimeDistributed, Dense, Input, Dense, Dropout, Activation
from keras.layers.core import Lambda, Reshape
from keras.optimizers import Adam
from keras import backend as K
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation



class DeepTag:
	"""
	"""
	def __init__(self, wdim=40, ep=50, bs=16,
					   sdim=256, ddim=256, swpad=40, spad=40,
					   ltype="", comp="td", mode="deep",
					   gt=0, act="sigmoid"):
		self.composition = comp #composition function
		self.wdim = wdim   #word vectors dimension
		self.wpad = swpad*spad #max words per document
		self.swpad = swpad  #max words per sentence
		self.spad = spad   #max sentences per document
		self.sdim = sdim  #sentence embedding dimension
		self.ddim = ddim  #document embedding dimension
		self.bs = bs     #size of batch
		self.mode = mode #type of model
		self.epochs = ep # num of epochs
		self.ltype = ltype
		self.gt = gt
		self.activation = act
		self.func = 'sum'

	def eval(self, x, y, t, wvec, simple_mode, bs=512, av='micro'):
		accs, preds, scores, real, satts, watts = [], [], [], [], [], []
		batch, elapsed, curbatch = 0, 0, 0
		print()
		while( batch < (len(x)/(1.0*bs)) ):
			cur_x = x[curbatch:curbatch+bs]
			cur_y = y[curbatch:curbatch+bs]
			x_vecs, y_vecs = load_vecs_multilabel(wvec, self.labels, cur_x, cur_y, self.swpad, self.spad, simple_mode)
			pred = self.model.predict(x_vecs)
			try:
				ss = self.satt(x_vecs)
				ww = self.watt(x_vecs)
				satts.append(ss)
				watts.append(ww)
			except:
				watts.append(np.zeros(self.swpad))
				satts.append(np.zeros(self.spad))
				pass
			preds.append(pred>t)
			scores.append(pred)
			real.append(y_vecs)
			sys.stdout.write(("%d/%d\r"%(((batch+1)*bs), len(x))))
			sys.stdout.flush()
			batch += 1
			curbatch += bs
		real = np.array([rr for r in real for rr in r])
		preds = np.array([pp for p in preds for pp in p])
		satts = np.array([ss for s in satts for ss in s])
		watts = np.array([ww for w in watts for ww in w])

		return [satts, watts, scores]


	def build_model(self, labels):
		self.labels = labels
		L = len(self.labels)
		if self.composition in ["td","atttd","datttd"]:
			label_embedding =  Dense(self.sdim,
											 input_shape=(self.wdim,),
											 activation=self.activation,
											)
			sentence_embedding = TimeDistributed(Dense(self.sdim,
											 input_shape=(self.wpad, self.wdim),
											 activation=self.activation,
											),
											input_shape=(self.wpad, self.wdim))
			document_embedding = TimeDistributed(Dense(self.ddim,
											 input_shape=(self.spad, self.wdim),
											 activation=self.activation,
											),
											 input_shape=(self.spad, self.wdim))


		input_model = Sequential()
		if self.composition in ["td", "atttd"]:
			words = Input(shape=(self.wpad,self.wdim,))
			forward_words = sentence_embedding(words)
			if self.composition in ["atttd"]:
				forward_words = self.word_attention(forward_words)
			word_pooling = Lambda(self.compose, output_shape=self.compose_output)
			sentences = word_pooling(forward_words)
			forward_doc = document_embedding(sentences)
			if self.composition in ["atttd"]:
				forward_doc = self.sent_attention(forward_doc)

			sentence_pooling = Lambda(self.dcompose, output_shape=self.dcompose_output, name='doc_pool')
			document = sentence_pooling(forward_doc)
			classifier = Dense(L, input_dim=(document._keras_shape[1]),
								  activation='sigmoid')
			decision = classifier(document)
			input_model = Model(input=words, output=decision)

		input_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["accuracy"])
		self.model = input_model
		print("Storing forward functions...")
		if self.model.get_layer('watt') is not None:
			self.watt = theano.function([self.model.layers[0].input],
										 self.model.get_layer('watt').get_output_at(0),
										 allow_input_downcast=True)
			self.satt = theano.function([self.model.layers[0].input],
										 self.model.get_layer('satt').get_output_at(0) ,
										 allow_input_downcast=True)

	def word_attention(self, forward_words):
		pad = forward_words._keras_shape[1]
		hdim = forward_words._keras_shape[2]
		attention = TimeDistributed(Dense(1, activation=self.activation))
		weights = attention(forward_words)
		weights = Flatten()(weights)
		reshaped = Reshape((self.spad,self.swpad))(weights)
		submaxed = Lambda(self.submax, output_shape=self.submax_output)(reshaped)
		weights = Activation(activation="softmax", name='watt')(submaxed)
		weights = Flatten()(weights)
		weights = RepeatVector(hdim)(weights)
		weights = Permute((2,1))(weights)
		merged = merge([weights, forward_words], "mul")
		return merged

	def submax(self, x):
		return x - K.max(x, axis=-1, keepdims=True)

	def submax_output(self, input_shape):
		return tuple(input_shape)

	def sent_attention(self, forward_doc):
		hdim = forward_doc._keras_shape[2]
		attention = TimeDistributed(Dense(1, activation=self.activation))
		sent_weights = attention(forward_doc)
		sent_weights = Flatten()(sent_weights)
		submaxed = Lambda(self.submax, output_shape=self.submax_output)(sent_weights)
		sent_weights = Activation(activation="softmax", name='satt')(submaxed)
		sent_weights = RepeatVector(hdim)(sent_weights)
		sent_weights = Permute((2,1))(sent_weights)
		sent_merged = merge([sent_weights, forward_doc], "mul")
		return sent_merged


	def compose(self, encoded_words):
		cur_shape = encoded_words._keras_shape
		reshape = Reshape((self.spad,self.swpad,cur_shape[2]), input_shape=cur_shape)
		if self.func == 'mean':
			return K.mean(reshape(encoded_words),axis=2)
		elif self.func == 'max':
			return K.max(reshape(encoded_words),axis=2)
		elif self.func == 'min':
			return K.min(reshape(encoded_words),axis=2)
		else:
			return K.sum(reshape(encoded_words),axis=2)

	def compose_output(self, input_shape):
		return tuple([None, self.spad, input_shape[2]])

	def dcompose(self, encoded_document):
		if self.func == 'mean':
			return K.mean(encoded_document, axis=1)
		elif self.func == 'max':
			return K.max(encoded_document, axis=1)
		elif self.func == 'min':
			return K.min(encoded_document, axis=1)
		else:
			return K.sum(encoded_document, axis=1)


	def dcompose_output(self, input_shape):
		return tuple([None, input_shape[2]])

if __name__ == "__main__":
	print("STARTED")
	language = sys.argv[1]
	input_language = sys.argv[2]
	label_type = sys.argv[3].split("--lt=")[1]
	comp = sys.argv[4].split("--comp=")[1]
	act = sys.argv[5].split("--act=")[1]
	wdim = sys.argv[6].split("--wdim=")[1]
	gt = int(sys.argv[7].split("--gt=")[1])
	swpad = int(sys.argv[8].split("--swpad=")[1])
	spad = int(sys.argv[9].split("--spad=")[1])
	sdim = int(sys.argv[10].split("--sdim=")[1])
	ddim = int(sys.argv[11].split("--ddim=")[1])
	bs = int(sys.argv[12].split("--bs=")[1])
	ep = int(sys.argv[13].split("--ep=")[1])
	test_path = ""
	if len(sys.argv) > 14:
		test_path = sys.argv[14].split('--test=')[1]
	top_k = 10

	label_types = ['kw','cat','rou']
	dt = DeepTag(ep=ep,
				 wdim=int(wdim.replace('g-','')),
				 comp=comp,
				 act=act,
				 swpad=swpad,
				 spad=spad,
				 sdim=sdim,
				 ddim=ddim,
				 ltype=label_type,
				 gt=gt,
				 bs=bs)
	labels = json.load(open('models/labels-'+label_type+'_'+language+'.p'))
	wvec, vocab = load_word_vectors(['%s.pkl' % input_language, '%s.pkl' % language], emb=wdim)
	wvec_input =  np.array(wvec[input_language])
	xt_ids, yt_ids, fnames = load_data(input_language, label_type, vocab, path='documents/%s/' % input_language, emb=wdim)
	dt.build_model(labels)
	dt.model.load_weights(test_path)
	res = dt.eval(xt_ids[:], yt_ids[:], 0.20, wvec_input, False, bs=128, av='micro')
	lang_tags = dict()
	satts = res[0]
	watts = res[1]
	for res_i, rawscores in enumerate(res[2][0]):
		idxs = np.argsort(rawscores)[::-1]
		scores = np.sort(rawscores)[::-1]
		lang_tags[fnames[res_i]] = {'tags':[], 'satts':[], 'watts':[], 'text':[], 'labels':[]}
		for i, idx in enumerate(idxs[:top_k]):
			toks = labels[idx].split('_')
			tag = []
			for j, tok in enumerate(toks):
				tag.append(vocab[language][int(tok)])
			lang_tags[fnames[res_i]]['tags'].append([' '.join(tag), '%.5f' % scores[i]])
			if len(satts) > 0:
				sats_text, wats_text = [], []
				for sidx, sat in enumerate(satts[res_i]):
					wat = watts[res_i][sidx]
					cur_text = []
					sats_text.append("%.3f" % sat)
					for wval in wat:
						cur_text.append("%.3f" % wval)
					wats_text.append(cur_text)
				lang_tags[fnames[res_i]]['satts'] = sats_text
				lang_tags[fnames[res_i]]['watts'] = wats_text
			text, label_text = [], []
			for sentence_ids in xt_ids[res_i]:
				text.append([vocab[input_language][idx] for idx in sentence_ids])
			for label_ids in yt_ids[res_i]:
				lab = []
				for idy in label_ids:
					lab.append(vocab[input_language][idy])
				label_text.append(' '.join(lab))
			lang_tags[fnames[res_i]]['text'] = text
			lang_tags[fnames[res_i]]['labels'] = label_text
	
	directory = 'documents/%s/tags' % input_language
	if not os.path.exists(directory):
    		os.makedirs(directory)
	json.dump(lang_tags, open('documents/%s/tags/%s_%s.json' % (input_language, label_type,language) , 'w'))
