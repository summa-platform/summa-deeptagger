#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import pickle
import json
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from keras.utils.np_utils import to_categorical
import gzip

embeddings_folder = "word_vectors/"


def load_word_vectors(langs, emb='40'):
	"""
		Function to load pre-trained word vectors.
		Output:
				dictionary {
							"french": np.array([w-fr_1,w-fr_2,...,w-fr_|V-fr|])
							"german": np.array([w-de_1,w-de_2,...,w-de_|V-de|])
							...
						 }
				dictionary {
							"french": [w-fr_1, ...]
							"german": [w-de_1, ...]
						  }
	"""
	print("[*] Loading %s word vectors..." % ','.join(langs))
	wvec, vocab = {}, {}
	print(emb, embeddings_folder)
	for fname in os.listdir(embeddings_folder):
		if fname in langs:
			with (gzip.open if fname.endswith('.gz') else open)(embeddings_folder+fname, 'rb') as f:
				try:
					lang_embeddings = pickle.load(f, encoding='latin1')
				except TypeError:
					f.seek(0)   # reset
					lang_embeddings = pickle.load(f)
			lang = fname.split('.')[0]
			wvec[lang] = lang_embeddings[1]
			vocab[lang] = list(lang_embeddings[0])
			sys.stdout.write("\t%s\r" % (fname) )
			sys.stdout.flush()
	return wvec, vocab


def compute_label_indexes(x, y, labels):
	Y_labs, X_labs, ids = [], [], []
	for idx in range(len(y)):
		y_labs = []
		for yy, yidx in enumerate(y[idx]):
			y_key = '_'.join([str(yy) for yy in yidx])
			if len(yidx) > 0 and y_key in labels:
				y_labs.append(labels.index(y_key))
		if len(y_labs) > 0 and len(x[idx]) > 0:
			Y_labs.append(y_labs)
			X_labs.append(x[idx])
			ids.append(idx)
	return X_labs, Y_labs, ids


def load_yvec(wvec, y_idxs, wpad):
	wdim = wvec[0].shape[0]
	total = wpad
	y_vec = np.zeros((wpad, wdim))
	vecs = wvec[y_idxs[:wpad]]
	y_vec[0:len(vecs)] = vecs
	return y_vec


def load_xvec(wvec, x_idxs, wpad, spad):
	wdim = wvec[0].shape[0]
	total = spad*wpad

	x_vec =[]
	for j,x in enumerate(x_idxs):
		vecs = wvec[x[:wpad]]
		zeros = np.zeros((wpad, wdim))
		zeros[0:len(vecs)] = vecs
		if j == 0:
			x_vec = zeros
		else:
			x_vec = np.vstack([x_vec, zeros])

	if x_vec.shape[0] < total:
		szeros = np.zeros((total - x_vec.shape[0], wdim))
		x_vec = np.vstack([x_vec,szeros])
	else:
		x_vec = x_vec[:total,:]

	return x_vec

def load_vecs_multilabel(wvec, labels, x_idxs, y_idxs, wpad, spad, simple_mode):
	X_vecs, Y_labels = [],  []
	wdim = wvec[0].shape[0]
	total = spad*wpad

	for idx, x_idx_set in enumerate(x_idxs):
		x_vec =[]
		for j,x in enumerate(x_idx_set):
			vecs = wvec[x[:wpad]]
			zeros = np.zeros((wpad, wdim))
			zeros[0:len(vecs)] = vecs
			if j == 0:
				x_vec = zeros
			else:
				x_vec = np.vstack([x_vec, zeros])

		if x_vec.shape[0] < total:
			szeros = np.zeros((total - x_vec.shape[0], wdim))
			x_vec = np.vstack([x_vec,szeros])
		else:
			x_vec = x_vec[:total,:]

		X_vecs.append(x_vec)
		try:
			y_cats =  np.sum(to_categorical(y_idxs[idx], nb_classes=len(labels)),axis=0)
		except:
			y_cats = np.array([])
		y_cats[y_cats>1] = 1
		Y_labels.append(y_cats)
	return np.array(X_vecs), np.array(Y_labels)


def load_data(lang, ltype, vocab, path='', emb='512'):
	print("[*] Loading %s documents..." % lang)
	print("\t%s" % path)
	doc_folder = path
	docs = os.listdir(doc_folder)
	X_idxs, Y_idxs, Y_cidxs, Y_ridxs, fnames = [],[],[],[], []

	for i, fname in enumerate(docs):
		if fname.find('.json') > -1:
			doc = json.load(open(doc_folder+fname))
			sys.stdout.write("\t%s (%d/%d)\r" % (lang, i+1, len(docs)) )
			sys.stdout.flush()
			title = doc["name"].lower()
			if 'teaser' in doc:
				teaser = doc["teaser"].lower()
			else:
				teaser = ""
			if 'text' in doc:
				body = doc['text'].lower()
			else:
				continue
			refgroups, keywords = [], []
			if "referenceGroups" in doc:
				refgroups = doc["referenceGroups"]

			for refgroup in refgroups:
				if refgroup["type"] == "Keywords":

					if lang == "arabic":
						keywords = [w["name"].strip().lower() for w in refgroup['items']]
					else:
						keywords = [w["name"].strip().lower() for w in refgroup['items']]
			category = doc["categoryName"].lower()
			routes = doc["trackingInfo"]["customCriteria"]["x10"].lower().split("::")[2:]
			sentences = [clean(title)]
			sentences += sent_tokenize(clean(teaser))
			sentences += sent_tokenize(clean(body))
			# Exctract word ids and vectors per sentence
			x, x_ids = [], []
			for sentence in sentences:
				vecs, vecs_ids = [], []
				for word in word_tokenize(sentence):
					try:
						idx = vocab[lang].index(word)
						vecs_ids.append(idx)
					except:
						continue
				if len(vecs_ids) > 0:
					x_ids.append(vecs_ids)
			y_ids = extract_wordids(keywords, lang, vocab) #
			y_cids = extract_wordids([category], lang, vocab)
			y_rids = extract_wordids(routes, lang, vocab)

			X_idxs.append(x_ids)
			Y_idxs.append(y_ids)
			Y_cidxs.append(y_cids)
			Y_ridxs.append(y_rids)
			fnames.append(fname)

	h = {'X_idxs': X_idxs,
		 'Y_idxs': Y_idxs,
		 'Y_cidxs': Y_cidxs,
		 'Y_ridxs': Y_ridxs,
         'fnames':fnames}
	if ltype == "kw":
		return X_idxs, Y_idxs, fnames
	elif ltype == "cat":
		return X_idxs, Y_cidxs, fnames
	elif ltype == "rou":
		return X_idxs, Y_ridxs, fnames

def extract_wordids(keywords, lang, vocab):
	y_ids = []
	for keyword in keywords:
		keyword = keyword.strip()
		vecs_ids = []
		for word in keyword.split():
			try:
				idy = vocab[lang].index(word)
				vecs_ids.append(idy)
			except:
				continue
		if len(vecs_ids) > 0:
			y_ids.append(vecs_ids)
	return y_ids


def clean(text):
	text = text.replace('\n', ' ')
	text = text.replace('\r', ' ')
	text = text.replace('\t', ' ')
	return text
