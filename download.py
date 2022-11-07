#!/usr/bin/env python3

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

import gensim.downloader
v2w_model = gensim.downloader.load('word2vec-google-news-300')
glove_model = gensim.downloader.load('glove-twitter-25')
