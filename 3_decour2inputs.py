# coding=latin-1
import util200818 as ut
import argparse, os, re, sys, time
import numpy as np
from scipy.sparse import csr_matrix, save_npz
import pandas as pd
from scipy import stats
from collections import defaultdict, Counter
from math import sin, cos, sqrt, atan2, radians
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from gensim.corpora import Dictionary
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
###################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-path_decour_tsv", type=str,   default='1_xmls2df/201002190320/decour.tsv')
parser.add_argument("-path_emb",        type=str,   default='2_fasttext2embs/200626104921/it_emb.json')
parser.add_argument("-path_w2i",        type=str,   default='2_fasttext2embs/200626104921/it_w2i.json')
parser.add_argument("-no_below",        type=int,   default=0,     help='min word freq')
parser.add_argument("-no_above",        type=float, default=100,   help='max % of docs where the word is found')
parser.add_argument("-pad_rate",        type=float, default=99.5,  help='rate pad wrt maxlen')
parser.add_argument("-answer_padsize",  type=int,   default=100)
parser.add_argument("-context_padsize", type=int,   default=400)
args = parser.parse_args()
sys.stdout = sys.stderr = log = ut.log(__file__, ut.__file__)
startime = ut.start()
ut.print_args(args)
print("dirout:\t", log.pathtime)
###################################################################################################
df = pd.read_csv(args.path_decour_tsv, sep="\t", index_col=0)
print(df.head())
print(df.shape)
# for col in df.columns: print(col)
print(df.columns)

emb = ut.readjson(args.path_emb)
w2i = ut.readjson(args.path_w2i)
print(f"fasttext embs shape: {np.shape(emb)}")
dictionary = Dictionary([lemmas.split() for lemmas in df.lemmas.values])
print("dictionary size before filtering:\t", len(dictionary))
dictionary.filter_extremes(no_below=args.no_below, no_above=args.no_above)
print("dictionary size after filtering:\t", len(dictionary))
vocab = set(dictionary.token2id.keys())
voc_intersection = vocab.intersection(w2i.keys())
lenintersection = len(voc_intersection)
lenvocab = len(vocab)
coverage = round(lenintersection / lenvocab * 100, 2)
print("dictionary size:\t{}\nintersection:\t\t{}\ncoverage:\t\t{}".format(lenvocab, lenintersection, coverage))

embmatrix = np.array([emb[w2i[word]] for word in voc_intersection])
zeros = np.zeros((1, embmatrix.shape[1])) # pad
meanvector = np.reshape(embmatrix.mean(axis=0), (1, embmatrix.shape[1])) # oov
embmatrix = np.concatenate((zeros, meanvector, embmatrix), axis=0)  # null word in row 0
word2index = {word: (index + 2) for index, word in enumerate(voc_intersection)} # lascio 0 per pad e 1 per oov
# ut.printjson(word2index)
np.save(log.pathtime + 'embmatrix', embmatrix)
ut.writejson(word2index, log.pathtime + 'word2index.json')

for word in word2index:
    if word2index[word] == 2:
        print('tf word con index 2:  ', word)
        print('index in fasttext:    ', w2i[word])
        print('fasttext vector[:5]:  ', emb[w2i[word]][:5])
        print('tf vector[:5]:        ', embmatrix[word2index[word]][:5])
        print('fasttext matrix shape:', np.shape(emb))
        print('tf matrix shape:      ', np.shape(embmatrix))


print('in caso di testi > padsize, prendo la parte finale')
print('pad_rate a 99.5 permette di selezionare answer e contesto complessivamente < 500, a scapito però della lunghezza massima della answer (99.5 -> 90, 99.9 -> 130)')


def pad(texts, padsize, name):
    lens = [len(text.split())for text in texts]
    theoretic_padsize = int(round(np.percentile(lens, args.pad_rate), -1))
    unpad_ids = [[word2index.get(word, 1) for word in text.split()] for text in texts] # index 1 for oov. lo uso anche come parola unica nelle frasi mancanti, [1]
    pad_ids  = csr_matrix([row + [0] * (padsize - len(row)) if len(row) < padsize else row[-padsize:] for row in unpad_ids]) # index 0 for pad
    pad_mask = csr_matrix([[1] * len(row) + [0] * (padsize - len(row)) if len(row) < padsize else [1] * padsize for row in unpad_ids])
    save_npz(f"{log.pathtime}{name}_ids", pad_ids)
    save_npz(f"{log.pathtime}{name}_mask", pad_mask)
    print(f"shape ids/mask {name:.<50} {pad_ids.shape} (pad size teorico con pad rate {args.pad_rate}: {theoretic_padsize})")
    if (name == 'prev_speakerutt_tokens') or (name == 'tokens'):
        print(f"esempio per {name}:")
        print(unpad_ids[0], texts[0])
        print(pad_ids[0].toarray())
        print(pad_mask[0].toarray())
    return 1


pad(df.tokens.values, args.answer_padsize,                            'decour6139_tokens')
pad(df.lemmas.values, args.answer_padsize,                            'decour6139_lemmas')
pad(df.prev_utt1_tokens.values, args.context_padsize,                 'decour6139_prev_utt1_tokens')
pad(df.prev_utt2_tokens.values, args.context_padsize,                 'decour6139_prev_utt2_tokens')
pad(df.prev_utt3_tokens.values, args.context_padsize,                 'decour6139_prev_utt3_tokens')
pad(df.prev_utt4_tokens.values, args.context_padsize,                 'decour6139_prev_utt4_tokens')
pad(df.prev_utt5_tokens.values, args.context_padsize,                 'decour6139_prev_utt5_tokens')
pad(df.prev_speakerutt_tokens.values, args.context_padsize,           'decour6139_prev_speakerutt_tokens')
pad(df.prev_speakerutts_tokens.values, args.context_padsize,          'decour6139_prev_speakerutts_tokens')
pad(df.prev_speakerutts_and_turn_tokens.values, args.context_padsize, 'decour6139_prev_speakerutts_and_turn_tokens')
pad(df.prev_turn_tokens.values, args.context_padsize,                 'decour6139_prev_turn_tokens')
pad(df.prev_utt1_lemmas.values, args.context_padsize,                 'decour6139_prev_utt1_lemmas')
pad(df.prev_utt2_lemmas.values, args.context_padsize,                 'decour6139_prev_utt2_lemmas')
pad(df.prev_utt3_lemmas.values, args.context_padsize,                 'decour6139_prev_utt3_lemmas')
pad(df.prev_utt4_lemmas.values, args.context_padsize,                 'decour6139_prev_utt4_lemmas')
pad(df.prev_utt5_lemmas.values, args.context_padsize,                 'decour6139_prev_utt5_lemmas')
pad(df.prev_speakerutt_lemmas.values, args.context_padsize,           'decour6139_prev_speakerutt_lemmas')
pad(df.prev_speakerutts_lemmas.values, args.context_padsize,          'decour6139_prev_speakerutts_lemmas')
pad(df.prev_speakerutts_and_turn_lemmas.values, args.context_padsize, 'decour6139_prev_speakerutts_and_turn_lemmas')
pad(df.prev_turn_lemmas.values, args.context_padsize,                 'decour6139_prev_turn_lemmas')


df = df[df.label != 3]
df.reset_index(drop=True, inplace=True)
print(df.head())
print(df.shape)

df.to_csv(f"{log.pathtime}decour_targets.tsv", sep="\t", header=True)
# df = pd.read_csv(f"{log.pathtime}decour_targets.tsv", sep="\t", index_col=0)


pad(df.tokens.values, args.answer_padsize,                            'decour3015_tokens')
pad(df.lemmas.values, args.answer_padsize,                            'decour3015_lemmas')
pad(df.prev_utt1_tokens.values, args.context_padsize,                 'decour3015_prev_utt1_tokens')
pad(df.prev_utt2_tokens.values, args.context_padsize,                 'decour3015_prev_utt2_tokens')
pad(df.prev_utt3_tokens.values, args.context_padsize,                 'decour3015_prev_utt3_tokens')
pad(df.prev_utt4_tokens.values, args.context_padsize,                 'decour3015_prev_utt4_tokens')
pad(df.prev_utt5_tokens.values, args.context_padsize,                 'decour3015_prev_utt5_tokens')
pad(df.prev_speakerutt_tokens.values, args.context_padsize,           'decour3015_prev_speakerutt_tokens')
pad(df.prev_speakerutts_tokens.values, args.context_padsize,          'decour3015_prev_speakerutts_tokens')
pad(df.prev_speakerutts_and_turn_tokens.values, args.context_padsize, 'decour3015_prev_speakerutts_and_turn_tokens')
pad(df.prev_turn_tokens.values, args.context_padsize,                 'decour3015_prev_turn_tokens')
pad(df.prev_utt1_lemmas.values, args.context_padsize,                 'decour3015_prev_utt1_lemmas')
pad(df.prev_utt2_lemmas.values, args.context_padsize,                 'decour3015_prev_utt2_lemmas')
pad(df.prev_utt3_lemmas.values, args.context_padsize,                 'decour3015_prev_utt3_lemmas')
pad(df.prev_utt4_lemmas.values, args.context_padsize,                 'decour3015_prev_utt4_lemmas')
pad(df.prev_utt5_lemmas.values, args.context_padsize,                 'decour3015_prev_utt5_lemmas')
pad(df.prev_speakerutt_lemmas.values, args.context_padsize,           'decour3015_prev_speakerutt_lemmas')
pad(df.prev_speakerutts_lemmas.values, args.context_padsize,          'decour3015_prev_speakerutts_lemmas')
pad(df.prev_speakerutts_and_turn_lemmas.values, args.context_padsize, 'decour3015_prev_speakerutts_and_turn_lemmas')
pad(df.prev_turn_lemmas.values, args.context_padsize,                 'decour3015_prev_turn_lemmas')


ut.end(startime)
