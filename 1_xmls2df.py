# coding=latin-1
import util200818 as ut
import argparse, os, re, sys, time
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
from collections import defaultdict, Counter
from scipy import special
from scipy.sparse import csr_matrix, save_npz, load_npz
import random
import torch
import torch.nn as nn
from torch import optim
# import warnings filter
from warnings import simplefilter
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-dir_decour", type=str, default='0_data/decour.1/')
args = parser.parse_args()
sys.stdout = sys.stderr = log = ut.log(__file__, ut.__file__)
startime = ut.start()
align_size = ut.print_args(args)
print(f"{'dirout':.<{align_size}} {log.pathtime}")
###############################################################################
path_xmls = sorted([f"{args.dir_decour}{f}" for f in os.listdir(args.dir_decour) if re.search('xml$', f)])

rows = list()
for path_xml in path_xmls:
    tree = ET.parse(path_xml)
    hearing = tree.getroot()
    for turn in hearing.iter('turn'):
        for iutt, utt in enumerate(turn.iter('utterance')):
            rows.append([int(hearing.find('header').attrib['idhear']),
                         int(hearing.find('header').attrib['idsub']),
                         hearing.find('header').attrib['sex'],
                         hearing.find('header').attrib['age'],
                         hearing.find('header').attrib['birtharea'],
                         hearing.find('header').attrib['study'],
                         turn.attrib['speaker'],
                         'interviewee' if turn.attrib['speaker'] in 'def defwit expwit vic'.split() else 'interviewer',
                         int(turn.attrib['nrgen']),
                         iutt + 1,
                         int(utt.attrib['nrgen']),
                         utt.attrib['class'],
                         re.sub("\t+", ' ', utt.text.lower()),
                         utt.find('ttlemma').text])

df = pd.DataFrame(rows, columns='nhear idsub gender age birth study speaker role nturn nutt_inturn nutt label tokens lemmas'.split())
df.label = df.label.replace({'false': 0, 'true': 1, 'uncertain': 2, 'x': 3})

print(df.head())
print(df.shape)

col_prev_utt1_tokens                 = list()
col_prev_utt2_tokens                 = list()
col_prev_utt3_tokens                 = list()
col_prev_utt4_tokens                 = list()
col_prev_utt5_tokens                 = list()
col_prev_utt1_lemmas                 = list()
col_prev_utt2_lemmas                 = list()
col_prev_utt3_lemmas                 = list()
col_prev_utt4_lemmas                 = list()
col_prev_utt5_lemmas                 = list()
col_prev_speakerutt_tokens           = list()
col_prev_speakerutts_tokens          = list()
col_prev_turn_tokens                 = list()
col_prev_speakerutts_and_turn_tokens = list()
col_prev_speakerutt_lemmas           = list()
col_prev_speakerutts_lemmas          = list()
col_prev_turn_lemmas                 = list() 
col_prev_speakerutts_and_turn_lemmas = list()

for nhear in range(1, max(df.nhear) + 1):
    for nturn in range(1, int(max(df.nturn[df.nhear == nhear])) + 1):
        for nutt_inturn in range(1, max(df.nutt_inturn[(df.nhear == nhear) & (df.nturn == nturn)]) + 1):
            utt = df.tokens[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn == nutt_inturn)].values[0]

            prev_utt1_tokens = ' '.join(df.tokens[df.index[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn == nutt_inturn)][0] - 1: df.index[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn == nutt_inturn)][0]].values)
            prev_utt2_tokens = ' '.join(df.tokens[df.index[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn == nutt_inturn)][0] - 2: df.index[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn == nutt_inturn)][0]].values)
            prev_utt3_tokens = ' '.join(df.tokens[df.index[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn == nutt_inturn)][0] - 3: df.index[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn == nutt_inturn)][0]].values)
            prev_utt4_tokens = ' '.join(df.tokens[df.index[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn == nutt_inturn)][0] - 4: df.index[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn == nutt_inturn)][0]].values)
            prev_utt5_tokens = ' '.join(df.tokens[df.index[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn == nutt_inturn)][0] - 5: df.index[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn == nutt_inturn)][0]].values)
            prev_utt1_lemmas = ' '.join(df.lemmas[df.index[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn == nutt_inturn)][0] - 1: df.index[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn == nutt_inturn)][0]].values)
            prev_utt2_lemmas = ' '.join(df.lemmas[df.index[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn == nutt_inturn)][0] - 2: df.index[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn == nutt_inturn)][0]].values)
            prev_utt3_lemmas = ' '.join(df.lemmas[df.index[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn == nutt_inturn)][0] - 3: df.index[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn == nutt_inturn)][0]].values)
            prev_utt4_lemmas = ' '.join(df.lemmas[df.index[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn == nutt_inturn)][0] - 4: df.index[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn == nutt_inturn)][0]].values)
            prev_utt5_lemmas = ' '.join(df.lemmas[df.index[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn == nutt_inturn)][0] - 5: df.index[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn == nutt_inturn)][0]].values)

            prev_speakerutt_tokens           = df.tokens[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn == nutt_inturn - 1)].values[0] if nutt_inturn > 1 else ''
            prev_speakerutts_tokens          = ' '.join(df.tokens[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn < nutt_inturn)].values) if nutt_inturn > 1 else ''
            prev_turn_tokens                 = ' '.join(df.tokens[(df.nhear == nhear) & (df.nturn == nturn - 1)].values) if nturn > 1 else ''
            prev_speakerutts_and_turn_tokens = f"{prev_turn_tokens} {prev_speakerutts_tokens}".rstrip()

            prev_speakerutt_lemmas           = df.lemmas[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn == nutt_inturn - 1)].values[0] if nutt_inturn > 1 else ''
            prev_speakerutts_lemmas          = ' '.join(df.lemmas[(df.nhear == nhear) & (df.nturn == nturn) & (df.nutt_inturn < nutt_inturn)].values) if nutt_inturn > 1 else ''
            prev_turn_lemmas                 = ' '.join(df.lemmas[(df.nhear == nhear) & (df.nturn == nturn - 1)].values) if nturn > 1 else ''
            prev_speakerutts_and_turn_lemmas = f"{prev_turn_lemmas} {prev_speakerutts_lemmas}".rstrip()

            col_prev_utt1_tokens.append(prev_utt1_tokens)
            col_prev_utt2_tokens.append(prev_utt2_tokens)
            col_prev_utt3_tokens.append(prev_utt3_tokens)
            col_prev_utt4_tokens.append(prev_utt4_tokens)
            col_prev_utt5_tokens.append(prev_utt5_tokens)
            col_prev_utt1_lemmas.append(prev_utt1_lemmas)
            col_prev_utt2_lemmas.append(prev_utt2_lemmas)
            col_prev_utt3_lemmas.append(prev_utt3_lemmas)
            col_prev_utt4_lemmas.append(prev_utt4_lemmas)
            col_prev_utt5_lemmas.append(prev_utt5_lemmas)

            col_prev_speakerutt_tokens.append(prev_speakerutt_tokens)
            col_prev_speakerutts_tokens.append(prev_speakerutts_tokens)
            col_prev_turn_tokens.append(prev_turn_tokens)
            col_prev_speakerutts_and_turn_tokens.append(prev_speakerutts_and_turn_tokens)

            col_prev_speakerutt_lemmas.append(prev_speakerutt_lemmas)
            col_prev_speakerutts_lemmas.append(prev_speakerutts_lemmas)
            col_prev_turn_lemmas.append(prev_turn_lemmas)
            col_prev_speakerutts_and_turn_lemmas.append(prev_speakerutts_and_turn_lemmas)

            if (nhear == 35) and (nturn > 43):
                print(f"nhear {nhear},  nturn {nturn}, nutt_inturn {nutt_inturn}")
                print(f"{'prev_utt1_lemmas':<30}{prev_utt1_tokens[:80]}")
                print(f"{'prev_utt2_lemmas':<30}{prev_utt2_tokens[:80]}")
                print(f"{'prev_utt3_lemmas':<30}{prev_utt3_tokens[:80]}")
                print(f"{'prev_utt4_lemmas':<30}{prev_utt4_tokens[:80]}")
                print(f"{'prev_utt5_lemmas':<30}{prev_utt5_tokens[:80]}")
                print(f"{'prev_turn':<30}{prev_turn_tokens[:80]}")
                print(f"{'prev_speakerutts':<30}{prev_speakerutts_tokens[:80]}")
                print(f"{'prev_speakerutt':<30}{prev_speakerutt_tokens[:80]}")
                print(f"{'prev_speakerutts_and_turn':<30}{prev_speakerutts_and_turn_tokens[:80]}")
                print(f"{'utt':<30}{utt[:80]}")

df['prev_utt1_tokens'] = col_prev_utt1_tokens
df['prev_utt2_tokens'] = col_prev_utt2_tokens
df['prev_utt3_tokens'] = col_prev_utt3_tokens
df['prev_utt4_tokens'] = col_prev_utt4_tokens
df['prev_utt5_tokens'] = col_prev_utt5_tokens
df['prev_utt1_lemmas'] = col_prev_utt1_lemmas
df['prev_utt2_lemmas'] = col_prev_utt2_lemmas
df['prev_utt3_lemmas'] = col_prev_utt3_lemmas
df['prev_utt4_lemmas'] = col_prev_utt4_lemmas
df['prev_utt5_lemmas'] = col_prev_utt5_lemmas
df['prev_speakerutt_tokens'] = col_prev_speakerutt_tokens
df['prev_speakerutts_tokens'] = col_prev_speakerutts_tokens
df['prev_turn_tokens'] = col_prev_turn_tokens
df['prev_speakerutts_and_turn_tokens'] = col_prev_speakerutts_and_turn_tokens
df['prev_speakerutt_lemmas'] = col_prev_speakerutt_lemmas
df['prev_speakerutts_lemmas'] = col_prev_speakerutts_lemmas
df['prev_turn_lemmas'] = col_prev_turn_lemmas
df['prev_speakerutts_and_turn_lemmas'] = col_prev_speakerutts_and_turn_lemmas


print(f"nr <unknown> nei contesti a 6139:")
df.replace('', np.nan, inplace=True) # passo da nan solo per contarli
print(df.isna().sum(axis=0))
print(f"nr <unknown> nei contesti a 3015:")
print(df[df.role == 'interviewee'].isna().sum(axis=0))
df.fillna('<unknown>', inplace=True)


df.to_csv(f"{log.pathtime}decour.tsv", sep="\t", header=True)
df = pd.read_csv(f"{log.pathtime}decour.tsv", sep="\t", index_col=0)

print(df.tail(15))
print(df.shape)
# print('df space on disk:', df.memory_usage(index=True, deep=True).sum()) # mah, non corrisponde al csv

ut.end(startime)
