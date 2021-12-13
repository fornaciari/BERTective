# coding=latin-1
import util200818 as ut
import step200928 as st
import models20928 as mod
import argparse, os, re, sys, time
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, log_loss, confusion_matrix
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
# paths
parser.add_argument("-path_decour6139",                                 type=str, default='1_xmls2df/200825171412/decour.tsv')
parser.add_argument("-path_decour3015",                                 type=str, default='3_decour2inputs/201002191852/decour_targets.tsv')

parser.add_argument("-path_fastext_emb",                                type=str, default='3_decour2inputs/200827103839/embmatrix.npy')
parser.add_argument("-path_fastext_w2i",                                type=str, default='3_decour2inputs/200827103839/word2index.json')

# input settings
parser.add_argument("-answer",       type=str, default='tokens', choices=['tokens', 'lemmas'])
parser.add_argument("-context",      type=str, default='prev_utt3_tokens', choices=['prev_speakerutt_tokens', 'prev_speakerutts_tokens', 'prev_speakerutts_and_turn_tokens', 'prev_turn_tokens', 'prev_speakerutt_lemmas', 'prev_speakerutts_lemmas', 'prev_speakerutts_and_turn_lemmas', 'prev_turn_lemmas', 'prev_utt1_tokens', 'prev_utt2_tokens', 'prev_utt3_tokens', 'prev_utt4_tokens', 'prev_utt5_tokens'])
parser.add_argument("-hier_context", type=int, default=5, help="nr di utt prima del target, per i modelli gerarchici")
# torch settings
parser.add_argument("-seed",   type=int, default=1234)
parser.add_argument("-device", type=str, default='cuda:2')
parser.add_argument("-dtype",  type=int, default=32, choices=[32, 64])
# preproc
parser.add_argument("-vocsize",         type=int,  default=25000)
parser.add_argument("-answer_padsize",  type=int,  default=200)
parser.add_argument("-context_padsize", type=int,  default=300)
parser.add_argument("-padsize",         type=int,  default=505)
# bert/embeddings settings
parser.add_argument("-pairbert",        type=bool, default=True)
parser.add_argument("-trainable",       type=bool, default=False)
# model settings
parser.add_argument("-save",        type=bool,  default=False)
parser.add_argument("-experiments", type=int,   default=10)
parser.add_argument("-epochs",      type=int,   default=5)
# parser.add_argument("-splits",      type=int,   default=10, help='almeno 3 o dà un errore, credo dovuto all\'output dello stratified')
parser.add_argument("-batsize",     type=int,   default=8)
parser.add_argument("-learate",     type=float, default=0.0001)
parser.add_argument("-droprob",     type=float, default=0.1)
# conv settings
parser.add_argument("-conv_channels",     type=int, nargs='+', default=[32, 64, 128],  help="nr of channels conv by conv")
parser.add_argument("-conv_filter_sizes", type=int, nargs='+', default=[2, 4, 6], help="sizes of filters: window, in each conv")
parser.add_argument("-conv_stridesizes",  type=int, nargs='+', default=[1, 1, 1],    help="conv stride size, conv by conv")
parser.add_argument("-pool_filtersizes",  type=int, nargs='+', default=[2, 2, 2],    help="pool filter size, conv by conv. in order to have a vector as output, the last value will be substituted with the column size of the last conv, so that the last column size will be 1, then squeezed")
parser.add_argument("-pool_stridesizes",  type=int, nargs='+', default=[1, 1, 1],    help="pool stride size, conv by conv")
# lstm settings
parser.add_argument("-lstm_layers", type=int,   default=1)
parser.add_argument("-lstm_size",   type=int,   default=128)
parser.add_argument("-bidir",       type=bool,  default=True)
# attention
parser.add_argument("-att_heads",   type=int, default=6)
parser.add_argument("-att_layers",  type=int, default=6)
# mlp settings
parser.add_argument("-mlp_layers", type=int, default=1)
args = parser.parse_args()
sys.stdout = sys.stderr = log = ut.log(__file__, ut.__file__, st.__file__, mod.__file__)
startime = ut.start()
align_size = ut.print_args(args)
print(f"{'dirout':.<{align_size}} {log.pathtime}")
###################################################################################################
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
dtype_float = torch.float64 if args.dtype == 64 else torch.float32
dtype_int = torch.int64 # if args.dtype == 64 else torch.int32 # o 64 o s'inkazza
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(f"{'GPU in use':.<{align_size}} {device}\n{'#'*80}" if torch.cuda.is_available() else f"No GPU available, using the CPU.\n{'#'*80}")
###############################################################################
df3015 = pd.read_csv(args.path_decour3015, sep="\t", index_col=0)
print(df3015.head())
print(df3015.shape)

df6139 = pd.read_csv(args.path_decour6139, sep="\t", index_col=0)
print(df6139.head())
print(df6139.shape)


proc = st.Processing(log.pathtime, device)
targs_all, preds_all = list(), list()

###################################################################################################


df = pd.read_csv(args.path_decour3015, sep="\t", index_col=0)
print(df.head())
print(df.shape)

assert (re.search('token', args.answer) and re.search('token', args.context)) or \
       (re.search('lemma', args.answer) and re.search('lemma', args.context)), 'tokens/lemmas non coerenti in answer/context'
answers  = df[args.answer].values
contexts = df[args.context].values
if args.pairbert:
    bert_word_ids, bert_word_mask = proc.bert_twosent_preproc('it', answers, contexts, args.padsize)
else:
    bert_word_ids, bert_word_mask, vocsize = proc.bert_preproc('it', answers, args.padsize)

# model = mod.StlBertMatSigmoidFc('it', trainable=args.trainable, fc_layers=args.mlp_layers, droprob=args.droprob, device=device)
for nhear in range(1, 36):
    model = mod.StlBertMatAttFcSigmoid('it', args.att_heads, args.att_layers, trainable=args.trainable, fc_layers=args.mlp_layers, droprob=args.droprob, device=device)
    lossfunc = nn.BCELoss().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learate)

    hearing_tst = nhear
    hearing_dev = nhear + 1 if nhear != 35 else 1
    hearings_trn = [n for n in range(1, 36) if n not in [hearing_tst, hearing_dev]]

    trn_df_ids = df.index[df.nhear.isin(hearings_trn)]
    dev_df_ids = df.index[df.nhear == hearing_dev]
    tst_df_ids = df.index[df.nhear == hearing_tst]

    y_trn = df.label[trn_df_ids].replace({2: 1}).to_numpy()
    y_dev = df.label[dev_df_ids].replace({2: 1}).to_numpy()
    y_tst = df.label[tst_df_ids].replace({2: 1}).to_numpy()

    ids_trn   = bert_word_ids[trn_df_ids]
    ids_dev   = bert_word_ids[dev_df_ids]
    ids_tst   = bert_word_ids[tst_df_ids]

    mask_trn  = bert_word_mask[trn_df_ids]
    mask_dev  = bert_word_mask[dev_df_ids]
    mask_tst  = bert_word_mask[tst_df_ids]

    # dirout = f"{model.__class__.__name__}/fold{nhear}/"
    # os.system(f"mkdir -p {dirout}")
    print(f"{'#'*80}\n{'hearing':.<25}{nhear}\n"
          # f"{'dirout':.<25}{dirout}\n"
          f"{'ids trn shape':.<25}{ids_trn.shape}\n"
          f"{'ids dev shape':.<25}{ids_dev.shape}\n"
          f"{'ids tst shape':.<25}{ids_tst.shape}\n"
          f"{'mask trn shape':.<25}{mask_trn.shape}\n"
          f"{'mask dev shape':.<25}{mask_dev.shape}\n"
          f"{'mask tst shape':.<25}{mask_tst.shape}\n"
          f"{'y trn shape':.<25}{y_trn.shape}\n"
          f"{'y dev shape':.<25}{y_dev.shape}\n"
          f"{'y tst shape':.<25}{y_tst.shape}")
    dirout_results, preds, targs = proc.holdout(model, optimizer, [lossfunc],
                   [ids_trn, mask_trn], [ids_dev, mask_dev], [ids_tst, mask_tst],
                   [y_trn], [y_dev], [y_tst], [dtype_float], args.batsize, args.epochs, save=args.save, str_info=f"fold{nhear}")
    preds_all.extend(preds)
    targs_all.extend(targs)
proc.metrics(targs_all, preds_all)
ut.end(startime)


###################################################################################################


emb = np.load(args.path_fastext_emb)
pad_answer_ids  = load_npz(args.path_lemmas_ids_3015).toarray()
pad_answer_mask = load_npz(args.path_lemmas_mask_3015).toarray()
for nhear in range(1, 36):
    model = mod.StlPrembsBottleneckSigmoid(emb, trainable=args.trainable, nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
    # model = mod.StlPrembsEncSigmoidFc(emb, trainable=args.trainable, att_heads=args.att_heads, att_layers=args.att_layers, droprob=args.droprob, device=device)
    # model = mod.StlPrembsConvSigmoidFc(emb, trainable=args.trainable, conv_channels=args.conv_channels, filter_sizes=args.conv_filter_sizes, conv_stridesizes=args.conv_stridesizes, pool_filtersizes=args.pool_filtersizes, pool_stridesizes=args.pool_stridesizes, nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
    lossfunc = nn.BCELoss().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learate)
    hearing_tst = nhear
    hearing_dev = nhear + 1 if nhear != 35 else 1
    hearings_trn = [n for n in range(1, 36) if n not in [hearing_tst, hearing_dev]]

    trn_df_ids = df3015.index[df3015.nhear.isin(hearings_trn)]
    dev_df_ids = df3015.index[df3015.nhear == hearing_dev]
    tst_df_ids = df3015.index[df3015.nhear == hearing_tst]

    y_trn = df3015.label[trn_df_ids].replace({2: 1}).to_numpy()
    y_dev = df3015.label[dev_df_ids].replace({2: 1}).to_numpy()
    y_tst = df3015.label[tst_df_ids].replace({2: 1}).to_numpy()

    pad_trn_ids   = pad_answer_ids[trn_df_ids]
    pad_dev_ids   = pad_answer_ids[dev_df_ids]
    pad_tst_ids   = pad_answer_ids[tst_df_ids]

    pad_trn_mask   = pad_answer_mask[trn_df_ids]
    pad_dev_mask   = pad_answer_mask[dev_df_ids]
    pad_tst_mask   = pad_answer_mask[tst_df_ids]

    # dirout = f"{log.pathtime}{model.__class__.__name__}/fold{nhear}/"
    # os.system(f"mkdir -p {dirout}")
    print(f"{'#'*80}\n{'hearing':.<25}{nhear}\n"
          # f"{'dirout':.<25}{dirout}\n"
          f"{'emb shape':.<25}{emb.shape}\n"
          f"{'pad trn shape':.<25}{pad_trn_ids.shape}\n"
          f"{'pad dev shape':.<25}{pad_dev_ids.shape}\n"
          f"{'pad tst shape':.<25}{pad_tst_ids.shape}\n"
          f"{'y trn shape':.<25}{y_trn.shape}\n"
          f"{'y dev shape':.<25}{y_dev.shape}\n"
          f"{'y tst shape':.<25}{y_tst.shape}")
    dirout_results, preds, targs = proc.holdout(model, optimizer, [lossfunc],
                   # [pad_trn_ids, pad_trn_mask], [pad_dev_ids, pad_dev_mask], [pad_tst_ids, pad_tst_mask], # for models with Enc
                   [pad_trn_ids], [pad_dev_ids], [pad_tst_ids],
                   [y_trn], [y_dev], [y_tst], [dtype_float], args.batsize, args.epochs, save=args.save, str_info=f"fold{nhear}")
    preds_all.extend(preds)
    targs_all.extend(targs)
proc.metrics(targs_all, preds_all)
accu = round(accuracy_score(targs_all, preds_all) * 100, 2)
fmea = round(f1_score(targs_all, preds_all, average='macro') * 100, 2)
print(f"{'accuracy':.<11} {accu:<10}{'f1':.<5} {fmea:<10}")
ut.end(startime)


###################################################################################################


emb = np.load(args.path_fastext_emb)
assert (re.search('token', args.answer) and re.search('token', args.context)) or \
       (re.search('lemma', args.answer) and re.search('lemma', args.context)), 'tokens/lemmas non coerenti in answer/context'
pad_ids_6139  = load_npz(args.path_lemmas_ids_6139).toarray() if args.answer == 'lemmas' else load_npz(args.path_tokens_ids_6139).toarray()
pad_mask_6139 = load_npz(args.path_lemmas_mask_6139).toarray() if args.answer == 'lemmas' else load_npz(args.path_tokens_ids_6139).toarray()

# old #############################################################################################
# pseudo_context_row = [1] + [0] * (pad_ids_6139.shape[1] - 1)
# contexts_answers_df_ids = [df6139.index[max([0, i - args.hier_context]): i + 1].values for i in df6139.index if df6139.label[i] < 3]
# contexts_answers_ids  = np.array([np.concatenate([
#                                                  [pseudo_context_row] * (args.hier_context - len(pad_ids_6139[context_answer_df_ids]) + 1),
#                                                  pad_ids_6139[context_answer_df_ids]
#                                                  ], axis=0)
#                                                  if (args.hier_context - len(pad_ids_6139[context_answer_df_ids]) + 1) > 0 else
#                                                  pad_ids_6139[context_answer_df_ids]
#                                                  for context_answer_df_ids in contexts_answers_df_ids])
# contexts_answers_mask = np.array([np.concatenate([
#                                                  [pseudo_context_row] * (args.hier_context - len(pad_ids_6139[context_answer_df_ids]) + 1),
#                                                  pad_mask_6139[context_answer_df_ids]
#                                                  ], axis=0)
#                                                  if (args.hier_context - len(pad_ids_6139[context_answer_df_ids]) + 1) > 0 else
#                                                  pad_mask_6139[context_answer_df_ids]
#                                                  for context_answer_df_ids in contexts_answers_df_ids])
# print(f"contexts_answers_ids shape {contexts_answers_ids.shape}\ncontexts_answers_mask shape {contexts_answers_mask.shape}")
# exps ok #########################################################################################
pseudo_context_row = np.array([[1] + [0] * (pad_ids_6139.shape[1] - 1)])
pad_ids_6139 = np.concatenate((pad_ids_6139, pseudo_context_row), axis=0)
pad_mask_6139 = np.concatenate((pad_mask_6139, pseudo_context_row), axis=0)
prevutt_answer_df_idss      = [[i_df - i if i_df - i in df6139.index and (df6139.nturn[i_df - i] == df6139.nturn[i_df]) else 6139 for i in range(2)[::-1]]                 for i_df in df6139.index if df6139.role[i_df] == 'interviewee']
prevutts_answer_df_idss     = [[i_df - i if i_df - i in df6139.index and (df6139.nturn[i_df - i] == df6139.nturn[i_df]) else 6139 for i in range(args.hier_context)[::-1]] for i_df in df6139.index if df6139.role[i_df] == 'interviewee']
prevuttsturn_answer_df_idss = [[i_df - i if i_df - i in df6139.index and (df6139.nturn[i_df - i] <= df6139.nturn[i_df]) else 6139 for i in range(args.hier_context)[::-1]] for i_df in df6139.index if df6139.role[i_df] == 'interviewee']
prevturn_answer_df_idss     = [[6139] * (args.hier_context - len(df6139.index[(df6139.nhear == df6139.nhear[i_df]) & (df6139.nturn == df6139.nturn[i_df] - 1)][-args.hier_context:])) +
                                list(df6139.index[(df6139.nhear == df6139.nhear[i_df]) & (df6139.nturn == df6139.nturn[i_df] - 1)][-args.hier_context:]) # le ultime n frasi del turn, eventualmente paddate in testa con [6139]
                                for i_df in df6139.index if df6139.role[i_df] == 'interviewee']

contexts_answers_df_idss = prevutt_answer_df_idss      if (args.context == 'prev_speakerutt_tokens')           or (args.context == 'prev_speakerutt_lemmas')           else \
                           prevutts_answer_df_idss     if (args.context == 'prev_speakerutts_tokens')          or (args.context == 'prev_speakerutts_lemmas')          else \
                           prevuttsturn_answer_df_idss if (args.context == 'prev_speakerutts_and_turn_tokens') or (args.context == 'prev_speakerutts_and_turn_lemmas') else \
                           prevturn_answer_df_idss     if (args.context == 'prev_turn_tokens')                 or (args.context == 'prev_turn_lemmas')                 else None

contexts_answers_ids  = np.array([pad_ids_6139[contexts_answers_df_ids] for contexts_answers_df_ids in contexts_answers_df_idss])
contexts_answers_mask = np.array([pad_mask_6139[contexts_answers_df_ids] for contexts_answers_df_ids in contexts_answers_df_idss])
###################################################################################################


for nhear in range(1, 36):
    # model = mod.StlHierLstmFcSigmoid(emb.shape[0], 300, nrlayer=args.lstm_layers, hid_size=args.lstm_size, bidir=args.bidir,
    # out_fc_layers= 1, out_fc_outsize= 1,
    # droprob=args.droprob, device=device)
    model = mod.StlHierTransFcSigmoid(emb.shape[0], 300, att_heads=args.att_heads, att_layers=args.att_layers,
    nr_wrd_in_txt=contexts_answers_ids.shape[2], nr_txt_in_doc=contexts_answers_ids.shape[1],
    txt_fc_layers= 1, txt_fc_outsize= 30,
    doc_fc_layers= 1, doc_fc_outsize= 30,
    out_fc_layers= 1, out_fc_outsize= 1,
    droprob=args.droprob, device=device)
    # model = mod.StlPrembInf(emb, 1, nrlayer=args.lstm_layers, hid_size=args.lstm_size, att_heads=args.att_heads, att_layers=args.att_layers, narrowing_rate=args.narrowing_rate, device=device)
    lossfunc = nn.BCELoss().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learate)
    hearing_tst = nhear
    hearing_dev = nhear + 1 if nhear != 35 else 1
    hearings_trn = [n for n in range(1, 36) if n not in [hearing_tst, hearing_dev]]

    trn_df_ids = df3015.index[df3015.nhear.isin(hearings_trn)]
    dev_df_ids = df3015.index[df3015.nhear == hearing_dev]
    tst_df_ids = df3015.index[df3015.nhear == hearing_tst]

    y_trn = df3015.label[trn_df_ids].replace({2: 1}).to_numpy()
    y_dev = df3015.label[dev_df_ids].replace({2: 1}).to_numpy()
    y_tst = df3015.label[tst_df_ids].replace({2: 1}).to_numpy()

    contexts_answers_ids_trn   = contexts_answers_ids[trn_df_ids]
    contexts_answers_ids_dev   = contexts_answers_ids[dev_df_ids]
    contexts_answers_ids_tst   = contexts_answers_ids[tst_df_ids]

    contexts_answers_mask_trn   = contexts_answers_mask[trn_df_ids]
    contexts_answers_mask_dev   = contexts_answers_mask[dev_df_ids]
    contexts_answers_mask_tst   = contexts_answers_mask[tst_df_ids]

    # dirout = f"{log.pathtime}{model.__class__.__name__}/fold{nhear}/"
    # os.system(f"mkdir -p {dirout}")
    print(f"{'#'*80}\n{'hearing':.<25}{nhear}\n"
          # f"{'dirout':.<25}{dirout}\n"
          f"{'emb shape':.<25}{emb.shape}\n"
          f"{'answer ids trn shape':.<25}{contexts_answers_ids_trn.shape}\n"
          f"{'answer ids dev shape':.<25}{contexts_answers_ids_dev.shape}\n"
          f"{'answer ids tst shape':.<25}{contexts_answers_ids_tst.shape}\n"
          f"{'answer mask trn shape':.<25}{contexts_answers_mask_trn.shape}\n"
          f"{'answer mask dev shape':.<25}{contexts_answers_mask_dev.shape}\n"
          f"{'answer mask tst shape':.<25}{contexts_answers_mask_tst.shape}\n"
          f"{'y trn shape':.<25}{y_trn.shape}\n"
          f"{'y dev shape':.<25}{y_dev.shape}\n"
          f"{'y tst shape':.<25}{y_tst.shape}")
    dirout_results, preds, targs = proc.holdout(model, optimizer, [lossfunc],
                   [contexts_answers_ids_trn, contexts_answers_mask_trn], [contexts_answers_ids_dev, contexts_answers_mask_dev], [contexts_answers_ids_tst, contexts_answers_mask_tst],
                   [y_trn], [y_dev], [y_tst], [dtype_float], args.batsize, args.epochs, str_info=f"fold{nhear}")
    preds_all.extend(preds)
    targs_all.extend(targs)
proc.metrics(targs_all, preds_all)
accu = round(accuracy_score(targs_all, preds_all) * 100, 2)
fmea = round(f1_score(targs_all, preds_all, average='macro') * 100, 2)
print(f"{'accuracy':.<11} {accu:<10}{'f1':.<5} {fmea:<10}")
ut.end(startime)


