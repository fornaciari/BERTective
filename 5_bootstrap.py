# coding=latin-1
import util200818 as ut
import step200928 as st
import models201001 as mod
import argparse, os, re, sys, time
import pandas as pd
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, precision_recall_fscore_support, log_loss, confusion_matrix
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
parser.add_argument("-n_loops",      type=int,   default=1000)
parser.add_argument("-perc_sample", type=float, default=.5)
args = parser.parse_args()
sys.stdout = sys.stderr = log = ut.log(__file__, ut.__file__, st.__file__, mod.__file__)
startime = ut.start()
align_size = ut.print_args(args)
print(f"{'dirout':.<{align_size}} {log.pathtime}")
###################################################################################################

path_targs                 = "0_data/targs.txt"
path_fake_preds            = "0_data/fake_preds.txt"
path_mv_preds              = "0_data/mv_preds.txt"

cond2dir = {
'dir_mlp_randemb_6145':       "exp200907decour/mlp2_randemb_6145_200921153815/",
'dir_mlp_fastemb_6170':       "exp200907decour/mlp2_fatembtrain_6170_200921154310/",
'dir_conv_randemb_6584':      "exp200907decour/conv_randemb_2_4_6_32_64_128_6584_200908112830/",
'dir_conv_fastemb_6506':      "exp200907decour/conv_fastembtrain_2_4_6_32_64_128_6506_201001085948/",
'dir_trans_randemb_6564':     "exp200907decour/enc_randemb_2_2_6564_200907183252/",
'dir_trans_fastemb_6652':     "exp200907decour/enc_fastemb_2_2_6652_200907205845/",
'dir_trans_context1_6425':    "exp200929decour/StlHierTransFcSigmoid6-6_context1_6425_201001053530/",
'dir_trans_context2_6329':    "exp200929decour/StlHierTransFcSigmoid6-6_context2_6329_201001053640/",
'dir_trans_context3_6375':    "exp200929decour/StlHierTransFcSigmoid6-6_context3_6375_201001085732/",
'dir_trans_utt_6454':         "exp200929decour/StlHierTransFcSigmoid6-6_utt_6454_201001195149/",
'dir_trans_utts_6426':        "exp200929decour/StlHierTransFcSigmoid6-6_utts_6426_201001195212/",
'dir_trans_utturn_6405':      "exp200929decour/StlHierTransFcSigmoid6-6_uttsturn_6405_201001195234/",
'dir_trans_turn_4617':        "exp200929decour/StlHierTransFcSigmoid6-6_turn_4617_201002104849/",
'dir_bert_dense_single_4560': "exp200907decour/bert_single_4560_200916131256/",
'dir_bert_trans_single_6657': "exp200907decour/bertatt_6_6_single_6657_200916130957/",
'dir_bert_trans_utt1_6497':   "exp200929decour/StlBertMatAttFcSigmoid6-6_utt1_6497_201002225710/",
'dir_bert_trans_utt2_6298':   "exp200929decour/StlBertMatAttFcSigmoid6-6_utt2_6298_201002230048/",
'dir_bert_trans_utt3_6155':   "exp200929decour/StlBertMatAttFcSigmoid6-6_utt3_6155_201002230120/",
'dir_bert_trans_utt_6719':    "exp200907decour/bertatt_6_6_speakerutt_6719_200907221938/",
'dir_bert_trans_utts_6663':   "exp200907decour/bertatt_6_6_speakerutts_6663_200908084021/",
'dir_bert_trans_utturn_6242': "exp200907decour/bertatt_6_6_speakerutturn_6242_200907222122/",
'dir_bert_trans_turn_6467':   "exp200907decour/bertatt_6_6_turn_6467_200907222016/"
}


bootdata = {'holdout': # defaultdict contiene la condizione sperimentale, le liste gli output di ogni esperimento
                       {'control':   defaultdict(lambda: {'dirs': list(), 'preds': list(), 'targs': list()}),
                        'treatment': defaultdict(lambda: {'dirs': list(), 'preds': list(), 'targs': list()})},
            'crossval':
                      {'control':   defaultdict(lambda: {'dirs': list(), 'preds': list(), 'targs': list()}),
                       'treatment': defaultdict(lambda: {'dirs': list(), 'preds': list(), 'targs': list()})}}


def bootstrap(data, n_loops, perc_sample=.1, verbose=False):
    """
    :param data:

      bootinput = {'holdout':
                            {'control':   defaultdict(lambda: {'dirs': list(), 'preds': list(), 'targs': list()}),
                             'treatment': defaultdict(lambda: {'dirs': list(), 'preds': list(), 'targs': list()})},
                   'crossval':
                            {'control':   defaultdict(lambda: {'dirs': list(), 'preds': list(), 'targs': list()}),
                             'treatment': defaultdict(lambda: {'dirs': list(), 'preds': list(), 'targs': list()})}}
    """

    startime = ut.start()

    def metrics(targs, control_preds, treatment_preds):
        rounding_value = 4
        control_acc    = round(accuracy_score(targs, control_preds) * 100, 2)
        control_f1     = round(f1_score(targs, control_preds, average='macro') * 100, 2)
        control_prec   = round(precision_score(targs, control_preds, average='macro') * 100, 2)
        control_rec    = round(recall_score(targs, control_preds, average='macro') * 100, 2)
        treatment_acc  = round(accuracy_score(targs, treatment_preds) * 100, 2)
        treatment_f1   = round(f1_score(targs, treatment_preds, average='macro') * 100, 2)
        treatment_prec = round(precision_score(targs, treatment_preds, average='macro') * 100, 2)
        treatment_rec  = round(recall_score(targs, treatment_preds, average='macro') * 100, 2)
        control_conf_matrix = confusion_matrix(targs, control_preds)
        # control_tn0 = control_conf_matrix[1, 1]
        # control_fn0 = control_conf_matrix[0, 1]
        # control_fp0 = control_conf_matrix[1, 0]
        # control_tp0 = control_conf_matrix[0, 0]
        # control_prec0 = round(control_tp0 / (control_tp0 + control_fp0) * 100, 2)
        # control_rec0  = round(control_tp0 / (control_tp0 + control_fn0) * 100, 2)
        treatment_conf_matrix = confusion_matrix(targs, treatment_preds)
        # treatment_tn0 = treatment_conf_matrix[1, 1]
        # treatment_fn0 = treatment_conf_matrix[0, 1]
        # treatment_fp0 = treatment_conf_matrix[1, 0]
        # treatment_tp0 = treatment_conf_matrix[0, 0]
        # treatment_prec0 = round(treatment_tp0 / (treatment_tp0 + treatment_fp0) * 100, 2)
        # treatment_rec0  = round(treatment_tp0 / (treatment_tp0 + treatment_fn0) * 100, 2)
        diff_acc  = round(treatment_acc - control_acc, 2)
        diff_f1   = round(treatment_f1  - control_f1, 2)
        diff_prec = round(treatment_prec - control_prec, 2)
        diff_rec  = round(treatment_rec  - control_rec, 2)

        return control_acc, treatment_acc, diff_acc, control_f1, treatment_f1, diff_f1, control_prec, treatment_prec, diff_prec, control_rec, treatment_rec, diff_rec

    df = pd.DataFrame(columns="acc prec rec f1".split())
    print(df)
    for val in data:
        for control_cond in data[val]['control']:
            print('#'*120)
            control_preds_all, control_targs_all, control_acc_all, control_f1_all = list(), list(), list(), list()
            for dire, preds, targs in zip(data[val]['control'][control_cond]['dirs'], data[val]['control'][control_cond]['preds'], data[val]['control'][control_cond]['targs']):
                acc = round(accuracy_score(targs, preds) * 100, 2)
                f1  = round(f1_score(targs, preds, average='macro') * 100, 2)
                control_preds_all.extend(preds)
                control_targs_all.extend(targs)
                # control_acc_all.append(acc)
                # control_f1_all.append(f1)
                if verbose:  print(f"{'control dir':.<25} {dire:.<50} accuracy {acc:<8} F-measure {f1}")
            for treatment_cond in data[val]['treatment']:
                print(f"{'#'*80}\n{val:<12}{control_cond}   vs   {treatment_cond}")
                treatment_preds_all, treatment_targs_all, treatment_acc_all, treatment_f1_all = list(), list(), list(), list()
                for dire, preds, targs in zip(data[val]['treatment'][treatment_cond]['dirs'],
                                              data[val]['treatment'][treatment_cond]['preds'],
                                              data[val]['treatment'][treatment_cond]['targs']):
                    acc = round(accuracy_score(targs, preds) * 100, 2)
                    f1  = round(f1_score(targs, preds, average='macro') * 100, 2)
                    treatment_preds_all.extend(preds)
                    treatment_targs_all.extend(targs)
                    # treatment_acc_all.append(acc)
                    # treatment_f1_all.append(f1)
                    if verbose: print(f"{'treatment dir':.<25} {dire:.50} accuracy {acc:<8} F-measure {f1}")
                assert control_targs_all == treatment_targs_all
                targs_all = control_targs_all
                tot_control_acc, tot_treatment_acc, tot_diff_acc, tot_control_f1, tot_treatment_f1, tot_diff_f1, tot_control_prec0, tot_treatment_prec0, tot_diff_prec0, tot_control_rec0, tot_treatment_rec0, tot_diff_rec0 = metrics(targs_all, control_preds_all, treatment_preds_all)
                print(f"{'control total accuracy':.<25} {tot_control_acc:<8} {'treatment total accuracy':.<30} {tot_treatment_acc:<8} {'diff':.<7} {tot_diff_acc}")
                print(f"{'control total F-measure':.<25} {tot_control_f1:<8} {'treatment total F-measure':.<30} {tot_treatment_f1:<8} {'diff':.<7} {tot_diff_f1}")
                print(f"{'control total precision 0':.<25} {tot_control_prec0:<8} {'treatment total precision 0':.<30} {tot_treatment_prec0:<8} {'diff':.<7} {tot_diff_prec0}")
                print(f"{'control total recall 0':.<25} {tot_control_rec0:<8} {'treatment total recall 0':.<30} {tot_treatment_rec0:<8} {'diff':.<7} {tot_diff_rec0}")
                if control_cond not in df.index:
                    df = df.append(pd.Series({"acc": tot_control_acc, "prec": tot_control_prec0, "rec": tot_control_rec0, "f1": tot_control_f1}, name=control_cond))
                if treatment_cond[4:-5] not in df.index:
                    df = df.append(pd.Series({"acc": tot_treatment_acc, "prec": tot_treatment_prec0, "rec": tot_treatment_rec0, "f1": tot_treatment_f1}, name=treatment_cond[4:-5]))

                tst_overall_size = len(targs_all)
                # estraggo l'equivalente di un esperimento. Più è piccolo il numero, più è facile avere significatività. In altre parole, più esperimenti si fanno più è facile
                samplesize = int(len(targs_all) * perc_sample)
                print(f"{'tot experiments size':.<25} {tst_overall_size}\n{'sample size':.<25} {samplesize}")
                twice_diff_acc  = 0
                twice_diff_f1   = 0
                twice_diff_prec = 0
                twice_diff_rec  = 0
                for loop in tqdm(range(n_loops), desc='bootstrap', ncols=80):
                    i_sample = np.random.choice(range(tst_overall_size), size=samplesize, replace=True)
                    sample_control_preds   = [control_preds_all[i]   for i in i_sample]
                    sample_treatment_preds = [treatment_preds_all[i] for i in i_sample]
                    sample_targs           = [targs_all[i]           for i in i_sample]
                    _, _, sample_diff_acc, _, _, sample_diff_f1, _, _, sample_diff_prec, _, _, sample_diff_rec = metrics(sample_targs, sample_control_preds, sample_treatment_preds)
                    if sample_diff_acc    > 2 * tot_diff_acc:    twice_diff_acc += 1
                    if sample_diff_f1     > 2 * tot_diff_f1:     twice_diff_f1 += 1
                    if sample_diff_prec  > 2 * tot_diff_prec0:  twice_diff_prec += 1
                    if sample_diff_rec   > 2 * tot_diff_rec0:   twice_diff_rec += 1
                str_out = f"{'count sample diff acc   is twice tot diff acc':.<50} {twice_diff_acc:<5}/ {n_loops:<8}p < {round((twice_diff_acc / n_loops), 4):<6} {ut.bcolors.red}{'**' if twice_diff_acc / n_loops < 0.01 else '*' if twice_diff_acc / n_loops < 0.05 else ''}{ut.bcolors.end}\n" \
                          f"{'count sample diff f1    is twice tot diff f1':.<50} {twice_diff_f1:<5}/ {n_loops:<8}p < {round((twice_diff_f1 / n_loops), 4):<6} {ut.bcolors.red}{'**' if twice_diff_f1 / n_loops < 0.01 else '*' if twice_diff_f1 / n_loops < 0.05 else ''}{ut.bcolors.end}\n" \
                          f"{'count sample diff prec0 is twice tot diff prec0':.<50} {twice_diff_prec:<5}/ {n_loops:<8}p < {round((twice_diff_prec / n_loops), 4):<6} {ut.bcolors.red}{'**' if twice_diff_prec / n_loops < 0.01 else '*' if twice_diff_prec / n_loops < 0.05 else ''}{ut.bcolors.end}\n" \
                          f"{'count sample diff rec0  is twice tot diff rec0 ':.<50} {twice_diff_rec:<5}/ {n_loops:<8}p < {round((twice_diff_rec / n_loops), 4):<6} {ut.bcolors.red}{'**' if twice_diff_rec / n_loops < 0.01 else '*' if twice_diff_rec / n_loops < 0.05 else ''}{ut.bcolors.end}"
                print(str_out)
                ut.sendslack(f"{val:<12}{control_cond}   vs   {treatment_cond}\n{str_out}")

    df.to_csv(f"{log.pathtime}df.csv")
    df = pd.read_csv(f"{log.pathtime}df.csv", index_col=0)
    print(df)
    ut.end(startime)
    return df


base_targs = ut.file2list(path_targs, filenc='utf-8', elemtype='int', sep="\n", emptyend=False)
fake_preds = ut.file2list(path_fake_preds, filenc='utf-8', elemtype='int', sep="\n", emptyend=False)
mv_preds   = ut.file2list(path_mv_preds, filenc='utf-8', elemtype='int', sep="\n", emptyend=False)
# print(type(base_targs), len(base_targs))
# print(type(fake_preds), len(fake_preds))
# print(type(mv_preds),   len(mv_preds))

bootdata['holdout']['control']['simul_svm'] = {'dirs': ['0_data/'], 'preds': [fake_preds], 'targs': [base_targs]}
bootdata['holdout']['control']['mv']        = {'dirs': ['0_data/'], 'preds': [mv_preds], 'targs': [base_targs]}


for cond in cond2dir:
    dirs = [f"{cond2dir[cond]}/{d}/" for d in os.listdir(cond2dir[cond]) if os.path.isdir(f"{cond2dir[cond]}/{d}/") and not re.match('__', d)]
    cond_targs = [targ for d in dirs for targ in ut.file2list(f"{d}targs.txt", filenc='utf-8', elemtype='int', sep="\n", emptyend=False)]
    cond_preds = [pred for d in dirs for pred in ut.file2list(f"{d}preds.txt", filenc='utf-8', elemtype='int', sep="\n", emptyend=False)]
    bootdata['holdout']['treatment'][cond] = {'dirs': [cond2dir[cond]], 'preds': [cond_preds], 'targs': [cond_targs]}
    # print(cond, type(cond_targs), type(cond_preds), len(cond_targs), len(cond_preds))

ut.writejson(bootdata, log.pathtime + 'bootstrap.json')
bootstrap(bootdata, n_loops=args.n_loops, perc_sample=args.perc_sample, verbose=True)
ut.end(startime)
