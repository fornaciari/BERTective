# coding=latin-1
import util200602 as ut
import argparse, os, re, sys, time, io
import numpy as np
###################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--en_crawl_embs", default='crawl-300d-2M.vec')
parser.add_argument("--en_embs", default='fasttext/cc.en.300.vec')
parser.add_argument("--it_embs", default='fasttext/cc.it.300.vec')
parser.add_argument("--fr_embs", default='fasttext/cc.fr.300.vec')
parser.add_argument("--de_embs", default='fasttext/cc.de.300.vec')
parser.add_argument("--nl_embs", default='fasttext/cc.nl.300.vec')
parser.add_argument("--da_embs", default='fasttext/cc.da.300.vec')
parser.add_argument("--sv_embs", default='fasttext/cc.sv.300.vec')
parser.add_argument("--max_vocsize", default=1000000)
args = parser.parse_args()
sys.stdout = sys.stderr = log = ut.log(__file__)
startime = ut.start()
ut.print_args(args)
print("dirout:\t", log.pathtime, "\n" + '#'*80)
###################################################################################################
# file_handler = io.open("cc.it.300.vec", 'r', encoding='utf-8', newline='\n', errors='ignore') # è già un generator
# emb_generator = (line for line in file_handler)
# print(file_handler, sys.getsizeof(emb_generator), sys.getsizeof(file_handler))

lang2path = {#'en_crawl': args.en_crawl_embs,
             # 'en': args.en_embs,
             'it': args.it_embs,
             # 'fr': args.fr_embs,
             # 'de': args.de_embs,
             # 'nl': args.nl_embs,
             # 'da': args.da_embs,
             # 'sv': args.sv_embs,
             }

for lang, pathfile in lang2path.items():
    with open(pathfile, 'r', encoding='utf-8', newline='\n', errors='ignore') as file_handler: # è già un generator
        vocsize, embsize = map(int, file_handler.__next__().split()) # la prima riga del file contiene il nr delle righe e la dim degli emb
        print("{} - voc size: {} - emb size: {}".format(lang, vocsize, embsize))
        w2i = dict()
        emb = list()
        for index in range(args.max_vocsize):
            try:
                row = file_handler.__next__().split() # le righe successive alla prima contengono una stringa con parola e valori dell'embedding, separati da spazio
                w2i[row[0]] = index
                emb.append(list(map(float, row[1:])))
                ut.say_progress(index)
            except StopIteration:
                print("Iteration ended at index", index)
                break
        print()
        ut.writejson(w2i, log.pathtime + lang + '_w2i.json')
        ut.writejson(emb, log.pathtime + lang + '_emb.json')
        print(len(w2i), np.shape(emb))

# a = ut.readjson(log.pathtime + 'it_w2i.json')
# b = ut.readjson(log.pathtime + 'it_emb.json')
# ut.printjson(a)
# print(type(a))
# ut.printjson(b)
# print(type(b))
ut.end(startime)
