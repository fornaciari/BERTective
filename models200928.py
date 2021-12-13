# coding=latin-1
import util200818 as ut
import re, os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pylab as plt
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertTokenizer, BertModel, BertConfig, AutoModel, AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification, FlaubertTokenizer, FlaubertModel
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=(FutureWarning, UserWarning))
###################################################################################################


###################################################################################################


def norm(x):
    return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)


###################################################################################################
# Samuel Lynn-Evans
# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, device='cuda:0', max_seq_len=512):
        self.device = device
        super().__init__()
        self.d_model = d_model
        # create constant 'pe' matrix with values dependant on pos and i
        # cioè sequence_length (position) ed emb_size
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
# If you have parameters in your model, which should be saved and restored in the state_dict, but not trained by the optimizer,
# you should register them as buffers. Buffers won?t be returned in model.parameters(), so that the optimizer won?t have a chance to update them.

    def forward(self, x):
        x = x * math.sqrt(self.d_model) # make embeddings relatively larger
        seq_len = x.size(1) # add constant to embedding
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).to(device=self.device) # Variable ammette la back propagation
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, device='cuda:0', dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model).to(device=device)
        self.v_linear = nn.Linear(d_model, d_model).to(device=device)
        self.k_linear = nn.Linear(d_model, d_model).to(device=device)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model).to(device=device)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, device='cuda:0', d_ff=2048, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff).to(device=device)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model).to(device=device)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, device='cuda:0', eps=1e-6):
        super().__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size).to(device=device))
        self.bias = nn.Parameter(torch.zeros(self.size).to(device=device))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, device='cuda:0', dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model, device)
        self.norm_2 = Norm(d_model, device)
        self.attn = MultiHeadAttention(heads, d_model, device)
        self.ff = FeedForward(d_model, device)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm_1.forward(x)
        x = x + self.dropout_1(self.attn.forward(x2, x2, x2, mask))
        x2 = self.norm_2.forward(x)
        x = x + self.dropout_2(self.ff.forward(x2))
        return x


class Encoder(nn.Module):
    def __init__(self, rep_size, n_heads, n_layers, device='cuda:0'):
        super().__init__()
        self.n_layers = n_layers
        self.pe = PositionalEncoder(rep_size, device)
        self.layers = get_clones(EncoderLayer(rep_size, n_heads, device), n_layers)
        self.norm = Norm(rep_size, device)

    def forward(self, src, mask=None):
        x = self.pe.forward(src)
        for i in range(self.n_layers):
            x = self.layers[i](x, mask)
        return self.norm.forward(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, device='cuda:0', dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model).cuda()

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1.forward(x)
        x = x + self.dropout_1(self.attn_1.forward(x2, x2, x2, trg_mask))
        x2 = self.norm_2.forward(x)
        x = x + self.dropout_2(self.attn_2.forward(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3.forward(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, device='cuda:0'):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, device)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed.forward(trg)
        x = self.pe.forward(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm.forward(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads)
        self.decoder = Decoder(trg_vocab, d_model, N, heads)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder.forward(src, src_mask)
        d_output = self.decoder.forward(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output


###################################################################################################


class StlBertMatFcSigmoid(nn.Module):
    def __init__(self, lang, trainable=False, fc_layers=2, droprob=.2, device='cuda:0', floatype=torch.float32):
        super().__init__()
        self.lang = lang
        self.trainable = trainable

        # ml_config = BertConfig.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True)
        # self.bert = AutoModelWithLMHead.from_pretrained("bert-base-multilingual-cased", config=ml_config).to(device=device, dtype=floatype) if self.lang == 'ml' else \
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased").to(device=device, dtype=floatype)  if self.lang == 'ml' else \
                    BertModel.from_pretrained('bert-base-cased').to(device=device, dtype=floatype)               if self.lang == 'en' else \
                    AutoModel.from_pretrained('dbmdz/bert-base-italian-cased').to(device=device, dtype=floatype) if self.lang == 'it' else \
                    AutoModel.from_pretrained('dbmdz/bert-base-german-cased').to(device=device, dtype=floatype)  if self.lang == 'de' else \
                    FlaubertModel.from_pretrained('flaubert-base-cased').to(device=device, dtype=floatype)       if self.lang == 'fr' else \
                    BertModel.from_pretrained("bert-base-dutch-cased").to(device=device, dtype=floatype)         if self.lang == 'nl' else None
        dictkey = 'hidden_size' if self.lang != 'fr' else 'emb_dim'
        embsize = self.bert.config.to_dict()[dictkey] # 768

        layersizes = [(embsize if i == 1 else int(embsize * ((fc_layers - i + 1)/fc_layers)), int(embsize * ((fc_layers - i)/fc_layers)) if i != fc_layers else 1) for i in range(1, fc_layers+1)]
        self.fc_layers = nn.ModuleList(nn.Linear(layersizes[il][0], layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}, trainable: {}".format(name_str, str(param.shape), param.numel(), param.requires_grad))
        print(f'The model has {sum(p.numel() for p in self.parameters() if p.requires_grad):,} trainable parameters')

    def forward(self, text, mask):
        with torch.set_grad_enabled(self.trainable):
            lookup = self.bert(input_ids=text, attention_mask=mask)[0]
        for layer in self.fc_layers:
            lookup = layer(lookup)
            # print(lookup.shape)
        out = lookup.mean(1)
        out = self.dropout(out)
        out = F.sigmoid(out).squeeze(1)
        return out


class StlBertMatAttFcSigmoid(nn.Module):
    def __init__(self, lang, att_heads, att_layers, trainable=False, fc_layers=2, droprob=.2, device='cuda:0', floatype=torch.float32):
        super().__init__()
        self.lang = lang
        self.trainable = trainable

        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased").to(device=device, dtype=floatype)  if self.lang == 'ml' else \
                    BertModel.from_pretrained('bert-base-cased').to(device=device, dtype=floatype)               if self.lang == 'en' else \
                    AutoModel.from_pretrained('dbmdz/bert-base-italian-cased').to(device=device, dtype=floatype) if self.lang == 'it' else \
                    AutoModel.from_pretrained('dbmdz/bert-base-german-cased').to(device=device, dtype=floatype)  if self.lang == 'de' else \
                    FlaubertModel.from_pretrained('flaubert-base-cased').to(device=device, dtype=floatype)       if self.lang == 'fr' else \
                    BertModel.from_pretrained("bert-base-dutch-cased").to(device=device, dtype=floatype)         if self.lang == 'nl' else None
        dictkey = 'hidden_size' if self.lang != 'fr' else 'emb_dim'
        embsize = self.bert.config.to_dict()[dictkey] # 768

        self.encoder = Encoder(embsize, att_heads, att_layers, device=device)

        layersizes = [(embsize if i == 1 else int(embsize * ((fc_layers - i + 1)/fc_layers)), int(embsize * ((fc_layers - i)/fc_layers)) if i != fc_layers else 1) for i in range(1, fc_layers+1)]
        self.fc_layers = nn.ModuleList(nn.Linear(layersizes[il][0], layersizes[il][1]) for il in range(fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}, trainable: {}".format(name_str, str(param.shape), param.numel(), param.requires_grad))
        print(f'The model has {sum(p.numel() for p in self.parameters() if p.requires_grad):,} trainable parameters')

    def forward(self, text, mask):
        with torch.set_grad_enabled(self.trainable):
            lookup = self.bert(input_ids=text, attention_mask=mask)[0]
        lookup = self.encoder.forward(lookup)
        for layer in self.fc_layers:
            lookup = layer(lookup)
            # print(lookup.shape)
        out = lookup.mean(1)
        out = self.dropout(out)
        out = F.sigmoid(out).squeeze(1)
        return out


class StlPrembsBottleneckSigmoid(nn.Module):
    def __init__(self, emb, trainable=False, nrlayer=2, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        emb = torch.from_numpy(emb).to(device=device, dtype=floatype)
        self.embs = nn.Embedding.from_pretrained(emb)
        self.embs.weight.requires_grad = True if trainable else False
        embsize = self.embs.embedding_dim
        layersizes = [(embsize if i == 1 else int(embsize * ((nrlayer - i + 1)/(nrlayer))), int(embsize * ((nrlayer - i)/(nrlayer))) if i != (nrlayer) else 1) for i in range(1, nrlayer+1)]
        self.fc_layers = nn.ModuleList(nn.Linear(layersizes[il][0], layersizes[il][1]) for il in range(nrlayer)).to(device=device)
        self.dropout = nn.Dropout(droprob)
        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}, trainable: {}".format(name_str, str(param.shape), param.numel(), param.requires_grad))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, text):
        lookup = self.embs(text)
        # print(lookup.shape)
        for layer in self.fc_layers:
            lookup = layer(lookup)
            # print(lookup.shape)
        out = lookup.mean(1)
        out = self.dropout(out)
        out = F.sigmoid(out)
        out = out.squeeze(1)
        # print(out.shape, out)
        return out


class StlPrembsEncSigmoidFc(nn.Module):
    def __init__(self, emb, trainable=False, att_heads=2, att_layers=2, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        emb = torch.from_numpy(emb).to(device=device, dtype=floatype)
        self.embs = nn.Embedding.from_pretrained(emb)
        self.embs.weight.requires_grad = True if trainable else False
        embsize = self.embs.embedding_dim

        self.encoder = Encoder(embsize, att_heads, att_layers, device=device)

        self.fc = nn.Linear(embsize, 1).to(device=device)
        self.dropout = nn.Dropout(droprob)
        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}, trainable: {}".format(name_str, str(param.shape), param.numel(), param.requires_grad))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, text, mask):
        lookup = self.embs(text)
        # print(lookup.shape, mask.unsqueeze(1).shape)
        lookup = self.encoder.forward(lookup, mask.unsqueeze(1))
        # lookup = self.encoder.forward(lookup, None)
        out = self.fc(lookup).mean(1)
        out = self.dropout(out)
        out = F.sigmoid(out).squeeze(1)
        return out


class StlPrembsConvSigmoidFc(nn.Module):
    def __init__(self, emb, trainable=False, conv_channels=[16, 32], filter_sizes=[3, 4, 5], conv_stridesizes=[1, 1], pool_filtersizes=[2, 2], pool_stridesizes=[1, 1], nrlayer=2, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        emb = torch.from_numpy(emb).to(device=device, dtype=floatype)
        self.embs = nn.Embedding.from_pretrained(emb)
        self.embs.weight.requires_grad = True if trainable else False
        emb_size = self.embs.embedding_dim
        self.pool_filtersizes = pool_filtersizes
        self.pool_stridesizes = pool_stridesizes
        self.nconv = len(conv_channels)
        self.nfilt = len(filter_sizes)
        self.ilastconv = self.nconv - 1
        self.convs = nn.ModuleList(nn.ModuleList(nn.Conv1d(in_channels=emb_size, out_channels=conv_channels[iconv], kernel_size=filter_sizes[ifilt], stride=conv_stridesizes[iconv])
                        for ifilt in range(self.nfilt))
                        for iconv in range(self.nconv)).to(device=device, dtype=floatype)
        convout_size = conv_channels[-1] * self.nfilt # conv size out = len last conv channel * nr of filters, infatti concatenerò i vettori in uscita di ogni filtro, che hanno il size dell'ultimo channel
        layersizes = [(convout_size if i == 1 else int(convout_size * ((nrlayer - i + 1)/(nrlayer))), int(convout_size * ((nrlayer - i)/(nrlayer))) if i != (nrlayer) else 1) for i in range(1, nrlayer+1)]
        self.fc_layers = nn.ModuleList(nn.Linear(layersizes[il][0], layersizes[il][1]) for il in range(nrlayer)).to(device=device)
        self.dropout = nn.Dropout(droprob)
        for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def conv_block(self, embs_lookup):
        conveds = [[F.relu(self.convs[iconv][ifilt](embs_lookup))
                    for ifilt in range(self.nfilt)]
                    for iconv in range(self.nconv)] # [batsize, nr_filters, sent len - filter_sizes[n] + 1]
        pool_filters = [[self.pool_filtersizes[iconv] if iconv != self.ilastconv else
                        conveds[iconv][ifilter].shape[2] # poiché voglio che esca un vettore, assegno all'ultimo filtro la stessa dim della colonna in input,
                        for ifilter in range(self.nfilt)] # così il nr delle colonne in uscita all'ultima conv sarà 1, ed eliminerò la dimensione con squueze.
                        for iconv in range(self.nconv)]
        pooleds = [[F.max_pool1d(conveds[iconv][ifilt], pool_filters[iconv][ifilt], stride=self.pool_stridesizes[iconv]) if iconv != self.ilastconv else
                    F.max_pool1d(conveds[iconv][ifilt], pool_filters[iconv][ifilt], stride=self.pool_stridesizes[iconv]).squeeze(2)
                    for ifilt in range(self.nfilt)]
                    for iconv in range(self.nconv)] # [batsize, nr_filters]
        concat = self.dropout(torch.cat([pooled for pooled in pooleds[self.ilastconv]], dim=1)) # [batsize, nr_filters * len(filter_sizes)]
        # for iconv in range(len(args.conv_channels)):
        #     for ifilter in range(len(args.filter_sizes)):
        #         print("$$$ conveds", conveds[iconv][ifilter].shape)
        # for iconv in range(len(args.conv_channels)):
        #     for ifilter in range(len(args.filter_sizes)):
        #         print("$$$ pooleds", pooleds[iconv][ifilter].shape)
        # print("$$$ concat", concat.shape)
        return concat

    def forward(self, text):
        lookup  = self.embs(text)
        lookup = lookup.permute(0, 2, 1)
        out_conv = self.conv_block(lookup)
        for layer in self.fc_layers:
            out_conv = layer(out_conv)
        out_conv = self.dropout(out_conv)
        out = F.sigmoid(out_conv).squeeze(1)
        return out


class StlHierTransFcSigmoid(nn.Module): # bello e inutile
    def __init__(self, vocsize, embsize, att_heads, att_layers, nr_wrd_in_txt, nr_txt_in_doc, txt_fc_layers=1, txt_fc_outsize=10, doc_fc_layers=1, doc_fc_outsize=10, out_fc_layers=1, out_fc_outsize=1, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()

        def layersizes(insize, outsize, layers):
            return [(insize if i == 1 else int((insize + outsize) * ((layers - i + 1)/layers)), int((insize + outsize) * ((layers - i)/layers)) if i != layers else outsize) for i in range(1, layers+1)]

        self.emb = nn.Embedding(vocsize, embsize, padding_idx=0).to(device=device, dtype=floatype)
        # self.emb = nn.Embedding(vocsize, embsize).to(device=device, dtype=floatype)

        self.txt_encoder = Encoder(embsize, att_heads, att_layers, device=device)

        txt_layersizes = layersizes(embsize, txt_fc_outsize, txt_fc_layers)
        self.txt_fc_layers = nn.ModuleList(nn.Linear(txt_layersizes[il][0], txt_layersizes[il][1]) for il in range(txt_fc_layers)).to(device=device)
        reshaped_txt_fc = nr_wrd_in_txt * txt_fc_outsize

        self.doc_encoder = Encoder(reshaped_txt_fc, att_heads, att_layers, device=device)

        doc_layersizes = layersizes(reshaped_txt_fc, doc_fc_outsize, doc_fc_layers)
        self.doc_fc_layers = nn.ModuleList(nn.Linear(doc_layersizes[il][0], doc_layersizes[il][1]) for il in range(doc_fc_layers)).to(device=device)
        reshaped_doc_fc = nr_txt_in_doc * doc_fc_outsize

        out_layersizes = layersizes(reshaped_doc_fc, out_fc_outsize, out_fc_layers)
        self.out_fc_layers = nn.ModuleList(nn.Linear(out_layersizes[il][0], out_layersizes[il][1]) for il in range(out_fc_layers)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f'The model has {sum(p.numel() for p in self.parameters() if p.requires_grad):,} trainable parameters')

    def forward(self, text, mask):
        # print()
        # print(text.shape, mask.shape)

        # lookup = self.emb(text)
        # # print(lookup.shape)
        # # print(lookup.view(-1, lookup.shape[2], lookup.shape[3]).shape, mask.view(-1, mask.shape[2]).unsqueeze(1).shape, "$$")
        # txt = self.txt_encoder.forward(lookup.view(-1, lookup.shape[2], lookup.shape[3]), mask.view(-1, mask.shape[2]).unsqueeze(1)).view(lookup.shape)
        # # print(txt.shape, "$")
        # for layer in self.txt_fc_layers:
        #     txt = layer(txt)
        #     # print(txt.shape, "$")

        # print(txt.shape, "$$")
        # doc = txt.view(-1, txt.shape[1], txt.shape[2] * txt.shape[3])
        # print(doc.shape)
        # doc = torch.mean(txt, dim=2)
        # print(doc.shape)
        # doc = torch.mean(txt, dim=3)
        # print(doc.shape)
        #
        # print("#####")
        txt = torch.stack([torch.stack([self.txt_encoder.forward(self.emb(text[ib, it, :]).unsqueeze(0), mask[ib, it, :].unsqueeze(0)).squeeze(0)
               for it in range(text.shape[1])])
               for ib in range(text.shape[0])])
        # print(txt.shape, "$$$")
        for layer in self.txt_fc_layers:
            txt = layer(txt)
            # print(txt.shape, "$")

        doc = txt.view(-1, txt.shape[1], txt.shape[2] * txt.shape[3])
        # print(doc.shape)

        doc = self.doc_encoder.forward(doc)#.mean(1) # [batsize, embsize]
        # print(doc.shape)

        for layer in self.doc_fc_layers:
            doc = layer(doc)
            # print(doc.shape, "$")

        out = doc.view(-1, doc.shape[1] * doc.shape[2])
        # print(out.shape)

        for layer in self.out_fc_layers:
            out = layer(out)
            # print(out.shape, "$")

        out = self.dropout(out)
        out = F.sigmoid(out).squeeze(1)
        return out

    # def forward(self, text, mask):
    #     # txt = list()
    #     # print()
    #     # for ib in range(text.shape[0]):
    #     #     texts = list()
    #     #     for it in range(text.shape[1]):
    #     #         textemb = self.emb(text[ib, it, :]).unsqueeze(0) # simulo batch a 1: [1, padsize, embsize]
    #     #         print(textemb.shape, "$$")
    #     #         textmask = mask[ib, it, :].unsqueeze(0)
    #     #         textencoded = self.txt_encoder.forward(textemb, textmask).squeeze(0).mean(0) # tolgo il finto batch e faccio la media sulle righe: [embsize]
    #     #         print(textencoded.shape, "$$")
    #     #         texts.append(textencoded)
    #     #     texts = torch.stack(texts) # [docsize, embsize]
    #     #     txt.append(texts)
    #     # txt = torch.stack(txt) # [batsize, docsize, embsize]
    #     # print()
    #     # print(text.shape, mask.shape)
    #     txt = torch.stack([torch.stack([self.txt_encoder.forward(self.emb(text[ib, it, :]).unsqueeze(0), mask[ib, it, :].unsqueeze(0)).squeeze(0)
    #            for it in range(text.shape[1])])
    #            for ib in range(text.shape[0])])
    #     # print(txt.shape)
    #     for layer in self.txt_fc_layers:
    #         txt = layer(txt)
    #         # print(txt.shape, "$")
    #
    #     doc = txt.view(-1, txt.shape[1], txt.shape[2] * txt.shape[3])
    #     # print(doc.shape)
    #
    #     doc = self.doc_encoder.forward(doc)#.mean(1) # [batsize, embsize]
    #     # print(doc.shape)
    #
    #     for layer in self.doc_fc_layers:
    #         doc = layer(doc)
    #         # print(doc.shape, "$")
    #
    #     out = doc.view(-1, doc.shape[1] * doc.shape[2])
    #     # print(out.shape)
    #
    #     for layer in self.out_fc_layers:
    #         out = layer(out)
    #         # print(out.shape, "$")
    #
    #     out = self.dropout(out)
    #     out = F.sigmoid(out).squeeze(1)
    #     return out




