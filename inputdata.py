import collections
import numpy as np

import math
import os
import random
import nltk
import json, operator

class Options(object):
    def __init__(self, embeds, netfile, netlist, rellist, lamtas, test_model=False, sampletype='geo'):
        self.wordindex = self.loaddict(netlist, test_model)
        self.reldict = self.loaddict2(rellist)
        self.sampletype = sampletype

        self.vocab_size = len(self.wordindex)
        self.rel_size = len(self.reldict)
        print('vocab_size: {}, relation_size: {}'.format(self.vocab_size, self.rel_size))

        self.id2word = {v: k for k, v in self.wordindex.items()}
        self.id2rel = {v: k for k, v in self.reldict.items()}
        subsampled_datas  = []
        meanlist = []
        for net in netfile:
            triples_id, self.count, self.frequency, meanscore = self.loadgraph(net)        
            subsampled_datas.append(self.subsampling(triples_id))
            meanlist.append(meanscore)
        benchmark = meanlist[0]
        self.subsampled_data = subsampled_datas[0]
        for sub_datam,ms in zip(subsampled_datas[1:], meanlist[1:]):
            div = ms/benchmark
            divdata = []
            for trip in sub_datam:
                a,b,c,d = trip[0], trip[1],trip[2], trip[3]/div
                divdata.append((a,b,c,d))
            self.subsampled_data += divdata
        self.meanscore = benchmark
        self.subsampled_data = np.array(self.subsampled_data)
        print('load graph finish')
        fo = open('subsampled_data.txt', 'w')
        for data in self.subsampled_data:
            fo.write(str(data)+'\n')

        self.embeds, self.dims, self.oovs = self.loadembeds(embeds, lamtas)
        print('load embeds finish')

        self.sample_table = self.init_sample_table(triples_id)


    def loaddict(self, filename, test_model=False):
        idx = 0
        rel = open(filename, encoding='utf-8')
        reldict = dict()            
        reldict['UNK'] = 0
        num = 0
        for line in rel:
            idx = idx+1
            reldict[line.strip()]=idx  
            if test_model:
                num = num+1
                if num>2000:
                    break      
        return reldict

    def loaddict2(self, filenames, test_model=False):
        idx = 0
        reldict = dict()            
        reldict['UNK'] = 0
        for filename in filenames:
            rel = open(filename, encoding='utf-8')
            num = 0
            for line in rel:
                idx = idx+1
                reldict[line.strip()]=idx  
                if test_model:
                    num = num+1
                    if num>2000:
                        break        
        return reldict

    def loadgraph(self, data_path):
        ent_vocab, rel_vocab = self.wordindex, self.reldict
        triples_id = []
        count = dict()
        score_sum = 0
        num = 0
        with open(data_path) as f:
            for line in f:
                eles = line.strip().split('\t')
                trip = eval(eles[0])
                score = float(eles[1])
                #trip = eval(line)
                if (trip[0] not in ent_vocab) or (trip[2] not in ent_vocab):
                    continue
                score_sum +=score
                num +=1
                triples_id.append((ent_vocab[trip[0]], rel_vocab[trip[1]], ent_vocab[trip[2]], score))
                if ent_vocab[trip[0]] in count:
                    count[ent_vocab[trip[0]]] += 1
                else:
                    count[ent_vocab[trip[0]]] = 1
                if ent_vocab[trip[2]] in count:
                    count[ent_vocab[trip[2]]] += 1
                else:
                    count[ent_vocab[trip[2]]] = 1
        # triple_count = []
        # for eles in triples_id:
        #     triple_count.append(math.sqrt(count[eles[0]]*count[eles[2]]))
        sum_count = 0
        for ele in count:
            sum_count += count[ele] 


        frequency = dict()
        for ele in count:
            frequency[ele] = count[ele]/sum_count
        meanscore = score_sum/num

        return np.array(triples_id), count, frequency, meanscore

    def batch_iter(self,triples_id, batchsize, rand_flg=True):

        indices = np.random.permutation(len(triples_id)) if rand_flg else np.arange(len(triples_id))       
        for start in range(0, len(triples_id), batchsize):
            yield triples_id[indices[start: start+batchsize]]


    def generate_epoch(self):
        graphlist = self.padding(self.valuelist, self.count, padding_idx=0) 
        return graphlist           




    def build_freq(self, graphdict):
        count = []
        words = []
        for key in graphdict:
            count.append(len(graphdict[key]))
            words.append(key)

        pow_freq = np.array(count)**0.75
        power = sum(pow_freq)
        ratio = pow_freq/power
        table_size = 1e8
        freq = np.round(ratio*table_size)
        wordfreq = dict()
        for word, f in zip(words, freq):
            wordfreq[word] = int(f)
        return wordfreq
        
    def init_sample_table(self, data):
        count = []
        num = 1
        for i in range(len(self.wordindex)):
            if i not in self.count:
                count.append(1)
            else:
                count.append(self.count[i])
                num +=1
        pow_frequency = np.array(count)**0.75
        power = sum(pow_frequency)
        ratio = pow_frequency/ power
        table_size = 1e8
        count = np.round(ratio*table_size)
        sample_table = []
        for idx, x in enumerate(count):
            sample_table +=  [idx]*int(x)

        return np.array(sample_table)

    def subsampling(self,data):

        frequency = self.frequency
        subsampled_data = []
        fnum = 1e-5
        if len(data)> 2000000:
            fnum = 1e-7
        for word in data:
            #x = math.sqrt(frequency[word[0]]*frequency[word[2]])
            if self.sampletype == 'mean':
                x = (frequency[word[0]]+frequency[word[2]])/2
            elif self.sampletype == 'max':
                x = max(frequency[word[0]], frequency[word[2]])
            else:
                x = math.sqrt(frequency[word[0]]*frequency[word[2]])
            y = (math.sqrt(x/fnum)+1)*fnum/x
            #P[str(data[idx])] = y
            if random.random()<y:#P[str(word)]:
                subsampled_data.append(word)
        return subsampled_data


    def keytovalue(self,graphdict):
        keylist = [i for i in range(self.vocab_size)]
        maxlen = 0
        valuelist = []
        numNeigh = []
        for key in keylist:
            if key in graphdict:
                value = graphdict[key]
            else:
                value = []
            neigh = len(value)

            numNeigh.append(neigh)

            if maxlen < neigh:
                maxlen = len(value)
            valuelist.append(value)

        return valuelist, numNeigh, maxlen
    def padding(self, valuelist, maxlen, padding_idx):
        valuelist_pad = []
        for value in valuelist:
            valuelist_pad.append(value+(maxlen-len(value))*[padding_idx])
        return valuelist_pad

    def loadembeds(self, embedfiles, lamtas):
        f = dict()
        for i in range(len(embedfiles)):
            f[i] = open(embedfiles[i])

        embeds =[]
        dims = []
        OOVs = []

        for embedi in range(len(embedfiles)):
            embed = dict()
            line1 = f[embedi].readline()
            dim = len(line1.strip().split()[1:])
            dims.append(dim)
            lamta = lamtas[embedi]
   
            embed['UNK'] = [0]*dims[embedi]
            print('load embed {}\r'.format(embedi), end="")

            while line1:
                eles = line1.strip().split()
                if eles[0] not in self.wordindex:
                    line1 = f[embedi].readline()
                    continue
                x = [float(i) for i in eles[1:]]
                z = sum([abs(i) for i in x])
                if z==0:
                    embed[eles[0]] = x
                else:
                    embed[eles[0]] = [i/z for i in x]
                line1 = f[embedi].readline()
            emb = []
            OOV = []
            for i in range(len(self.id2word)):
                word = self.id2word[i]
                if word in embed:
                    emb.append(embed[word])
                    OOV.append(lamta)
                else:
                    emb.append([0]*dims[embedi])
                    OOV.append(0)
            OOVs.append(OOV)

            embeds.append(emb)

        return embeds, dims, np.array(OOVs)




