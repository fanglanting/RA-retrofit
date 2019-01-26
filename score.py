import collections
import numpy as np

import math
import os
import random
import nltk
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from pyparsing import StringEnd, oneOf, FollowedBy, Optional, ZeroOrMore, SkipTo

import json, csv
from scipy.stats import spearmanr
import math
def cosine_similarity(v1,v2):
  "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
  sumxx, sumxy, sumyy = 0, 0, 0
  for i in range(len(v1)):
    x = v1[i]; y = v2[i]
    sumxx += x*x
    sumyy += y*y
    sumxy += x*y
  return sumxy/max(1e-8,math.sqrt(sumxx*sumyy))

def generatedict():
  f = open('./tmp/vocab.txt')
  line = f.readline()
  vocab = []
  wordindex = dict()
  index = 0
  while line:
    word = line.strip().split()[0]
    wordindex[word] = index
    index = index +1
    line = f.readline()
  f.close()
  wordslist = []
  with open('./wordsim353/combined.csv') as csvfile:
    filein = csv.reader(csvfile)
    index = 0
    consim = []
    humansim = []
    for eles in filein:
      if index==0:
        index = 1
        continue
      if (eles[0] not in wordindex) or (eles[1] not in wordindex):
        continue
      word1 = int(wordindex[eles[0]])
      word2 = int(wordindex[eles[1]])
      wordslist.append(word1)
      wordslist.append(word2) 
  lines = open('./rw/rw.txt','r').readlines()
  index = 0
  consim = []
  humansim = []
  for line in lines:
    eles = line.strip().split()
    if (eles[0] not in wordindex) or (eles[1] not in wordindex):
      continue
    word1 = int(wordindex[eles[0]])
    word2 = int(wordindex[eles[1]])
    wordslist.append(word1)
    wordslist.append(word2) 
  return wordindex, wordslist  
def scorefunction(wordindex,embed):
  ze = []
  with open('./testdata/wordsim353/combined.csv') as csvfile:
    filein = csv.reader(csvfile)
    index = 0
    consim = []
    humansim = []
    for eles in filein:
      if index==0:
        index = 1
        continue
      if (eles[0] not in wordindex) or (eles[1] not in wordindex):
        continue

      word1 = wordindex[eles[0]]
      word2 = wordindex[eles[1]]
      humansim.append(float(eles[2]))


      value1 =  embed[word1]
      value2 =  embed[word2]
      index =index + 1


      score = cosine_similarity(value1, value2)
      consim.append(score)

  cor1, pvalue1 = spearmanr(humansim, consim)

  if 1==1:
    lines = open('./testdata/rw/rw.txt','r').readlines()
    index = 0
    consim = []
    humansim = []
    for line in lines:
      eles = line.strip().split()
      if (eles[0] not in wordindex) or (eles[1] not in wordindex):
        continue
      word1 = wordindex[eles[0]]
      word2 = wordindex[eles[1]]
      humansim.append(float(eles[2]))
      
      value1 =  embed[word1]
      value2 =  embed[word2]
      index =index + 1
      score = cosine_similarity(value1, value2)
      consim.append(score)


  cor2, pvalue2 = spearmanr(humansim, consim)
  if 1==1:
    lines = open('./testdata/rg.csv','r').readlines()
    index = 0
    consim = []
    humansim = []
    for line in lines:
      eles = line.strip().split(';')
      if (eles[0] not in wordindex) or (eles[1] not in wordindex):
        continue
      word1 = wordindex[eles[0]]
      word2 = wordindex[eles[1]]
      humansim.append(float(eles[2]))
      
      value1 =  embed[word1]
      value2 =  embed[word2]
      index =index + 1
      score = cosine_similarity(value1, value2)
      consim.append(score)
  cor3, pvalue3 = spearmanr(humansim, consim)

  if 1==1:
    lines = open('./testdata/mc.csv','r').readlines()
    index = 0
    consim = []
    humansim = []
    for line in lines:
      eles = line.strip().split(';')
      if (eles[0] not in wordindex) or (eles[1] not in wordindex):
        continue
      word1 = wordindex[eles[0]]
      word2 = wordindex[eles[1]]
      humansim.append(float(eles[2]))
      
      value1 =  embed[word1]
      value2 =  embed[word2]
      index =index + 1
      score = cosine_similarity(value1, value2)
      consim.append(score)
  cor4, pvalue3 = spearmanr(humansim, consim)

  if 1==1:
    lines = open('./testdata/SimLex-999/SimLex-999.txt','r').readlines()
    index = 0
    consim = []
    humansim = []
    for line in lines:
      eles = line.strip().split('\t')
      if (eles[0] not in wordindex) or (eles[1] not in wordindex):
        continue
      word1 = wordindex[eles[0]]
      word2 = wordindex[eles[1]]
      humansim.append(float(eles[3]))
      
      value1 =  embed[word1]
      value2 =  embed[word2]
      index =index + 1
      score = cosine_similarity(value1, value2)
      consim.append(score)

  cor5, pvalue3 = spearmanr(humansim, consim)

  if 1==1:
    lines = open('./testdata/MEN/MEN_dataset_natural_form_full','r').readlines()
    index = 0
    consim = []
    humansim = []
    for line in lines:
      eles = line.strip().split(' ')
      if (eles[0] not in wordindex) or (eles[1] not in wordindex):
        continue
      word1 = wordindex[eles[0]]
      word2 = wordindex[eles[1]]
      humansim.append(float(eles[2]))
      
      value1 =  embed[word1]
      value2 =  embed[word2]
      index =index + 1
      score = cosine_similarity(value1, value2)
      consim.append(score)
  cor6, pvalue3 = spearmanr(humansim, consim)
  if 1==1:
    lines = open('./testdata/SCWS/ratings.txt','r').readlines()
    index = 0
    consim = []
    humansim = []
    for line in lines:
      eles = line.strip().split(' ')
      ele1 = eles[0].split('\t')[1]
      ele2 = eles[0].split('\t')[3]
      hscore = eles[-1].split('\t')[1]
      if index==0:
        index = 1
        continue
      if (ele1 not in wordindex) or (ele2 not in wordindex):
        continue

      word1 = wordindex[ele1]
      word2 = wordindex[ele2]
      humansim.append(float(hscore))
      
      value1 =  embed[word1]
      value2 =  embed[word2]
      index =index + 1
      score = cosine_similarity(value1, value2)
      consim.append(score)

    cor7, pvalue1 = spearmanr(humansim, consim)
  return cor1,cor2, cor3, cor4,cor5,cor6, cor7
