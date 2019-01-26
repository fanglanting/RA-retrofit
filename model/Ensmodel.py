import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import math
from torch.optim.lr_scheduler import MultiStepLR
import time
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import margin_ranking_loss

class ensemble(nn.Module):
	def __init__(self, vocab_size, rel_size, embedding_dim, embeds, gamma,l2, dims):
		super(ensemble, self).__init__()		
		self.gamma = gamma
		self.l2 = l2
		self.metaemb = nn.Embedding(vocab_size+1, embedding_dim, sparse=True)
		self.relemb = nn.Embedding(rel_size+1, embedding_dim, sparse=True)
		
		self.metaemb.weight.data.uniform_(-0, 0)	
		self.relemb.weight.data.uniform_(-0, 0)	
		self.embs = []
		self.metaemb.weight.data[0]=0	
		self.relu = nn.ReLU()
		
		for dim, embed in zip(dims, embeds):

			emb = nn.Embedding(vocab_size+1, dim, sparse=True)
			embed = torch.FloatTensor(embed).cuda() if torch.cuda.is_available() else  torch.FloatTensor(embed)
			emb.weight = nn.Parameter(embed)
			emb.weight.requires_grad = False
			self.embs.append(emb)
		stdv = 1 / math.sqrt(embedding_dim)
		self.W1 = nn.Parameter(torch.Tensor(embedding_dim,dims[0]))
		self.W1.data.uniform_(-stdv, stdv)
		self.Ws = [self.W1]
		if len(dims)>1:
			self.W2 = nn.Parameter(torch.Tensor(embedding_dim,dims[1]))
			self.W2.data.uniform_(-stdv, stdv)
			self.Ws.append(self.W2)
		if len(dims)>2:
			self.W3 = nn.Parameter(torch.Tensor(embedding_dim,dims[2]))
			self.W3.data.uniform_(-stdv, stdv)	
			self.Ws.append(self.W3)
		if len(dims)>3:		
			self.W4 = nn.Parameter(torch.Tensor(embedding_dim,dims[3]))
			self.W4.data.uniform_(-stdv, stdv)
			self.Ws.append(self.W4)
	
	def forward(self,inp, pos_samples, neg_samples, weight, meanscore):
		p_scores = self.cal_triplet_scores(pos_samples)

		n_scores = self.cal_triplet_scores(neg_samples)
		sGraph = torch.sum(torch.mul((meanscore+weight),self.relu(1 - (p_scores - n_scores))))
		embs = [self.embs[i](inp) for i in range(len(self.embs))]
		inpemb = self.metaemb(inp)
		sEmb = 0
		wnorm = 0
		for emb, W in zip(embs, self.Ws):
			lossE =  torch.norm(torch.matmul(inpemb, W)-emb,2,1)
			sEmb = sEmb + lossE
			wnorm += torch.norm(W,1)
		sEmb = torch.sum(sEmb)+self.l2*wnorm
		loss =  sEmb+(len(embs)/2)*self.gamma*sGraph/(meanscore+meanscore)

		return loss ,sEmb, sGraph 

	def _composite(self, sub_emb, rel_emb):
		return sub_emb + rel_emb

	def _cal_similarity(self, query, obj_emb):
		return - torch.sum((query - obj_emb)**2, dim=1)

	def cal_triplet_scores(self, samples):
		subs, rels, objs = samples[:, 0], samples[:, 1], samples[:, 2]
		sub_emb = self.pick_ent(subs)
		rel_emb = self.pick_rel(rels)
		obj_emb = self.pick_ent(objs)
		qs = self._composite(sub_emb, rel_emb)
		return self._cal_similarity(qs, obj_emb)

	def pick_ent(self, ents):
		return self.metaemb(ents)

	def pick_rel(self, rels):
		return self.relemb(rels)

