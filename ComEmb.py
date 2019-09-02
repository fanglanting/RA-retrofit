import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as Func
import math
from score import scorefunction
from torch.optim.lr_scheduler import MultiStepLR
import time
from inputdata import Options
from model.Ensmodel import ensemble
import torch.nn.functional as F
import numpy as np
#from anology import anologyscore

class UniformNegativeGenerator(object):
    def __init__(self, n_ent, sample_table, n_negative, train_graph=None):
    	self.n_ent = n_ent
    	self.n_negative = n_negative
    	self.sample_table = sample_table
    def generate(self, pos_triplets):
        _batchsize = len(pos_triplets)
        sample_size = _batchsize * self.n_negative
        #neg_ents = np.random.choice(self.sample_table, size=sample_size)
        neg_ents = np.random.randint(0, self.n_ent, size=sample_size)
        
        neg_triplets = np.tile(pos_triplets, (self.n_negative, 1))
        head_or_tail = 2 * np.random.randint(0, 2, sample_size)
        neg_triplets[np.arange(sample_size), head_or_tail] = neg_ents
        #neg_triplets[np.arange(sample_size), head_or_tail] = neg_ents
        return neg_triplets, neg_ents

class ComEmb(object):
	def __init__(self,embedsfile, netfile, netlist, rellist, n_negative, testmodel, lamtas=[1,1,8,8]):
		self.op = Options(embedsfile, netfile, netlist, rellist, lamtas, testmodel)
		self.neg_generator = UniformNegativeGenerator(self.op.vocab_size,self.op.sample_table,  n_negative=n_negative)
		self.n_negative = n_negative
		print('Inialize Finish')

	def train(self,gamma=1, l2 = 1e-3, epoch_num= 400, batch_size=32, embedding_dim=300, lr =0.01, emnames='w2v',sname='geo'):
		embeds = self.op.embeds
		dims = self.op.dims
		vocab_size = self.op.vocab_size
		rel_size = self.op.rel_size
		id2word = self.op.id2word
		id2rel = self.op.id2rel
		wordindex = self.op.wordindex
		triples_id = self.op.subsampled_data
		oovs = self.op.oovs
		mean_score = self.op.meanscore
		batch_num = math.ceil(len(triples_id)/batch_size)

		print('dims: '+str(dims))
		print('learning rate:' + str(lr))
		print('gamma: '+str(gamma))
		print('batch_num:'+str(batch_num))
		print('mean score:'+str(mean_score))

		
		model = ensemble(vocab_size, rel_size, embedding_dim, embeds, gamma,l2, dims)
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=lr)

		if torch.cuda.is_available():
			model = model.cuda()
		        
		# note = open('note.txt', 'w')
		# note.write('gamma= {}\n'.format(gamma))
		# scheduler = MultiStepLR(optimizer, milestones=[50,90,120], gamma=0.5)

		for t in range(epoch_num):
			#scheduler.step()
			batch_num = 0
			for pos_triplets in self.op.batch_iter(triples_id, batch_size):	
				neg_triplets, neg_ents = self.neg_generator.generate(pos_triplets)
				inp = np.concatenate((pos_triplets[:,0], pos_triplets[:,2], neg_ents))

				pos_triplets = np.tile(pos_triplets, (self.n_negative, 1))
				weight = pos_triplets[:,3]	

				weight = Variable(torch.FloatTensor(weight))
				inp = Variable(torch.LongTensor(inp))
				pos_triplets =  Variable(torch.LongTensor(pos_triplets))
				neg_triplets = Variable(torch.LongTensor(neg_triplets))

				if torch.cuda.is_available():
					weight = weight.cuda()
					inp = inp.cuda()
					pos_triplets = pos_triplets.cuda()
					neg_triplets = neg_triplets.cuda()

		
				optimizer.zero_grad()
				loss,sEmb, sGraph = model(inp, pos_triplets, neg_triplets, weight, mean_score)
						
				loss.backward()
				optimizer.step()
				# if batch_num==1000:
				# 	note.write('epoch %2d batch %2d sp=%1.3f %1.3f %1.3f %1.3f %1.3f %1.3f %1.3f loss=%2.5f sEmb=%2.5f sGraph=%2.5f \n'%(t,batch_num,sp1, sp2, sp3, sp4,sp5, sp6,sp7, loss.data[0], sEmb.data[0], sGraph.data[0]))
				
				if batch_num%100==0:
					word_embeddings = model.metaemb.weight.data.cpu().numpy()
					sp1, sp2, sp3, sp4,sp5, sp6,sp7 = scorefunction(wordindex,word_embeddings)     
					print('epoch %2d batch %2d sp=%1.3f %1.3f %1.3f %1.3f %1.3f %1.3f %1.3f loss=%2.5f sEmb=%2.5f sGraph=%2.5f \r'%(t,batch_num,sp1, sp2, sp3, sp4,sp5, sp6,sp7, loss.data[0], sEmb.data[0], sGraph.data[0]),end="")
				batch_num = batch_num+1
			
			word_embeddings = model.metaemb.weight.data.cpu().numpy()
			sp1, sp2, sp3, sp4,sp5, sp6,sp7 = scorefunction(wordindex,word_embeddings)     
			print('epoch=%2d sp=%1.3f %1.3f %1.3f %1.3f %1.3f %1.3f %1.3f loss=%2.5f sEmb=%2.5f sGraph=%2.5f \r'%(t,sp1, sp2, sp3, sp4,sp5, sp6,sp7, loss.data[0], sEmb.data[0], sGraph.data[0]),end="")

		print('t=%2d  sp=%1.3f %1.3f %1.3f %1.3f %1.3f %1.3f %1.3f loss=%7.2f'%(t,sp1, sp2, sp3, sp4,sp5, sp6,sp7, loss.data[0]))
				
		fo = open('Trans_multi_loss1_gamma{}'.format(gamma),'w')
		for k in range(len(word_embeddings[:-1])):
			emb = word_embeddings[k]
			emb = [str(i) for i in emb]
			fo.write(id2word[k]+' '+' '.join(emb)+'\n')


if __name__ == '__main__':	
	embedsfile = ['./data/wgc_w_zip.txt','./data/wgc_g_zip.txt']
	netfile = ['./data/cnet_graph_score.txt','./data/wordnet_graph_score.txt', './data/ppdb_graph_score.txt']	
	vocabfile = './data/vocab_Ins_wgc.txt'
	rellist = ['./data/cnetrellist.txt','./data/wordnetrellist.txt','./data/ppdbrellist.txt']
	testmodel = False	
	gamma = 0.1
	learning_rates = [0.05, 0.02, 0.002]
	learning_rate = 0.02
	batch_size = 256
	

	ce= ComEmb(embedsfile, netfile, vocabfile, rellist, n_negative=5, testmodel=testmodel)
	ce.train(gamma=gamma,epoch_num= 200, lr=learning_rate, batch_size=batch_size,emnames='Multi')





