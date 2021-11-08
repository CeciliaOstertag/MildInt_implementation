import os
import re
import time
import numpy as np
import random
import glob
import matplotlib
import matplotlib.pyplot as plt
import statistics
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import f1_score

import scipy.ndimage

from skimage import exposure

from sklearn.metrics import roc_curve

import csv
from itertools import zip_longest

import sklearn

import pandas as pd

from collections import OrderedDict


torch.backends.cudnn.enabled = False
torch.manual_seed(42)
print(torch.random.initial_seed())

class MultiDataset(Dataset):
	def __init__(self, root_dir, train, augment = False, dataset=None, list_ids=None, list_labels=None, list_infos=None, list_nbpersubj=None, list_uniqueids = None, val_size=None, fold=0, test = False, missing_data = False, inference=False):

		self.test = test
		valstart = fold * val_size
		valstop = fold * val_size + val_size
		if (train == True) and (dataset == None):
			print('Building new dataset\tTRAINING DATA')
			self.root_dir = root_dir
			
			self.dico_subjs = {}
			
			df = pd.read_csv(self.root_dir+"/data_rnn.csv")
			df = df.drop_duplicates()
			df = df[df['VISCODE'].isin(["bl","m06","m12","m18","m24"])]
			#(["bl","m03","m06","m12","m18","m24","m30","m36","m42","m48","m54","m60","m66","m72"])]
			df = df.sort_values(by=['ID'])
			df = df.sort_values(by=['VISCODE'])
			
			for subject, data in df.groupby(['ID']):
				df_labels = data[['ID','TRAJ']]
				df_labels = df_labels.drop_duplicates()
				df_dem = data[['ID','AGE','PTGENDER','APOE4']]
				df_cog = data[['ID','CDRSB','RAVLT_immediate','RAVLT_learning','RAVLT_forgetting','LDELTOTAL','DIGITSCOR','TRABSCOR',	'FAQ']]
				df_mri = data[['ID','Ventricles','Hippocampus','WholeBrain','Entorhinal','Fusiform','MidTemp','ICV']]

				self.dico_subjs[subject] = {"Id":subject,"Label":df_labels.to_numpy()[0,1],"Demographics":df_dem.to_numpy()[:,1:],"Cognitive":df_cog.to_numpy()[:,1:],"MRI":df_mri.to_numpy()[:,1:]}

			#seed = np.random.randint(0,1024)
			seed = 42
			#print("SEED ", seed)
			random.seed(seed)
			with open('subjects.npy', 'rb') as f:
				self.list_uniqueids = np.load(f).tolist()
			print("ID list loaded")
			self.list_files = []
			self.list_alllabels = []
			self.list_files = []
			self.list_allids = []
			for id_ in self.list_uniqueids :
				self.list_files.append([self.dico_subjs[id_]["Demographics"],self.dico_subjs[id_]["Cognitive"],self.dico_subjs[id_]["MRI"],self.dico_subjs[id_]["Label"]])
				self.list_alllabels.append(self.dico_subjs[id_]["Label"])
				self.list_allids.append(self.dico_subjs[id_]["Id"])

			
			teststop = round(0.2*len(self.list_uniqueids))
			maxidtest = teststop
			start = valstart
			stop = valstop
				
			self.dataset = self.list_files[maxidtest:]
			self.list_ids = self.list_allids[maxidtest:]
			self.list_uniqueids = self.list_uniqueids[teststop:]
			self.list_labels = self.list_alllabels[maxidtest:]
			self.test = self.list_files[:maxidtest]
			self.test_ids = self.list_allids[:maxidtest]
			self.test_uniqueids = self.list_uniqueids[:maxidtest]
			self.test_labels = self.list_alllabels[:maxidtest]
			#print(self.list_ids)
			if self.list_files[0:start] == None:
				self.data = self.dataset[stop:len(self.dataset)]
				self.labels = self.list_labels[stop:len(self.list_labels)]
				self.ids = self.list_ids[stop:len(self.list_ids)]
				self.uniqueids = self.list_uniqueids[valstop:len(self.list_uniqueids)]
			
			elif self.list_files[stop:len(self.list_files)] == None:
				self.data = self.dataset[0:start]
				self.labels = self.list_labels[0:start]
				self.ids = self.list_ids[0:start]
				self.uniqueids = self.list_uniqueids[0:valstart]
			
			else:
				self.data = self.dataset[0:start] + self.dataset[stop:len(self.dataset)]
				self.labels = self.list_labels[0:start] + self.list_labels[stop:len(self.list_labels)]
				self.ids = self.list_ids[0:start] + self.list_ids[stop:len(self.list_ids)]
				self.uniqueids = self.list_uniqueids[0:valstart] + self.list_uniqueids[valstop:len(self.list_uniqueids)]
			
				
		elif (train == True) and (dataset != None):
			print('Using pre-shuffled dataset\tTRAINING DATA')
			self.dataset = dataset
			self.list_ids = list_ids
			self.list_uniqueids = list_uniqueids
			self.list_labels = list_labels
			self.list_nbpersubj = list_nbpersubj
			start = valstart
			stop = valstop
			if self.dataset[0:start] == None:
				self.data = self.dataset[stop:len(self.dataset)]
				self.labels = self.list_labels[stop:len(self.list_labels)]
				self.ids = self.list_ids[stop:len(self.list_ids)]
				self.uniqueids = self.list_uniqueids[valstop:len(self.list_uniqueids)]
			
			elif self.dataset[stop:len(self.dataset)] == None:
				self.data = self.dataset[0:start]
				self.labels = self.list_labels[0:start]
				self.ids = self.list_ids[0:start]
				self.uniqueids = self.list_uniqueids[0:valstart]
			
			else:
				self.data = self.dataset[0:start] + self.dataset[stop:len(self.dataset)]
				self.labels = self.list_labels[0:start] + self.list_labels[stop:len(self.list_labels)]
				self.ids = self.list_ids[0:start] + self.list_ids[stop:len(self.list_ids)]
				self.uniqueids = self.list_uniqueids[0:valstart] + self.list_uniqueids[valstop:len(self.list_uniqueids)]

		elif (train == False) and (test == False):
			print('Using pre-shuffled dataset\tVALIDATION DATA')
			self.dataset = dataset
			self.list_ids = list_ids
			self.list_uniqueids = list_uniqueids
			self.list_labels = list_labels
			self.list_nbpersubj = list_nbpersubj
			start = valstart
			stop = valstop
			self.data = self.dataset[start:stop]
			self.labels = self.list_labels[start:stop]
			self.ids = self.list_ids[start:stop]
			self.uniqueids = self.list_uniqueids[valstart:valstop]
			
		elif (train == False) and (test == True):
			print('Using test dataset\tTEST DATA')
			self.data = dataset
			self.ids = list_ids
			self.labels = list_labels
			self.uniqueids = list_uniqueids

		print("TOTAL PAIRS ",len(self.ids))
		print("TOTAL UNIQUE SUBJS ",len(self.uniqueids))
		print("DECLINE ",self.labels.count(1))
		print("STABLE ",self.labels.count(0))
		
	def __len__(self):
		'Denotes the number of batches per epoch'
		return len(self.data)
		
	def __getitem__(self, idx):
		'Generate one batch of data'
		dem, cog, mri, labels = self.data[idx]
		dem = torch.Tensor(dem.astype(np.float32))
		cog = torch.Tensor(cog.astype(np.float32))
		mri = torch.Tensor(mri.astype(np.float32))
		labels = torch.Tensor([labels])
		labels = torch.squeeze(labels)
		return dem, cog, mri, labels
		
	def getShuffledDataset(self):
		return self.dataset, self.list_ids, self.list_labels, self.list_uniqueids
		
	def getTestDataset(self):
		return self.test, self.test_ids, self.test_labels, self.test_uniqueids		
		
class DemNet(nn.Module):
	def __init__(self):
		super(DemNet, self).__init__()		
		self.GRUDem = torch.nn.GRU(input_size=3, hidden_size=3, batch_first=True)
		self.OutDem = torch.nn.Linear(3,2)
		self.SoftDem = torch.nn.Softmax()
		
	def forward(self, dem):
		x = self.GRUDem(dem)
		x = x[0][:,-1,:]
		out = self.OutDem(x)
		outsoft = self.SoftDem(out)
		return x, out, outsoft

class CogNet(nn.Module):
	def __init__(self):
		super(CogNet, self).__init__()			
		self.GRUCog = torch.nn.GRU(input_size=8, hidden_size=8, batch_first=True)
		self.OutCog = torch.nn.Linear(8,2)
		self.SoftCog = torch.nn.Softmax()
		
	def forward(self, cog):
		x = self.GRUCog(cog)
		x = x[0][:,-1,:]
		out = self.OutCog(x)
		outsoft = self.SoftCog(out)
		return x, out, outsoft
		
class MRINet(nn.Module):
	def __init__(self):
		super(MRINet, self).__init__()			
		self.GRUMRI = torch.nn.GRU(input_size=7, hidden_size=7, batch_first=True)
		self.OutMRI = torch.nn.Linear(7,2)
		self.SoftMRI = torch.nn.Softmax()
		
	def forward(self, mri):
		x = self.GRUMRI(mri)
		x = x[0][:,-1,:]
		out = self.OutMRI(x)
		outsoft = self.SoftMRI(out)
		return x, out, outsoft
		
class MultiNet(nn.Module):
	def __init__(self):
		super(MultiNet, self).__init__()
		self.dem_module = DemNet()
		self.cog_module = CogNet()
		self.mri_module = MRINet()
		self.OutMulti = torch.nn.Linear(18,2)
		self.SoftMulti = torch.nn.Softmax()
		
	def forward(self, dem, cog, mri):
		dem, _, _ = self.dem_module(dem)
		cog, _, _ = self.cog_module(cog)
		mri, _, _ = self.mri_module(mri)
		x = torch.cat([dem, cog], dim=1)
		x = torch.cat([x, mri], dim=1)
		out = self.OutMulti(x)
		outsoft = self.SoftMulti(out)
		return x, out, outsoft


batch_size = 40
num_classes = 2
epochs = 50
val_size = round(0.2*381) 

path = "./ADNI_full"
datatrain = MultiDataset(path, train=True, augment=False, val_size=val_size)
shuffled_dataset, ids, labels, uniqueids = datatrain.getShuffledDataset()

test_dataset, test_ids, test_labels, test_uniqueids = datatrain.getTestDataset()
print("Example of test data: ")
print(test_ids[:10])

dataval = MultiDataset(path, train = False, augment=False, dataset = shuffled_dataset, list_ids = ids, list_labels = labels, list_uniqueids = uniqueids, val_size=val_size, missing_data=False)

nb_folds = (len(datatrain) + len(dataval)) // len(dataval)
print((len(datatrain) + len(dataval)) % val_size)
print("\nRunning training with "+str(nb_folds)+"-fold validation")


#p = input("\nPress Enter to continue\n")

for fold in range(nb_folds):

	datatrain = MultiDataset(path, train=True, augment=True, dataset = shuffled_dataset, list_ids = ids, list_labels = labels, list_uniqueids = uniqueids, val_size=val_size, missing_data=False, fold=fold)
	train_dataloader = DataLoader(datatrain, shuffle=True, num_workers=1,batch_size=batch_size, drop_last=True) #sampler mutually exclusive with shuffle

	dataval = MultiDataset(path, train = False, augment=False, dataset = shuffled_dataset, list_ids = ids, list_labels = labels, list_uniqueids = uniqueids, val_size=val_size, missing_data=False,fold=fold)
	val_dataloader = DataLoader(dataval, shuffle=True, num_workers=1,batch_size=batch_size, drop_last=True)
	
	datatest = MultiDataset(path, train = False, augment=False, dataset = test_dataset, list_ids = test_ids, list_labels = test_labels, list_uniqueids = test_uniqueids, val_size=val_size, test= True, missing_data=False,fold=fold)
	test_dataloader = DataLoader(datatest, shuffle=True, num_workers=0,batch_size=1, drop_last=True)

	# OPTIM SEPARATELY THE 3 MODELS (TRAIN AND VAL) THEN SAVE WEIGHTS
	# CREATE MULTI MODEL, LOAD AND FIX WEIGHTS
	# OPTIM MULTI MODEL (TRAIN, VAL, TEST)
	
	"""
	print("\n\t\tDEMOGRAPHICS MODULE\n")
	start_time = time.time() 
	demnet = DemNet()
	#tdsnet = torch.nn.DataParallel(tdsnet,device_ids=[0,1])
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	demnet = demnet.to(device)

	ce = torch.nn.CrossEntropyLoss()
	optimizer = optim.AdamW(demnet.parameters(), lr=0.005, weight_decay=0.2)
	
	valloss_history = []
 
	for epoch in range(epochs): 
		print ("Fold "+str(fold)+" Epoch "+str(epoch+1)+"/"+str(epochs))		
		 
		total=0
		correct=0
		tot_train_f1 = 0
		tot_train_loss=0

		demnet.train()
		for i, data in enumerate(train_dataloader):
			dem, cog, mri, label = data
			optimizer.zero_grad()
			label = label.to(device)
			dem = dem.to(device)

			x, out, outsoft = demnet(dem)

			ce_loss = ce(out, label.to(torch.long))
			train_loss =  ce_loss #+ mse_loss
			train_loss.backward()
			optimizer.step()
		
			tot_train_loss += train_loss.item()
			total += label.size(0)
			predicted = np.argmax(outsoft.cpu().detach().numpy(),axis=1)
			label = label.data.cpu().detach().numpy()
			correct += (predicted == label).sum().item()
			tot_train_f1 += f1_score(label, predicted)

		print("Training loss ",tot_train_loss/float(i+1))
		print("Training acc ",float(correct)/float(total))
		print("Training f1 ",float(tot_train_f1)/float(i+1))
		
		total=0
		correct=0
		tot_val_loss=0
		tot_val_f1 = 0
		demnet.train(False)
		with torch.no_grad():
			for j, data in enumerate(val_dataloader):
				dem, cog, mri, label = data
				optimizer.zero_grad()
				label = label.to(device)
				dem = dem.to(device)

				x, out, outsoft = demnet(dem)

				ce_loss = ce(out, label.to(torch.long))
				val_loss =  ce_loss #+ mse_loss
		
				tot_val_loss += val_loss.item()
				total += label.size(0)
				predicted = np.argmax(outsoft.cpu().detach().numpy(),axis=1)
				label = label.data.cpu().detach().numpy()
				correct += (predicted == label).sum().item()
				tot_val_f1 += f1_score(label, predicted)
		
		print("Val loss ",tot_val_loss/float(j+1))
		valloss_history.append(tot_val_loss/float(j+1))
		print("Val acc ",float(correct)/float(total))
		print("Val f1 ",float(tot_val_f1)/float(j+1))
				
		
		if (epoch > 0) and (valloss_history[-1] < min(valloss_history[:-1])):
			torch.save(demnet.state_dict(), "DemNet_"+str(fold)+".pt")
			print("Lowest loss so far")
			print("Model saved")
		print("\n")
		demnet.train()
	print("Time (s): ",(time.time() - start_time))
	
	del demnet
	
	demnet = DemNet()
	trained_model = "DemNet_"+str(fold)+".pt"
	state_dict = torch.load(trained_model)
	from collections import OrderedDict
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		#name = k[7:] # remove `module.`
		name = k
		new_state_dict[name] = v
	# load params
	demnet.load_state_dict(new_state_dict)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	demnet = demnet.to(device)
	
	total=0
	correct=0
	tot_val_loss=0
	tot_val_f1 = 0
	label_ = np.asarray([])
	predicted_ = np.asarray([])
	out_ = np.asarray([])
	demnet.train(False)

	with torch.no_grad():

		for j, data in enumerate(test_dataloader):
			dem, cog, mri, label = data
			optimizer.zero_grad()
			label = label.to(device)
			dem =dem.to(device)

			x, out, outsoft = demnet(dem)
		
			label = label.cpu().detach().numpy()
			label_ = np.concatenate((label_,label))
			predicted = np.argmax(outsoft.cpu().detach().numpy(),axis=1)
			predicted_ = np.concatenate((predicted_, predicted))
			

	total = len(label_)
	correct = np.sum(np.equal(predicted_, label_))
	tot_val_f1 = f1_score(label_, predicted_)
	confmat = sklearn.metrics.confusion_matrix(label_, predicted_)

	print("Test acc ",float(correct)/float(total))
	print("Test f1 ",float(tot_val_f1))
	print(confmat)
	tn, fp, fn, tp = confmat.ravel()
	
	
	print("\n\t\tCOGNITIVE MODULE\n")
	start_time = time.time() 
	cognet = CogNet()
	#tdsnet = torch.nn.DataParallel(tdsnet,device_ids=[0,1])
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	cognet = cognet.to(device)

	ce = torch.nn.CrossEntropyLoss()
	optimizer = optim.AdamW(cognet.parameters(), lr=0.005, weight_decay=0.2)
	
	valloss_history = []
 
	for epoch in range(epochs): 
		print ("Fold "+str(fold)+" Epoch "+str(epoch+1)+"/"+str(epochs))		
		 
		total=0
		correct=0
		tot_train_f1 = 0
		tot_train_loss=0

		cognet.train()
		for i, data in enumerate(train_dataloader):
			dem, cog, mri, label = data
			optimizer.zero_grad()
			label = label.to(device)
			cog = cog.to(device)

			x, out, outsoft = cognet(cog)

			ce_loss = ce(out, label.to(torch.long))
			train_loss =  ce_loss #+ mse_loss
			train_loss.backward()
			optimizer.step()
		
			tot_train_loss += train_loss.item()
			total += label.size(0)
			predicted = np.argmax(outsoft.cpu().detach().numpy(),axis=1)
			label = label.data.cpu().detach().numpy()
			correct += (predicted == label).sum().item()
			tot_train_f1 += f1_score(label, predicted)

		print("Training loss ",tot_train_loss/float(i+1))
		print("Training acc ",float(correct)/float(total))
		print("Training f1 ",float(tot_train_f1)/float(i+1))
		
		total=0
		correct=0
		tot_val_loss=0
		tot_val_f1 = 0
		cognet.train(False)
		with torch.no_grad():
			for j, data in enumerate(val_dataloader):
				dem, cog, mri, label = data
				optimizer.zero_grad()
				label = label.to(device)
				cog = cog.to(device)

				x, out, outsoft = cognet(cog)

				ce_loss = ce(out, label.to(torch.long))
				val_loss =  ce_loss #+ mse_loss
		
				tot_val_loss += val_loss.item()
				total += label.size(0)
				predicted = np.argmax(outsoft.cpu().detach().numpy(),axis=1)
				label = label.data.cpu().detach().numpy()
				correct += (predicted == label).sum().item()
				tot_val_f1 += f1_score(label, predicted)
		
		print("Val loss ",tot_val_loss/float(j+1))
		valloss_history.append(tot_val_loss/float(j+1))
		print("Val acc ",float(correct)/float(total))
		print("Val f1 ",float(tot_val_f1)/float(j+1))
				
		
		if (epoch > 0) and (valloss_history[-1] < min(valloss_history[:-1])):
			torch.save(cognet.state_dict(), "CogNet_"+str(fold)+".pt")
			print("Lowest loss so far")
			print("Model saved")
		print("\n")
		cognet.train()
	print("Time (s): ",(time.time() - start_time))
	
	del cognet
	
	cognet = CogNet()
	trained_model = "CogNet_"+str(fold)+".pt"
	state_dict = torch.load(trained_model)
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		#name = k[7:] # remove `module.`
		name = k
		new_state_dict[name] = v
	# load params
	cognet.load_state_dict(new_state_dict)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	cognet = cognet.to(device)
	
	total=0
	correct=0
	tot_val_loss=0
	tot_val_f1 = 0
	label_ = np.asarray([])
	predicted_ = np.asarray([])
	out_ = np.asarray([])
	demnet.train(False)

	with torch.no_grad():

		for j, data in enumerate(test_dataloader):
			dem, cog, mri, label = data
			optimizer.zero_grad()
			label = label.to(device)
			cog = cog.to(device)

			x, out, outsoft = cognet(cog)
		
			label = label.cpu().detach().numpy()
			label_ = np.concatenate((label_,label))
			predicted = np.argmax(outsoft.cpu().detach().numpy(),axis=1)
			predicted_ = np.concatenate((predicted_, predicted))
			

	total = len(label_)
	correct = np.sum(np.equal(predicted_, label_))
	tot_val_f1 = f1_score(label_, predicted_)
	confmat = sklearn.metrics.confusion_matrix(label_, predicted_)

	print("Test acc ",float(correct)/float(total))
	print("Test f1 ",float(tot_val_f1))
	print(confmat)
	tn, fp, fn, tp = confmat.ravel()
	
	
	print("\n\t\tMRI MODULE\n")
	start_time = time.time() 
	mrinet = MRINet()
	#tdsnet = torch.nn.DataParallel(tdsnet,device_ids=[0,1])
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	mrinet = mrinet.to(device)

	ce = torch.nn.CrossEntropyLoss()
	optimizer = optim.AdamW(mrinet.parameters(), lr=0.005, weight_decay=0.2)
	
	valloss_history = []
 
	for epoch in range(epochs): 
		print ("Fold "+str(fold)+" Epoch "+str(epoch+1)+"/"+str(epochs))		
		 
		total=0
		correct=0
		tot_train_f1 = 0
		tot_train_loss=0

		mrinet.train()
		for i, data in enumerate(train_dataloader):
			dem, cog, mri, label = data
			optimizer.zero_grad()
			label = label.to(device)
			mri = mri.to(device)

			x, out, outsoft = mrinet(mri)

			ce_loss = ce(out, label.to(torch.long))
			train_loss =  ce_loss #+ mse_loss
			train_loss.backward()
			optimizer.step()
		
			tot_train_loss += train_loss.item()
			total += label.size(0)
			predicted = np.argmax(outsoft.cpu().detach().numpy(),axis=1)
			label = label.data.cpu().detach().numpy()
			correct += (predicted == label).sum().item()
			tot_train_f1 += f1_score(label, predicted)

		print("Training loss ",tot_train_loss/float(i+1))
		print("Training acc ",float(correct)/float(total))
		print("Training f1 ",float(tot_train_f1)/float(i+1))
		
		total=0
		correct=0
		tot_val_loss=0
		tot_val_f1 = 0
		mrinet.train(False)
		with torch.no_grad():
			for j, data in enumerate(val_dataloader):
				dem, cog, mri, label = data
				optimizer.zero_grad()
				label = label.to(device)
				mri = mri.to(device)

				x, out, outsoft = mrinet(mri)

				ce_loss = ce(out, label.to(torch.long))
				val_loss =  ce_loss #+ mse_loss
		
				tot_val_loss += val_loss.item()
				total += label.size(0)
				predicted = np.argmax(outsoft.cpu().detach().numpy(),axis=1)
				label = label.data.cpu().detach().numpy()
				correct += (predicted == label).sum().item()
				tot_val_f1 += f1_score(label, predicted)
		
		print("Val loss ",tot_val_loss/float(j+1))
		valloss_history.append(tot_val_loss/float(j+1))
		print("Val acc ",float(correct)/float(total))
		print("Val f1 ",float(tot_val_f1)/float(j+1))
				
		
		if (epoch > 0) and (valloss_history[-1] < min(valloss_history[:-1])):
			torch.save(mrinet.state_dict(), "MRINet_"+str(fold)+".pt")
			print("Lowest loss so far")
			print("Model saved")
		print("\n")
		mrinet.train()
	print("Time (s): ",(time.time() - start_time))
	
	
	del mrinet
	
	mrinet = MRINet()
	trained_model = "MRINet_"+str(fold)+".pt"
	state_dict = torch.load(trained_model)
	from collections import OrderedDict
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		#name = k[7:] # remove `module.`
		name = k
		new_state_dict[name] = v
	# load params
	mrinet.load_state_dict(new_state_dict)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	mrinet = mrinet.to(device)

	total=0
	correct=0
	tot_val_loss=0
	tot_val_f1 = 0
	label_ = np.asarray([])
	predicted_ = np.asarray([])
	out_ = np.asarray([])
	mrinet.train(False)

	with torch.no_grad():

		for j, data in enumerate(test_dataloader):
			dem, cog, mri, label = data
			optimizer.zero_grad()
			label = label.to(device)
			mri = mri.to(device)

			x, out, outsoft = mrinet(mri)
		
			label = label.cpu().detach().numpy()
			label_ = np.concatenate((label_,label))
			predicted = np.argmax(outsoft.cpu().detach().numpy(),axis=1)
			predicted_ = np.concatenate((predicted_, predicted))
			

	total = len(label_)
	correct = np.sum(np.equal(predicted_, label_))
	tot_val_f1 = f1_score(label_, predicted_)
	confmat = sklearn.metrics.confusion_matrix(label_, predicted_)

	print("Test acc ",float(correct)/float(total))
	print("Test f1 ",float(tot_val_f1))
	print(confmat)
	tn, fp, fn, tp = confmat.ravel()
	"""
	
	print("\n\t\tMULTIMODAL MODULE\n")
	start_time = time.time() 
	multinet = MultiNet()
	model_dict = multinet.state_dict()
	new_state_dict = OrderedDict()
	trained_model = "DemNet_"+str(fold)+".pt"
	state_dict = torch.load(trained_model)
	for k, v in state_dict.items():
		name = "dem_module."+k
		new_state_dict[name] = v
		#v.require_grad = False
		trained_model = "CogNet_"+str(fold)+".pt"
	state_dict = torch.load(trained_model)
	for k, v in state_dict.items():
		name = "cog_module."+k
		new_state_dict[name] = v
		#v.require_grad = False
	trained_model = "MRINet_"+str(fold)+".pt"
	state_dict = torch.load(trained_model)
	for k, v in state_dict.items():
		name = "mri_module."+k
		new_state_dict[name] = v
		#v.require_grad = False
	# load params
	model_dict.update(new_state_dict)
	multinet.load_state_dict(model_dict)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	multinet = multinet.to(device)

	ce = torch.nn.CrossEntropyLoss()
	optimizer = optim.AdamW(multinet.parameters(), lr=0.005, weight_decay=0.2)
	
	valloss_history = []
 
	for epoch in range(epochs): 
		print ("Fold "+str(fold)+" Epoch "+str(epoch+1)+"/"+str(epochs))		
		 
		total=0
		correct=0
		tot_train_f1 = 0
		tot_train_loss=0

		multinet.train()
		for i, data in enumerate(train_dataloader):
			dem, cog, mri, label = data
			optimizer.zero_grad()
			label = label.to(device)
			dem = dem.to(device)
			cog = cog.to(device)
			mri = mri.to(device)

			x, out, outsoft = multinet(dem, cog, mri)

			ce_loss = ce(out, label.to(torch.long))
			train_loss =  ce_loss #+ mse_loss
			train_loss.backward()
			optimizer.step()
		
			tot_train_loss += train_loss.item()
			total += label.size(0)
			predicted = np.argmax(outsoft.cpu().detach().numpy(),axis=1)
			label = label.data.cpu().detach().numpy()
			correct += (predicted == label).sum().item()
			tot_train_f1 += f1_score(label, predicted)

		print("Training loss ",tot_train_loss/float(i+1))
		print("Training acc ",float(correct)/float(total))
		print("Training f1 ",float(tot_train_f1)/float(i+1))
		
		total=0
		correct=0
		tot_val_loss=0
		tot_val_f1 = 0
		multinet.train(False)
		with torch.no_grad():
			for j, data in enumerate(val_dataloader):
				dem, cog, mri, label = data
				optimizer.zero_grad()
				label = label.to(device)
				dem = dem.to(device)
				cog = cog.to(device)
				mri = mri.to(device)

				x, out, outsoft = multinet(dem, cog,mri)

				ce_loss = ce(out, label.to(torch.long))
				val_loss =  ce_loss #+ mse_loss
		
				tot_val_loss += val_loss.item()
				total += label.size(0)
				predicted = np.argmax(outsoft.cpu().detach().numpy(),axis=1)
				label = label.data.cpu().detach().numpy()
				correct += (predicted == label).sum().item()
				tot_val_f1 += f1_score(label, predicted)
		
		print("Val loss ",tot_val_loss/float(j+1))
		valloss_history.append(tot_val_loss/float(j+1))
		print("Val acc ",float(correct)/float(total))
		print("Val f1 ",float(tot_val_f1)/float(j+1))
				
		
		if (epoch > 0) and (valloss_history[-1] < min(valloss_history[:-1])):
			torch.save(multinet.state_dict(), "MultiNet_"+str(fold)+".pt")
			print("Lowest loss so far")
			print("Model saved")
		print("\n")
		multinet.train()
	print("Time (s): ",(time.time() - start_time))
	
	
	del multinet
	
	multinet = MultiNet()
	trained_model = "MultiNet_"+str(fold)+".pt"
	state_dict = torch.load(trained_model)
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k
		new_state_dict[name] = v
	# load params
	multinet.load_state_dict(new_state_dict)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	multinet = multinet.to(device)

	total=0
	correct=0
	tot_val_loss=0
	tot_val_f1 = 0
	label_ = np.asarray([])
	predicted_ = np.asarray([])
	out_ = np.asarray([])
	multinet.train(False)

	with torch.no_grad():

		for j, data in enumerate(test_dataloader):
			dem, cog, mri, label = data
			optimizer.zero_grad()
			label = label.to(device)
			label = label.to(device)
			dem = dem.to(device)
			cog = cog.to(device)
			mri = mri.to(device)

			x, out, outsoft = multinet(dem, cog,mri)
		
			label = label.cpu().detach().numpy()
			label_ = np.concatenate((label_,label))
			predicted = np.argmax(outsoft.cpu().detach().numpy(),axis=1)
			predicted_ = np.concatenate((predicted_, predicted))
			out = outsoft[:,-1].cpu().detach().numpy()
			out_ = np.concatenate((out_, out))
			

	total = len(label_)
	correct = np.sum(np.equal(predicted_, label_))
	tot_val_f1 = f1_score(label_, predicted_)
	confmat = sklearn.metrics.confusion_matrix(label_, predicted_)
	fpr, tpr, thresholds = sklearn.metrics.roc_curve(label_, out_, drop_intermediate=False)
	auc = sklearn.metrics.auc(fpr, tpr)
	print("AUC ",auc)

	print("Test acc ",float(correct)/float(total))
	print("Test f1 ",float(tot_val_f1))
	print(confmat)
	tn, fp, fn, tp = confmat.ravel()
		
