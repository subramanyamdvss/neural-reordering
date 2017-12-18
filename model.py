import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import gc
import math
import re
hyper = {'temp':0.01,'lr':0.035,'numepochs':1000,'optim':'momentum','batchsize':32,'lrdecaystepsize':30,'lrdecay':0.5,'weightnorm':3,'momentum':0.9,'weightdecay':0.0}
lcontrols = {'valstepsize':1,'savedir':'models2/','epoch':1,'PATH':'models2/46-train-0.128189-0.001782-val-0.127008-0.821692.model5'}
torch.backends.cudnn.benchmark= True

def weightini(modules,nl):
    for i,mod in enumerate(modules):
        mod.weight.data.normal_(0.0,math.sqrt(2.0/nl[i]))
        mod.bias.data.fill_(0.0)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        #self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))--this is how embeddings are defined.
        self.embedding = nn.Embedding(8000,100)
        self.embedding.weight.requires_grad = True
        #load the weights from finwordvecs.npy
        iniarr = np.load(open("finwordvecs.npy",'rb'))
        iniarr = torch.from_numpy(iniarr)
        self.iniarr = iniarr
        self.convst = nn.Conv2d(1,100,(5,100),1,(2,0))
        # 2 3x3 blocks
        self.BN1B1 = nn.BatchNorm2d(100)
        self.conv1B1 = nn.Conv2d(100,100,(3,1),1,(1,0))
        self.BN2B1 = nn.BatchNorm2d(100)
        self.conv2B1 = nn.Conv2d(100,100,(3,1),1,(1,0))

        self.BN1B2 = nn.BatchNorm2d(100)
        self.conv1B2 = nn.Conv2d(100,100,(3,1),1,(1,0))
        self.BN2B2 = nn.BatchNorm2d(100)
        self.conv2B2 = nn.Conv2d(100,100,(3,1),1,(1,0))
        #2 5x5 blocks
        self.BN1B3 = nn.BatchNorm2d(100)
        self.conv1B3 = nn.Conv2d(100,100,(5,1),1,(2,0))
        self.BN2B3 = nn.BatchNorm2d(100)
        self.conv2B3 = nn.Conv2d(100,100,(5,1),1,(2,0))

        self.BN1B4 = nn.BatchNorm2d(100)
        self.conv1B4 = nn.Conv2d(100,100,(5,1),1,(2,0))
        self.BN2B4 = nn.BatchNorm2d(100)
        self.conv2B4 = nn.Conv2d(100,100,(5,1),1,(2,0))
        #2 7x7 blocks
        self.BN1B5 = nn.BatchNorm2d(100)
        self.conv1B5 = nn.Conv2d(100,100,(7,1),1,(3,0))
        self.BN2B5 = nn.BatchNorm2d(100)
        self.conv2B5 = nn.Conv2d(100,100,(7,1),1,(3,0))

        self.BN1B6 = nn.BatchNorm2d(100)
        self.conv1B6 = nn.Conv2d(100,100,(7,1),1,(3,0))
        self.BN2B6 = nn.BatchNorm2d(100)
        self.conv2B6 = nn.Conv2d(100,100,(7,1),1,(3,0))

        self.convfin = nn.Conv2d(100,80,(1,1),1,(0,0))
        self.BNfin = nn.BatchNorm2d(80)
        # fully connected layer
        self.fc = nn.Linear(80,81)
        weightini([self.convst,self.conv1B1,self.conv2B1,self.conv1B2,self.conv2B2,self.conv1B3,self.conv2B3\
            ,self.conv1B4,self.conv2B4,self.conv1B5,self.conv2B5,self.conv1B6,self.conv2B6,self.convfin,self.fc]\
        ,[50000,300,300,300,300,500,500,500,500,700,700,700,700,80,6480])
        
    def forward(self, x):
        #32x80
        x = self.embedding(x)
        x = torch.unsqueeze(x,1)
        x = F.relu(x)
        x = self.convst(x)
        #2 3x3 blocks 32x100x80x1
        x = x + self.conv2B1(F.relu(self.BN2B1(self.conv1B1(F.relu(self.BN1B1(x))))))
        x = x + self.conv2B2(F.relu(self.BN2B2(self.conv1B2(F.relu(self.BN1B2(x))))))
        #2 5x5 blocks 32x100x80x1 
        x = x + self.conv2B3(F.relu(self.BN2B3(self.conv1B3(F.relu(self.BN1B3(x))))))
        x = x + self.conv2B4(F.relu(self.BN2B4(self.conv1B4(F.relu(self.BN1B4(x))))))
        #2 7x7 blocks 32x100x80x1
        x = x + self.conv2B5(F.relu(self.BN2B5(self.conv1B5(F.relu(self.BN1B5(x))))))
        x = x + self.conv2B6(F.relu(self.BN2B6(self.conv1B6(F.relu(self.BN1B6(x))))))
        #32x100x80x1
        x = F.relu(self.BNfin(self.convfin(x)))
        #32x80x80x1
        x = torch.squeeze(x,3)
        #32x80x80
        x = self.fc(x)
        #32x80x81
        x = x.view(-1,81)
        _,y = torch.max(x,-1)#size of y=[32*80]
        return x,y

    

net = Net()
net.load_state_dict(torch.load(lcontrols['PATH']))
# vali = int(lcontrols['PATH'].split('/')[1].split('-'))+1
vali = 0
net.zero_grad()

#defining the loss function
# lossfn = torch.nn.CrossEntropyLoss(weight=None, size_average=True, ignore_index=-100, reduce=True)
criterion = nn.CrossEntropyLoss().cuda()
#regularizer - batchnorm and
#l2 decay on all parameters

#Dataset Class
class WordReorderData(Dataset):

    def __init__(self, root_dir):
        self.lendict = []
        self.dataset = []
        if not (os.path.isfile("fold2/"+root_dir+"dict.p") and os.path.isfile("fold2/"+root_dir+"data.p")):
            self.root_dir = "fold1"+'/'+root_dir
            lst = os.listdir(self.root_dir)
            splf = lst[0].split('.')
            self.dataset = np.load(open(self.root_dir+"/"+lst[0],'rb'))
            self.dataset = np.array(self.dataset)
            leng = len(self.dataset)
            tmp = np.arange(80-leng)
            tmp.fill(0)
            tmp[0] = 2
            self.dataset = np.concatenate((self.dataset,tmp),0)
            self.dataset = [self.dataset]
            self.lendict.append(leng+1)
            for f in lst[1:]:
                splf = f.split('.')
                leng = int(splf[1])
                if  leng!= 0 :
                    fl = np.load(open(self.root_dir+"/"+f,'rb'))
                    fl = np.array(fl)
                    leng = len(fl)
                    tmp = np.arange(80-leng)
                    tmp.fill(0)
                    if leng<80:
                        tmp[0] = 2
                    else:
                        fl[-1]=2
                        continue
                    bl = True
                    fl = np.concatenate((fl,tmp),0)
                    for k in list(fl):
                        if k>=0 and k<=7999:
                            continue
                        else:
                            bl = False
                            break
                    if bl:
                        self.dataset.append(fl)
                        self.lendict.append(int(splf[1])+1)

            self.dataset = torch.from_numpy(np.array(self.dataset))
            pickle.dump(self.lendict,open("fold2/"+root_dir+"dict.p",'wb'))
            pickle.dump(self.dataset,open("fold2/"+root_dir+"data.p",'wb'))
        else:
            self.dataset = pickle.load(open("fold2/"+root_dir+"data.p",'rb'))
            self.lendict = pickle.load(open("fold2/"+root_dir+"dict.p",'rb'))
    def __len__(self):
        return len(self.lendict)

    def __getitem__(self, idx):
        grsent = self.dataset[idx]
        rnd = torch.randperm(self.lendict[idx])
        _,indices = torch.sort(rnd)
        shuffsent = torch.index_select(grsent[:self.lendict[idx]], 0, rnd, out=None)
        # now pad the rest of the sentence with 2
        tmp = torch.LongTensor(80-self.lendict[idx]).fill_(0)
        shuffsent = torch.cat((shuffsent,tmp),0)
        tmp = torch.LongTensor(80-self.lendict[idx]).fill_(80)
        invgrnd = torch.cat((indices,tmp),0)
        instance = {'shuffsent': shuffsent, 'groundsent': grsent,'groundtruth':invgrnd,'len':self.lendict[idx]}
        return instance


trainset = WordReorderData("train")
valset = WordReorderData("val")
testset = WordReorderData("test")

net.cuda()

# #defining the optim function
optimizer = torch.optim.SGD(net.parameters(), lr=hyper['lr'],momentum=hyper['momentum'], weight_decay=hyper['weightdecay'])
optimizer.zero_grad() 
#this is going to take a step before every epoch starts
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, hyper['lrdecaystepsize'], gamma=hyper['lrdecay'], last_epoch=-1)


#defining dataloaders for train val and test datasets.
trainloader = DataLoader(trainset, batch_size=hyper['batchsize'],shuffle=True, num_workers=4)
valloader = DataLoader(valset, batch_size=hyper['batchsize'],shuffle=False, num_workers=4)
# trainloader = DataLoader(trainset, batch_size=hyper['batchsize'],shuffle=True, num_workers=4)
epoch = 0
nsteps = len(trainset)/hyper['batchsize']
batchsize = hyper['batchsize']

# coding: utf-8

# In[53]:

sentence_gt = '1 3 4'


# In[54]:

sentence_pred = '2 3 1 4'


# In[64]:

regex1 = re.compile('[ ]+')


# In[72]:

def Pairwise_Metric(ground_truth, predicted):
    # Find the skip bigram pairs
    #list_gt = regex1.split(ground_truth)
    #list_pred = regex1.split(predicted)
    list_gt = ground_truth
    list_pred = predicted
    skip_bigrams_gt = set([(list_gt[i], list_gt[j]) for i in range(len(list_gt)) for j in range(i+1, len(list_gt))])
    skip_bigrams_pred = set([(list_pred[i], list_pred[j]) for i in range(len(list_pred)) for j in range(i+1, len(list_pred))])

    # Calculate and return precision, recall, f-score
    P = float(len(skip_bigrams_gt.intersection(skip_bigrams_pred))/len(skip_bigrams_pred))

    R = float(len(skip_bigrams_gt.intersection(skip_bigrams_pred))/len(skip_bigrams_gt))

    fs = 2*P*R/max((P+R),1e-8)
    
    return P,R,fs

def lcs_metric(a, b):
    # The dynamic programming algo for longest common subsequence
    lengths = np.zeros((len(a) + 1, len(b) + 1))

    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1, j+1] = lengths[i, j] + 1
            else:
                lengths[i+1, j+1] = max(lengths[i+1, j], lengths[i, j+1])

    return float(lengths[-1, -1])


def Perfect_Match(ground_truth, predicted):
    # Perfect match b/w ground_truth sequence and predicted sequence
    # list_gt = regex1.split(ground_truth)
    # list_pred = regex1.split(predicted)
    list_gt = ground_truth
    list_pred = predicted
    pm = 0
    for i, gx in enumerate(list_gt):
        if gx == list_pred[i]:
            pm += 1
    pm = pm/float(len(list_pred))
    return pm


 


def perfmetric(calcsent,instance):
    #p-all
    #args-calcsent-batchsizex80 groundsent-batchsizex80
    groundsent = instance['groundtruth'].cuda()
    j2 = groundsent.size()[-1]
    leng = instance['len'].cuda()
    perf = 0
    calcsent = calcsent.data
    tot =int(leng.sum())
    for i in range(hyper['batchsize']):
        #print(calcsent[i,:leng[i]].size(),groundsent[i,:leng[i]].size())
        #print(type(calcsent[i,:leng[i]]),type(groundsent[i,:leng[i]]))
        sb = calcsent[i,:leng[i]]-groundsent[i,:leng[i]]
        perf += (int(leng[i])-len(torch.nonzero(sb)))
    matches = perf
    del groundsent
    del leng
    torch.cuda.empty_cache()
    return perf,tot

def inference(instance,val=False):
    #args shuffset-batchsizex80 groundsent-batchsizex80
    #returns calcsent,loss
    if val:
        groundsent = Variable(instance['groundsent'].cuda(),volatile = True)
        shuffsent = Variable(instance['shuffsent'].cuda(),volatile = True)
    else:
        groundsent = Variable(instance['groundsent'].cuda())
        shuffsent = Variable(instance['shuffsent'].cuda())
    x,calcsent = net.forward(shuffsent)
    calcsent.detach()
    x.detach()
    #print(x.size(),groundsent.view(-1).size())
    loss = criterion(x,groundsent.view(-1))
    return calcsent,loss


def validationbeamm(k):
    metrics = torch.Tensor(5).fill_(0)
    for i_batch,sent_batch in enumerate(valloader):
        if(len(sent_batch['len'])!=hyper['batchsize']):
            continue
        groundsent = sent_batch['groundsent'].cuda()
        shuffsent = sent_batch['shuffsent'].cuda()
        groundtruth = sent_batch['groundtruth'].cuda()
        groundsent = Variable(groundsent)
        shuffsent = Variable(shuffsent)
        groundtruth = Variable(groundtruth)
        x,calcsent = net.forward(shuffsent)
        # x = softy(x).view(hyper['batchsize'],80,81)
        
        #print(x.size(),groundsent.view(-1).size())
        #x->32x81 groundtruth.view(-1)->32*81
        x = x.view(hyper['batchsize'],80,81)
        leng = sent_batch['len'].cuda()    
        
        calcsent = calcsent.data.view(hyper['batchsize'],-1)
        tot =int(leng.sum())
        F = 0
        P = 0
        R = 0
        lcs = 0
        pm = 0
        for i in range(hyper['batchsize']):
            #print(calcsent[i,:leng[i]].size(),groundsent[i,:leng[i]].size())
            #print(type(calcsent[i,:leng[i]]),type(groundsent[i,:leng[i]]))
            probs = x[i].data#80x81
            # print(probs.size())
            # _,indices = probs.topk(2,dim=1)
            # print(indices.size())
            # mask = torch.eq(indices[:,0],torch.cuda.LongTensor(80).fill_(80)).type(torch.cuda.ByteTensor)
            # msel = torch.masked_select(torch.arange(80).type(torch.cuda.LongTensor),mask)
            # calcsent[i][mask] = indices[:,1][mask]

            testleng = leng[i]
            # print(int(testleng))
            calcsent = beamdecode(probs,int(testleng),k)
            # print(calcsent[i])
            # minsent = calcsent.index_select(0,torch.nonzero(calcsent[i]-80).squeeze(1)) 
            #print(minsent)
            # for i in range(80):
            #     if i=
            tmp = torch.index_select(shuffsent.data[i],0,calcsent[:leng[i]])
            a = list(tmp)
            #pada = list(torch.cat((tmp,torch.cuda.LongTensor(80-len(minsent)).fill_(0)),0))
            b = list(torch.index_select(shuffsent.data[i],0,groundtruth[i,:leng[i]].data))
            # sb = calcsent[i,:leng[i]]-groundtruth[i,:leng[i]].data
            # perf += (int(leng[i])-len(torch.nonzero(sb)))
            p,r,f = Pairwise_Metric(b,a)
            P+=p
            R+=r
            F+=f
            lcs+=lcs_metric(b,a)/float(leng[i])
            pm+=Perfect_Match(b,a)
        bmetrics = torch.Tensor([F,P,R,lcs,pm]).div_(hyper['batchsize'])
        print('metrics:[F,P,R,lcs,pm]',list(bmetrics))
        metrics = metrics+bmetrics
        
        # perf,tot = perfmetric(calcsent.view(hyper['batchsize'],-1),sent_batch)
        # totloss+=float(loss.data)
        # totperf+=perf
        # totmatches+=tot
    return metrics.div_(i_batch)
    #eturn float(totperf)/totmatches,float(totloss)/(i_batch)


def beamdecode(probs,leng,width):
    #probs-80x81-torch.cuda.FloatTensor leng-1-int
    #returns list of length 80 with indices.
    #beam width 4000
    #l initialization for length 0
    l = torch.arange(80).type(torch.LongTensor).unsqueeze(1).cuda()
    pr = probs[0]
    for j in range(leng-1):
        #expand
        tmp1 = []
        tmp2 = []
        for i,ls in enumerate(l):
            seti = set(l[i])
            setc = set(torch.arange(81).type(torch.LongTensor))-seti
            for k in setc:
                if k==80:
                    continue
                tmp1.append(list(l[i])+[int(k)])
                tmp2.append(float(pr[i])*float(probs[j+1,int(k)]))                
        pr = tmp2
        l = tmp1
        tmp2 = torch.from_numpy(np.array(pr))
        #indices = 0
        #if(len(tmp2)>=4000):
        _,indices = tmp2.cuda().topk(width)
        #cutoff and make the final length k
        # print(len(l),len(l[0]),len(l[1]),type(l))
        l = torch.from_numpy(np.array(l)).cuda()
        l = torch.index_select(l,0,indices.cuda())
        pr = torch.from_numpy(np.array(pr)).cuda()
        pr = torch.index_select(pr,0,indices.cuda())
    _,indices = torch.max(pr,dim=-1)
    # print(l[int(indices[0])].size(),type(l[int(indices[0])]))
    #return list(torch.cat((l[int(indices[0])],torch.LongTensor(80-leng).fill_(80).cuda()),0))
    return l[int(indices[0])]
def beamsearch(x,leng,k):
    #x-2560x81
    #returns calcsent-32x80
    x = softy(x).view(hyper['batchsize'],80,81)
    #now x has probabilities 
    calcsent = []
    for i in range(hyper['batchsize']):
        probs = x[i]#80x81
        calc = beamdecode(probs,leng[i],k)#80 with indices
        calcsent.append(calc)
    return torch.from_numpy(np.array(calcsent))#32x80 




# def maxdecoder(probs,leng):
#     #probs-80x81-float.cuda leng-1-int
#     #returned a tensor of length leng-long.cuda
#     _,indices = probs.topk(2,dim=1)
#     indices = indices[:leng,:]
#     indices[:,1] = indices[:,1].sub_(80)
#     ind,_ = torch.max(indices,dim=1)

def testbeam(k):
    test = valset[1]
    testdic = test
    testleng = test['len']
    test = test['shuffsent'].cuda()
    test = Variable(test.unsqueeze(0).cuda(),volatile=True)
    #1x80
    x,_ = net.forward(test)
    #x-1x80x81
    x = F.softmax(x).view(1,80,81)
    probs = x[0].data
    # print(int(testleng))
    calcsent = beamdecode(probs,int(testleng),k)
    #print(testleng,calcsent)
    pred = testdic['shuffsent'].cuda().index_select(0,calcsent)
    print(testdic['shuffsent'],pred,testdic['groundsent'])


def validationm(k):
    metrics = torch.Tensor(6).fill_(0)
    for i_batch,sent_batch in enumerate(valloader):
        if(len(sent_batch['len'])!=hyper['batchsize']):
            continue
        groundsent = sent_batch['groundsent'].cuda()
        shuffsent = sent_batch['shuffsent'].cuda()
        groundtruth = sent_batch['groundtruth'].cuda()
        groundsent = Variable(groundsent)
        shuffsent = Variable(shuffsent)
        groundtruth = Variable(groundtruth)
        x,calcsent = net.forward(shuffsent)
        # x = softy(x).view(hyper['batchsize'],80,81)
        loss = criterion(x,groundtruth.view(-1))
        loss = float(loss.data)
        #print(x.size(),groundsent.view(-1).size())
        #x->32x81 groundtruth.view(-1)->32*81
        x = x.view(hyper['batchsize'],80,81)
        leng = sent_batch['len'].cuda()    
        
        
        tot =int(leng.sum())
        fs = 0
        P = 0
        R = 0
        lcs = 0
        pm = 0
        probs = x.data
        probs = probs.view(-1,81)
        _,indices = probs.topk(2,dim=1)
        mask = torch.eq(indices[:,0],torch.cuda.LongTensor(hyper['batchsize']*80).fill_(80)).type(torch.cuda.ByteTensor)
        calcsent[mask] = indices[:,1][mask]
        calcsent = calcsent.data.view(hyper['batchsize'],-1)
        for i in range(hyper['batchsize']):
            #print(calcsent[i,:leng[i]].size(),groundsent[i,:leng[i]].size())
            #print(type(calcsent[i,:leng[i]]),type(groundsent[i,:leng[i]]))
            #80x81
            # print(probs.size())
            
            # print(indices.size())
            
            # msel = torch.masked_select(torch.arange(80).type(torch.cuda.LongTensor),mask)
            

            # testleng = leng[i]
            # print(int(testleng))
            # calcsent = beamdecode(probs,int(testleng),k)
            #print(torch.nonzero(calcsent[i]-80).squeeze(1))
            #minsent = calcsent[i].index_select(0,torch.nonzero(calcsent[i]-80).squeeze(1)) 
            #print(minsent)
            # for i in range(80):
            #     if i=
            tmp = torch.index_select(shuffsent.data[i],0,calcsent[i,:leng[i]])
            a = list(tmp)
            #pada = list(torch.cat((tmp,torch.cuda.LongTensor(80-len(minsent)).fill_(0)),0))
            b = list(torch.index_select(shuffsent.data[i],0,groundtruth[i,:leng[i]].data))
            # sb = calcsent[i,:leng[i]]-groundtruth[i,:leng[i]].data
            # perf += (int(leng[i])-len(torch.nonzero(sb)))
            p,r,f = Pairwise_Metric(b,a)
            P+=p
            R+=r
            fs+=f
            lcs+=lcs_metric(b,a)/float(leng[i])
            pm+=Perfect_Match(b,a)
        bmetrics = torch.Tensor([0,fs,P,R,lcs,pm]).div_(hyper['batchsize'])
        bmetrics[0] = loss
        print('Validation metrics:[loss,F,P,R,lcs,pm] %d'%(i_batch),list(bmetrics))
        metrics = metrics+bmetrics
        
        # perf,tot = perfmetric(calcsent.view(hyper['batchsize'],-1),sent_batch)
        # totloss+=float(loss.data)
        # totperf+=perf
        # totmatches+=tot
    return metrics.div_(i_batch)
    #eturn float(totperf)/totmatches,float(totloss)/(i_batch)


def validation():
    nsteps = len(valset)/hyper['batchsize'] 
    totloss = 0
    totperf = 0
    totmatches = 0
    for i_batch,sent_batch in enumerate(valloader):
        if(len(sent_batch['len'])!=hyper['batchsize']):
            continue
        groundsent = sent_batch['groundsent'].cuda()
        shuffsent = sent_batch['shuffsent'].cuda()
        groundtruth = sent_batch['groundtruth'].cuda()
        groundsent = Variable(groundsent)
        shuffsent = Variable(shuffsent)
        groundtruth = Variable(groundtruth)
        x,calcsent = net.forward(shuffsent)
        #print(x.size(),groundsent.view(-1).size())
        #x->32x81 groundtruth.view(-1)->32*81
        loss = criterion(x,groundtruth.view(-1))
        j2 = groundsent.size()[-1]
        leng = sent_batch['len'].cuda()
        perf = 0    
        calcsent = calcsent.data.view(hyper['batchsize'],-1)
        tot =int(leng.sum())
        for i in range(hyper['batchsize']):
            #print(calcsent[i,:leng[i]].size(),groundsent[i,:leng[i]].size())
            #print(type(calcsent[i,:leng[i]]),type(groundsent[i,:leng[i]]))
            sb = calcsent[i,:leng[i]]-groundtruth[i,:leng[i]].data
            perf += (int(leng[i])-len(torch.nonzero(sb)))
        matches = perf
        # perf,tot = perfmetric(calcsent.view(hyper['batchsize'],-1),sent_batch)
        totloss+=float(loss.data)
        totperf+=perf
        totmatches+=tot
    return float(totperf)/totmatches,float(totloss)/(i_batch)


# validationm(1)

# def l2constraint(modu):
#     #normalize the conv layers
#     cn = torch.norm(modu.weight,p=2,dim=-1).detach() 
#     if cn > hyper['weightnorm']:
#         modu.weight = modu.weight.div(cn.expand_as(modu.weight))


f = open('val2.log','w+')
trperf = 0
trtot = 0
totloss= 0 
net.train()
i_batch = 0
travgloss = 0.0
travgperf = 0.0
for epoch in range(lcontrols['epoch']):
    metrics = torch.Tensor(6).fill_(0)
    trcnt = 0
    for j,sent_batch in enumerate(trainloader):
        if(len(sent_batch['len'])!=hyper['batchsize']): 
            continue
        
        groundsent = sent_batch['groundsent'].cuda()
        shuffsent = sent_batch['shuffsent'].cuda()
        groundtruth = sent_batch['groundtruth'].cuda()
        groundsent = Variable(groundsent)
        shuffsent = Variable(shuffsent)
        groundtruth = Variable(groundtruth)
        x,calcsent = net.forward(shuffsent)
        #print(x.size(),groundsent.view(-1).size())
        #x->32x81 groundtruth.view(-1)->32*81
        loss = criterion(x,groundtruth.view(-1))
        j2 = groundsent.size()[-1]
        leng = sent_batch['len'].cuda()
        perf = 0    
        # calcsent = calcsent.data.view(hyper['batchsize'],-1)
        tot =int(leng.sum())
        
        # for i in range(hyper['batchsize']):
        #     #print(calcsent[i,:leng[i]].size(),groundsent[i,:leng[i]].size())
        #     #print(type(calcsent[i,:leng[i]]),type(groundsent[i,:leng[i]]))
        #     sb = calcsent[i,:leng[i]]-groundtruth[i,:leng[i]].data
        #     perf += (int(leng[i])-len(torch.nonzero(sb)))
        # calcsent = calcsent.data.view(hyper['batchsize'],-1)
        tot =int(leng.sum())
        fs = 0
        P = 0
        R = 0
        lcs = 0
        pm = 0
        trcnt+=1
        calcsent = calcsent.data
        probs = x.data.view(-1,81)
        _,indices = probs.topk(2,dim=1)
        mask = torch.eq(indices[:,0],torch.cuda.LongTensor(hyper['batchsize']*80).fill_(80)).type(torch.cuda.ByteTensor)
        calcsent[mask] = indices[:,1][mask]
        calcsent = calcsent.view(hyper['batchsize'],-1)
        for i in range(hyper['batchsize']):
            #print(calcsent[i,:leng[i]].size(),groundsent[i,:leng[i]].size())
            #print(type(calcsent[i,:leng[i]]),type(groundsent[i,:leng[i]]))
            # probs = x[i].data#80x81
            # print(probs.size())
            # _,indices = probs.topk(2,dim=1)
            # print(indices.size())
            # mask = torch.eq(indices[:,0],torch.cuda.LongTensor(80).fill_(80)).type(torch.cuda.ByteTensor)
            # msel = torch.masked_select(torch.arange(80).type(torch.cuda.LongTensor),mask)
            # calcsent[i][mask] = indices[:,1][mask]

            # testleng = leng[i]
            # print(int(testleng))
            # calcsent = beamdecode(probs,int(testleng),k)
            #print(torch.nonzero(calcsent[i]-80).squeeze(1))
            #minsent = calcsent[i].index_select(0,torch.nonzero(calcsent[i]-80).squeeze(1)) 
            #print(minsent)
            # for i in range(80):
            #     if i=
            tmp = torch.index_select(shuffsent.data[i],0,calcsent[i,:leng[i]])
            a = list(tmp)
            #pada = list(torch.cat((tmp,torch.cuda.LongTensor(80-len(minsent)).fill_(0)),0))
            b = list(torch.index_select(shuffsent.data[i],0,groundtruth[i,:leng[i]].data))
            # sb = calcsent[i,:leng[i]]-groundtruth[i,:leng[i]].data
            # perf += (int(leng[i])-len(torch.nonzero(sb)))
            p,r,f = Pairwise_Metric(b,a)
            P+=p
            R+=r
            fs+=f
            lcs+=lcs_metric(b,a)/float(leng[i])
            pm+=Perfect_Match(b,a)
        bmetrics = torch.Tensor([0,fs,P,R,lcs,pm]).div_(hyper['batchsize'])
        bmetrics[0] = float(loss)
        # print('metrics:[loss,F,P,R,lcs,pm]',list(bmetrics))
        metrics = metrics+bmetrics
        # matches = perf
        # trperf+=perf
        # trtot+=tot
        # totloss+=float(loss.data)
        # avgloss = totloss/(i_batch+1)
        # avgperf = float(trperf)/trtot
        # perf = perf/tot
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 
        i_batch+=1
        print('Training metrics:[epoch,step,loss,F,P,R,lcs,pm] %d %d'%(epoch,i_batch),list(bmetrics))
    travgloss = float(loss)
    trainmetrics = metrics.div_(trcnt)
        # prints currently alive Tensors and Variables
        # m  = 0
        # try:
        #     for obj in gc.get_objects():
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(m,obj.__name__,type(obj), obj.size())
        #             m+=1
        # except Exception:
        #     continue

        # if 1:
        #     # print(shuffsent,groundsent)
        #     # print(shuffsent.size(),groundsent.size(),leng.size())
        #     # print(perfmetric(shuffsent,sent_batch))
        #     # calcsent,loss = inference(sent_batch)
        #     # print(calcsent,loss)
        #     d = trainset[len(trainset)-1]
        #     t = trainset[len(trainset)-2]
        #     print(len(trainset),list(d['groundsent']),d['groundtruth'],d['shuffsent'],d['len'])
        #     break
    if(1):
        epoch+=1
        
        #scheduler.step()
        if epoch%lcontrols['valstepsize']==0:
            #get validation bleu score.
            
            net.eval()
            net.training = False
            valmetrics = validationm(1)
            net.training = True
            net.train()
            torch.save(net.state_dict(),lcontrols['savedir']+'%d-train-%f-%f-%f-%f-%f-%f-val-%f-%f-%f-%f-%f-%f.model5'%(vali,float(trainmetrics[0]),float(trainmetrics[1])\
                ,float(trainmetrics[2]),float(trainmetrics[3]),float(trainmetrics[4]),float(trainmetrics[5]),float(valmetrics[0]),float(valmetrics[1])\
                ,float(valmetrics[2]),float(valmetrics[3]),float(valmetrics[4]),float(valmetrics[5])))
            vali+=1
        trperf = 0
        trtot = 0
        totloss= 0 
        scheduler.step()
