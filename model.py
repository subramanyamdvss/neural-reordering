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
hyper = {'temp':0.01,'lr':0.1,'numepochs':1000,'optim':'momentum','batchsize':32,'lrdecaystepsize':30,'lrdecay':0.1,'weightnorm':3,'momentum':0.9,'weightdecay':0.0}
lcontrols = {'valstepsize':20,'savedir':'models2/','epoch':10000}
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
        if(idx == 88833):
            print(self.lendict[idx],grsent)
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
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, hyper['lrdecaystepsize'], gamma=hyper['lrdecay'], last_epoch=-1)


#defining dataloaders for train val and test datasets.
trainloader = DataLoader(trainset, batch_size=hyper['batchsize'],shuffle=True, num_workers=4)
valloader = DataLoader(valset, batch_size=hyper['batchsize'],shuffle=False, num_workers=4)
# trainloader = DataLoader(trainset, batch_size=hyper['batchsize'],shuffle=True, num_workers=4)
epoch = 0
nsteps = len(trainset)/hyper['batchsize']
batchsize = hyper['batchsize']

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




# def l2constraint(modu):
#     #normalize the conv layers
#     cn = torch.norm(modu.weight,p=2,dim=-1).detach() 
#     if cn > hyper['weightnorm']:
#         modu.weight = modu.weight.div(cn.expand_as(modu.weight))

vali = 1
f = open('val2.log','w+')
trperf = 0
trtot = 0
totloss= 0 
net.train()
i_batch = 0
travgloss = 0.0
travgperf = 0.0
for epoch in range(lcontrols['epoch']):
    
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
        calcsent = calcsent.data.view(hyper['batchsize'],-1)
        tot =int(leng.sum())
        for i in range(hyper['batchsize']):
            #print(calcsent[i,:leng[i]].size(),groundsent[i,:leng[i]].size())
            #print(type(calcsent[i,:leng[i]]),type(groundsent[i,:leng[i]]))
            sb = calcsent[i,:leng[i]]-groundtruth[i,:leng[i]].data
            perf += (int(leng[i])-len(torch.nonzero(sb)))
        matches = perf
        trperf+=perf
        trtot+=tot
        totloss+=float(loss.data)
        avgloss = totloss/(i_batch+1)
        avgperf = float(trperf)/trtot
        perf = perf/tot
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 
        i_batch+=1
        print("(epoch,step,loss,perf,avgloss,avgperf)",epoch,i_batch,float(loss.data),perf,avgloss,avgperf)
        travgloss = avgloss
        travgperf = avgperf
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
        if epoch%lcontrols['valstepsize']!=0:
            #get validation bleu score.
            
            net.eval()
            net.training = False
            pall,avgloss = validation()
            net.training = True
            net.train()
            print("Validation-metrics-(vali,epoch,step,pall,avgloss)=(%d,%d,%d,%f,%f)"%(vali,epoch,i_batch,pall,avgloss))
            f.write("Validation-metrics-(vali,epoch,step,pall,avgloss)=(%d,%d,%d,%f,%f)"%(vali,epoch,i_batch,pall,avgloss))
            print("saving-model....")   
            f.write("saving-model....")
            torch.save(net.state_dict(),lcontrols['savedir']+'%d-train-%f-%f-val-%f-%f.model4'%(vali,travgperf,travgloss,pall,avgloss))
            vali+=1
