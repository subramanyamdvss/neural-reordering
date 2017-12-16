import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import gc
hyper = {'temp':0.01,'lr':0.1,'numepochs':1000,'optim':'Adadelta','batchsize':32,'lrdecaystepsize':30,'lrdecay':0.1,'weightnorm':3}
lcontrols = {'valstepsize':20,'savedir':'models2/','epoch':10000}
torch.backends.cudnn.benchmark= True
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        #self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))--this is how embeddings are defined.
        self.embedding = nn.Embedding(8000,100)
        self.embedding.weight.requires_grad = True
        #load the weights from finwordvecs.npy
        iniarr = np.load(open("finwordvecs.npy",'rb'))
        iniarr = torch.from_numpy(iniarr)
        normini = torch.norm(iniarr,p=2,dim=1,keepdim=True)
        iniarr.div_(normini)
        self.iniarr = iniarr

        #here  a torch.unsqueeze(input,1) is used.
        self.embedding.weight.data.copy_(iniarr)
        #no,of input channels 1, no.of output channels 200,5x100 kernels,stride 2,size of output batchsizex200x38x1,size of input batchsizex1x80x100
        self.conv1 = nn.Conv2d(1,200,(5,100),2)
        #no,of input channels 200, no.of output channels 400,7x1 kernels,stride 2,size of output batchsizex400x16x1,size of input batchsizex200x38x1
        self.conv2 = nn.Conv2d(200,400,(7,1),2)
        #no,of input channels 400, no.of output channels 800,16x1 kernels,stride 1,size of output batchsizex800x1x1,size of input batchsizex400x16x1
        self.conv3 = nn.Conv2d(400,800,(16,1),1)
        #no,of input channels 800, no.of output channels 400,16x1 kernels,stride 1,size of output batchsizex400x16x1,size of input batchsizex800x1x1
        self.deconv1 = nn.ConvTranspose2d(800,400,(16,1),1)
        self.deconv2 = nn.ConvTranspose2d(400,200,(8,1),2)
        self.deconv3 = nn.ConvTranspose2d(200,1,(6,100),2)
        #here the output size becomes batchsizex1x80x100
        # here a squeeze(input,1) is used
        # here a nn.F.normalize is used
        self.fc = nn.Linear(100,8000,bias=False)
        self.fc.weight.requires_grad = True
        self.fc.weight.data.copy_(iniarr)
        #by now output would be bx80x8000
        #apply view(-1,8000)
        #here crossentropy with temperature happenns
        self.training = True
        self.conv1.weight.data.normal_(0.0,0.02)
        self.conv2.weight.data.normal_(0.0,0.02)
        self.conv3.weight.data.normal_(0.0,0.02)
        self.deconv1.weight.data.normal_(0.0,0.02)
        self.deconv2.weight.data.normal_(0.0,0.02)
        self.deconv3.weight.data.normal_(0.0,0.02)
    def forward(self, x):
        # # Max pooling over a (2, 2) window
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # # If the size is a square you can only specify a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = x.view(-1, self.num_flat_features(x))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        #x will be of size batchsizex80
        #embed bx80 to bx80x100
        #dropout  -- torch.nn.functional.dropout(input, p=0.5, training=False, inplace=False)
        x = F.relu(self.embedding(x))
        x = torch.unsqueeze(x,1)
        x = F.dropout(x,p = 0.9,training = self.training)
        #conv1
        x = F.relu(self.conv1(x))
        x = F.dropout(x,p = 0.9,training = self.training)
        x = F.relu(self.conv2(x))
        x = F.dropout(x,p = 0.7,training = self.training)
        x = F.relu(self.conv3(x))
        x = F.dropout(x,p = 0.9,training = self.training)
        x = F.relu(self.deconv1(x))
        x = F.dropout(x,p = 0.9,training = self.training)
        x = F.relu(self.deconv2(x))
        x = F.dropout(x,p = 0.9,training = self.training)
        x = F.relu(self.deconv3(x))
        x = torch.squeeze(x,dim=1)
        #x = F.normalize(x,dim=2)
        x = self.fc(x)
        x = x.view(-1,8000)
        x = x.div(hyper['temp'])
        _,y = torch.max(x,-1)
        return x,y

    

net = Net()
net.zero_grad()

#defining the loss function
#torch.nn.CrossEntropyLoss(weight=None, size_average=True, ignore_index=-100, reduce=True)
criterion = nn.CrossEntropyLoss(ignore_index = 0)
#regularizer - dropout and
#l2 constraint on all parameters

#Dataset Class
class WordReorderData(Dataset):

    def __init__(self, root_dir):
        self.lendict = {}
        self.dataset = []
        if not (os.path.isfile("fold2/"+root_dir+"dict.p") and os.path.isfile("fold2/"+root_dir+"data.p")):
            self.root_dir = "fold1"+'/'+root_dir
            j = 0
            lst = os.listdir(self.root_dir)
            splf = lst[0].split('.')
            self.dataset = np.load(open(self.root_dir+"/"+lst[0],'rb'))
            self.dataset = np.array(self.dataset)
            tmp = np.arange(80-int(splf[1]))
            tmp.fill(0)
            tmp[0] = 2
            self.dataset = np.concatenate((self.dataset,tmp),0)
            self.dataset = [self.dataset]
            self.lendict[0] = int(splf[1])
            for f in lst[1:]:
                splf = f.split('.')
                if int(splf[1]) != 0 :
                        self.lendict[j] = int(splf[1])
                        fl = np.load(open(self.root_dir+"/"+f,'rb'))
                        fl = np.array(fl)
                        tmp = np.arange(80-int(splf[1]))
                        tmp.fill(0)
                        if int(splf[1])!=80:
                            tmp[0] = 2
                        else:
                            fl[-1]=2
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
                        print(j)
                        j = j+1
            self.dataset = torch.from_numpy(np.array(self.dataset))
            pickle.dump(self.lendict,open("fold2/"+root_dir+"dict.p",'wb'))
            pickle.dump(self.dataset,open("fold2/"+root_dir+"data.p",'wb'))
            self.ln = j
        else:
            self.dataset = pickle.load(open("fold2/"+root_dir+"data.p",'rb'))
            self.lendict = pickle.load(open("fold2/"+root_dir+"dict.p",'rb'))
            self.ln = len(self.lendict)
    def __len__(self):
        return self.ln

    def __getitem__(self, idx):
        grsent = self.dataset[idx,:]
        rnd = torch.randperm(self.lendict[idx])
        shuffsent = torch.index_select(grsent[:self.lendict[idx]], 0, rnd, out=None)
        # now pad the rest of the sentence with 2
        tmp = torch.LongTensor(80-self.lendict[idx]).fill_(0)
        if self.lendict[idx] == 80:
            shuffsent[-1] = 2
        else:
            tmp[0] = 2
        shuffsent = torch.cat((shuffsent,tmp),0)
        instance = {'shuffsent': shuffsent, 'groundsent': grsent,'len':self.lendict[idx]}
        return instance


trainset = WordReorderData("train")
valset = WordReorderData("val")
testset = WordReorderData("test")

net.cuda()
#defining the optim function
optimizer = torch.optim.Adadelta([p for p in net.parameters() if p.requires_grad], lr=hyper['lr'], rho=0.9, eps=1e-06, weight_decay=0)
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
    groundsent = instance['groundsent'].cuda()
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
    #print(x.size(),groundsent.view(-1).size())
    loss = F.cross_entropy(x,groundsent.view(-1),ignore_index=0)
    return calcsent,loss

def validation():
    nsteps = len(valset)/hyper['batchsize'] 
    totloss = 0
    totperf = 0
    totmatches = 0
    for i_batch,sent_batch in enumerate(valloader):
        if(len(sent_batch['len'])!=hyper['batchsize']):
            continue
        calcsent,loss = inference(sent_batch,True)
        perf,tot = perfmetric(calcsent.view(hyper['batchsize'],-1),sent_batch)
        totloss+=loss
        totperf+=perf
        totmatches+=tot
        if(i_batch==2*nsteps):
            break
    return float(totperf)/totmatches,float(totloss)/(2*nsteps)
vali = 1


def l2constraint(modu):
    #normalize the conv layers
    cn = torch.norm(modu.weight,p=2,dim=-1).detach() 
    if cn > hyper['weightnorm']:
        modu.weight = modu.weight.div(cn.expand_as(modu.weight))


f = open('val.log','w+')
trperf = 0
trtot = 0
totloss= 0 
net.train()
i_batch = 0
for epoch in range(lcontrols['epoch']):
    for j,sent_batch in enumerate(trainloader):
        if(len(sent_batch['len'])!=hyper['batchsize']): 
            continue
        
        groundsent = sent_batch['groundsent'].cuda()
        shuffsent = sent_batch['shuffsent'].cuda()
        groundsent = Variable(groundsent)
        shuffsent = Variable(shuffsent)
        x,calcsent = net.forward(shuffsent)
        #print(x.size(),groundsent.view(-1).size())
        loss = F.cross_entropy(x,groundsent.view(-1),ignore_index=0)
        j2 = groundsent.size()[-1]
        leng = sent_batch['len'].cuda()
        perf = 0    
        calcsent = calcsent.data.view(hyper['batchsize'],-1)
        tot =int(leng.sum())
        for i in range(hyper['batchsize']):
            #print(calcsent[i,:leng[i]].size(),groundsent[i,:leng[i]].size())
            #print(type(calcsent[i,:leng[i]]),type(groundsent[i,:leng[i]]))
            sb = calcsent[i,:leng[i]]-groundsent[i,:leng[i]].data
            perf += (int(leng[i])-len(torch.nonzero(sb)))
        matches = perf
        trperf+=perf
        trtot+=tot
        totloss+=loss.data
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 
        i_batch+=1
        print(epoch,i_batch,loss.data)
        # prints currently alive Tensors and Variables
        # m  = 0
        # try:
        #     for obj in gc.get_objects():
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(m,obj.__name__,type(obj), obj.size())
        #             m+=1
        # except Exception:
        #     continue

        # if i_batch == 4:
        #     # print(shuffsent,groundsent)
        #     # print(shuffsent.size(),groundsent.size(),leng.size())
        #     # print(perfmetric(shuffsent,sent_batch))
        #     # calcsent,loss = inference(sent_batch)
        #     # print(calcsent,loss)
        #     print(trainset[len(trainset)-1]['groundsent'])
        #     break
    if(1):
        epoch+=1
        pall = trperf/max(float(trtot),1e-8)
        totloss = float(totloss)/max(1e-8,(nsteps-1))
        print("Training-metrics-(epoch,step,pall,avgloss)=(%d,%d,%f,%f)"%(epoch,i_batch,pall,totloss))
        f.write("Training-metrics-(epoch,step,pall,avgloss)=(%d,%d,%f,%f)"%(epoch,i_batch,pall,totloss))
        #scheduler.step()
        if epoch%lcontrols['valstepsize']==0:
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
            torch.save(net.state_dict(),lcontrols['savedir']+'%d-%f-%f.model'%(vali,pall,avgloss))
            vali+=1
