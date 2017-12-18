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
