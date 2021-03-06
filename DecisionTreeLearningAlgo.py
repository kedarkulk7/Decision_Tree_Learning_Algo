# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 23:12:45 2018

@author: kedar
"""
import math
import pandas as pd
import random
import datetime as dt
import copy
import sys

class Node:
    def __init__(self):
        self.connection='root'
        self.name=''
        self.gain=''
        self.cnodes = []

def getEntropy(data, method,distCount):
    list = data.groupby(data.iloc[:,-1]).count().iloc[:,-1]
    ep = 0
    count=0
    if method == 'Variance':
        ep=1
        for i,a in enumerate(list):
            count +=1
            ep*= (a/sum(list))
    else:
        for i,a in enumerate(list):
            ep-= (a/sum(list))*math.log((a/sum(list)),2)
    if method == 'Variance' and count<distCount:
        ep=0
    return ep,list.index[0]

def getVariance(data):
    list = data.groupby(data.iloc[:,-1]).count().iloc[:,-1]
    ep = 1
    for i,a in enumerate(list):
        ep*= (a/sum(list))
    return ep,list.index[0]

def getGain(data,x,method,distCount):
      m = data.groupby(data.columns[x])
      e2=0
      total = sum(m.count().iloc[:,-1])
      for a,b in m:
          cnt = (b.count().iloc[-1])
          ent = getEntropy(b,method,distCount)
          e2 += (cnt/total)*ent[0]
      return e2,ent[1]

def findBestAttribute(data,e1,method,distCount):
    l = []
    map = {}
    for i,column in enumerate(data):
       if column!=data.columns[-1]:
        ent = getGain(data,i,method,distCount)
        gain = e1-ent[0]
        l.append(gain)
        map[gain]=column
    n = Node()
    if not l:
        n.name = 0
        n.gain = 0
    else:
        n.gain = max(l)
        n.name = map.get(max(l))
    return n

def getNode(data,e1,method,distCount):
        n = findBestAttribute(data,e1,method,distCount)
        cat = data.groupby(n.name)
        data,n,dump=getChilds(cat,n,' ',method,distCount)
        return n

def getChilds(cat,nn,s,method,distCount):
        for c,newdata in cat:
            nx = Node()
            data = newdata.drop(nn.name,axis=1)
            e = getEntropy(data, method,distCount)
            e1 = e[0]
            a=e[1]
            nx = findBestAttribute(data,e1,method,distCount)
            nx.connection = c
            if (nx.gain == 0):
                nx.name=a
            else:
                nx = getChilds(data.groupby(nx.name),nx,s+' ',method,distCount)[1]
            nn.cnodes.append(nx)
        return data,nn,s

def printTree(s,node,nfnc,yn):
  if yn=='yes':
    for nd in node.cnodes:
        print(s+node.name+' = '+str(nd.connection)+' : ',end=''),
        if not nd.cnodes:
            print(str(nd.name))
        else:
            print()
            printTree(s+' ',nd,nfnc,yn)
  else:
    for nd in node.cnodes:
            nfnc=nfnc+1
            nfnc = printTree(s+' ',nd,nfnc,yn)

    return int(nfnc)

def test(row,node):
    val=row[node.name]
    if not node.cnodes:
        return node
    else:
        for a in node.cnodes:
            if val==a.connection:
                a=test(row,a)
                return a

def prunedNode(bn,cnt,pk,fd):
    for a in bn.cnodes:
        for ca in a.cnodes:
            cnt+=1
            if cnt==pk:
                a.cnodes=[]
                new=fd.groupby(bn.name)
                for x,y in new:
                    if a.connection==x:
                        u = y.groupby(y.iloc[:,-1]).count().iloc[:,-1]
                        a.name=u[u==max(u)].index[0]
                        break
            elif cnt<pk:
                a=prunedNode(a,cnt,pk,fd)
    return bn

def validate(vdata,node):
    rightcount=0
    for index,row in vdata.iterrows():
        if row[-1] == test(row,node).name:
            rightcount +=1
    iaccu=100*(rightcount/len(vdata))
    return iaccu

def mainmethod(data,vdata,method,l,k):
 t1=dt.datetime.now()
 e=0
 distCount=len(data.groupby(data.columns[-1]))
 e = getEntropy(data,method,distCount)
 e1 = e[0]
 a=e[1]
 nfnc=1
 node = Node()
 if e1==0:
     node.name=a
     print('The output is',a)
 else:
     dd = data
     node=getNode(dd,e1,method,distCount)
     nfnc=printTree('',node,nfnc,'no')
 t2=dt.datetime.now()
 print('Training time :',t2-t1,'sec')
 iaccu=validate(vdata,node)
 t3=dt.datetime.now()
 print('Validating time :',t3-t2,'sec')

 bn = Node()
 inbn=Node()
 prbn=Node()
 bn = copy.deepcopy(node)
 inbn=copy.deepcopy(bn)
 prbn=copy.deepcopy(inbn)

 for ll in range(l):
     m = random.randint(1,k)
     for mm in range(m):
         pk=random.randint(2,nfnc)
         cnt = 1
         pn=prunedNode(inbn,cnt,pk,vdata)
         iac=validate(vdata,pn)
         if iac>iaccu:
             iaccu = iac
             inbn=copy.deepcopy(pn)
             prbn=copy.deepcopy(inbn)
             nfnc=1
             nfnc=printTree('',bn,nfnc,'n')
         else:
           inbn = copy.deepcopy(prbn)


 t4=dt.datetime.now()
 print('Pruning time :',t4-t3,'sec')
 print('Maximum accuracy achieved in',method,iaccu,'%')
 return inbn

l = int(sys.argv[1])
k = int(sys.argv[2])
data = pd.read_csv(str(sys.argv[3]))
vdata = pd.read_csv(str(sys.argv[4]))
tdata = pd.read_csv(str(sys.argv[5]))
yn = str(sys.argv[6]).lower()

print('----------------------------Information Gain------------------------------------')
bn = mainmethod(data,vdata,'Entropy',int(l),int(k))
if yn=='yes':
  print('Decision Tree')
  printTree('',bn,int(0),yn)
iaccu=validate(tdata,bn)
print('Final accuracy for test set is',iaccu,'%')

print('------------------------------------Variance Impurity----------------------------------')
bn = mainmethod(data,vdata,'Variance',int(l),int(k))
if yn=='yes':
  print('Decision Tree')
  printTree('',bn,int(0),yn)
iaccu=validate(tdata,bn)
print('Final accuracy for test set is',iaccu,'%')