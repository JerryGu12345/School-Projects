import numpy as np 
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import tree

n1=2000
n2=600
k1=95
k2=46
B=125
N=100
np.random.seed(1)

#training data covariates

X1=np.concatenate((np.random.randn(n1,5),np.random.randint(0,3,(n1,5))),1)

X2=np.empty((n2,500))
for i in range(n2):
  X2[i,0]=np.random.randn()
  for j in range(1,500):
    X2[i,j]=np.random.randn()+X2[i,j-1]*0.15
print(" ")

#test data covariates

X1t=np.concatenate((np.random.randn(N,5),np.random.randint(0,3,(N,5))),1)

X2t=np.empty((N,500))
for i in range(N):
  X2t[i,0]=np.random.randn()
  for j in range(1,500):
    X2t[i,j]=np.random.randn()+X2t[i,j-1]*0.15
print(" ")

#permutation test

def importance(X,Xt,Y,Yt,i,k,type="regression",method="permtest"):
  n=X.shape[0]
  N=Xt.shape[0]
  Xpi=np.copy(X)
  np.random.shuffle(Xpi[:,i])
  T=np.empty((B,N))
  Tpi=np.empty((B,N))
  for b in range(B):
    samp=np.random.choice(n,k,False)
    if (type=="regression"):
      t=tree.DecisionTreeRegressor(min_samples_leaf=10).fit(X[samp],Y[samp])
    else:
      t=tree.DecisionTreeClassifier(min_samples_leaf=10).fit(X[samp],Y[samp])
    if (type=="regression"):
      tpi=tree.DecisionTreeRegressor(min_samples_leaf=10).fit(Xpi[samp],Y[samp])
    else:
      tpi=tree.DecisionTreeClassifier(min_samples_leaf=10).fit(Xpi[samp],Y[samp])
    T[b]=t.predict(Xt)
    Tpi[b]=tpi.predict(Xt)
  meanT=sum(T)/B
  meanTpi=sum(Tpi)/B
  #plt.scatter(Xt[:,i],Yt)
  #plt.scatter(Xt[:,i],meanT)
  #plt.scatter(Xt[:,i],meanTpi)
  #plt.show()
  #plt.clf()
  mse0=sum((meanT-Yt)**2)/N
  mse0pi=sum((meanTpi-Yt)**2)/N
  TTpi=np.concatenate((T,Tpi))
  d=0
  for k in range(N):
    perm=np.random.choice(2*B,2*B,False)
    Tstar=TTpi[perm[0:B]]
    Tpistar=TTpi[perm[B:2*B]]
    meanTstar=sum(Tstar)/B
    meanTpistar=sum(Tpistar)/B
    msej=sum((meanTstar-Yt)**2)/N
    msejpi=sum((meanTpistar-Yt)**2)/N
    d+=(mse0pi-mse0<=msejpi-msej)
  return (d+1)/(N+1)

#model 1
beta=10
for i in [0,1,5,6]:
  rejects=np.zeros(9)
  J=np.linspace(0.005,2.25,9)
  for j in range(9):
    sig=10/J[j]
    Y=np.empty(n1)
    for h in range(n1):
      Y[h]=beta*X1[h,0]+beta*(X1[h,5]==2)+np.random.randn()*sig
    Yt=np.empty(N)
    for h in range(N):
      Yt[h]=beta*X1t[h,0]+beta*(X1t[h,5]==2)+np.random.randn()*sig
    for h in range(100):
      rejects[j]+=(importance(X1,X1t,Y,Yt,i,k1)<0.05)
  plt.plot(J,rejects/100)
  plt.title("variable importance in linear model for X"+str(i+1))
  plt.ylabel("probability of rejection")
  plt.xlabel("coefficient to sigma ratio")
  plt.ylim([-0.01,1.01])
  plt.show()
  plt.clf()
print(" ")

#model 2
beta=10
for i in [2,4,6,8]:
  rejects=np.zeros(9)
  J=np.linspace(0.005,2.25,9)
  for j in range(9):
    sig=10/J[j]
    Y=np.empty(n1)
    for h in range(n1):
      Y[h]=beta*np.sin(np.pi*(X1[h,6]==2)*X1[h,0])+2*beta*(X1[h,2]-0.05)**2+beta*X1[h,3]+beta*X1[h,1]+np.random.randn()*sig
    Yt=np.empty(N)
    for h in range(N):
      Yt[h]=beta*np.sin(np.pi*(X1t[h,6]==2)*X1t[h,0])+2*beta*(X1t[h,2]-0.05)**2+beta*X1t[h,3]+beta*X1t[h,1]+np.random.randn()*sig
    for h in range(100):
      rejects[j]+=(importance(X1,X1t,Y,Yt,i,k1)<0.05)
  plt.plot(J,rejects/100)
  plt.title("variable importance in MARS model for X"+str(i+1))
  plt.ylabel("probability of rejection")
  plt.xlabel("coefficient to sigma ratio")
  plt.ylim([-0.01,1.01])
  plt.show()
  plt.clf()
print(" ")

#model 3
for i in [0,1,499]:
  rejects=np.zeros(8)
  J=np.linspace(0.01,2.5,8)
  for j in range(8):
    beta=J[j]
    Y=np.empty(n2)
    for h in range(n2):
      Y[h]=(np.random.rand()<1/(1+np.exp(-beta*sum(X2[h,1:5]))))
    Yt=np.empty(N)
    for h in range(N):
      Yt[h]=(np.random.rand()<1/(1+np.exp(-beta*sum(X2t[h,1:5]))))
    for h in range(100):
      rejects[j]+=(importance(X2,X2t,Y,Yt,i,k2,"classification")<0.05)
  plt.plot(J,rejects/100)
  plt.title("variable importance in correlated logit model for X"+str(i+1))
  plt.ylabel("probability of rejection")
  plt.xlabel("coefficient")
  plt.ylim([-0.01,1.01])
  plt.show()
  plt.clf()
print(" ")


np.random.seed(1)

#OOB variable importance
def oobvi(X,Y,i,type="regression"):
  n=X.shape[0]
  Xpi=np.copy(X)
  np.random.shuffle(Xpi[:,i])
  vis=np.ones(n,dtype="bool")
  samp=np.random.choice(n,n)
  for i in range(n):
    vis[samp[i]]=False
  if (type=="regression"):
    t=tree.DecisionTreeRegressor(min_samples_leaf=10).fit(X[samp],Y[samp])
  else:
    t=tree.DecisionTreeClassifier(min_samples_leaf=10).fit(X[samp],Y[samp])
  if (type=="regression"):
    tpi=tree.DecisionTreeRegressor(min_samples_leaf=10).fit(Xpi[samp],Y[samp])
  else:
    tpi=tree.DecisionTreeClassifier(min_samples_leaf=10).fit(Xpi[samp],Y[samp])
  Xt=X[vis]
  Yt=Y[vis]
  N=Xt.shape[0]
  mse0=sum((t.predict(Xt)-Yt)**2)/N
  mse0pi=sum((tpi.predict(Xt)-Yt)**2)/N
  return mse0pi-mse0

#model 1
beta=10
for i in [0,1,5,6]:
  upper=np.zeros(8)
  lower=np.zeros(8)
  J=np.linspace(0.005,2.25,9)[1:]
  for j in range(8):
    sig=10/J[j]
    Y=np.empty(n1)
    for h in range(n1):
      Y[h]=beta*X1[h,0]+beta*(X1[h,5]==2)+np.random.randn()*sig
    diff=np.empty(100)
    for h in range(100):
      diff[h]=oobvi(X1,Y,i)
    diff=np.sort(diff)
    upper[j]=(diff[96]+diff[97])/2
    lower[j]=(diff[2]+diff[3])/2
  plt.plot(J,upper)
  plt.plot(J,lower)
  plt.plot(J,np.zeros(8),color="black")
  plt.title("OOB var importance in linear model for X"+str(i+1))
  plt.ylabel("95%CI of variable importance")
  plt.xlabel("coefficient to sigma ratio")
  plt.show()
  plt.clf()
print(" ")

#model 2
beta=10
for i in [2,4,6,8]:
  upper=np.zeros(8)
  lower=np.zeros(8)
  J=np.linspace(0.005,2.25,9)[1:]
  for j in range(8):
    sig=10/J[j]
    Y=np.empty(n1)
    for h in range(n1):
      Y[h]=beta*np.sin(np.pi*(X1[h,6]==2)*X1[h,0])+2*beta*(X1[h,2]-0.05)**2+beta*X1[h,3]+beta*X1[h,1]+np.random.randn()*sig
    diff=np.empty(100)
    for h in range(100):
      diff[h]=oobvi(X1,Y,i)
    diff=np.sort(diff)
    upper[j]=(diff[96]+diff[97])/2
    lower[j]=(diff[2]+diff[3])/2
  plt.plot(J,upper)
  plt.plot(J,lower)
  plt.plot(J,np.zeros(8),color="black")
  plt.title("OOB var importance in MARS model for X"+str(i+1))
  plt.ylabel("95%CI of variable importance")
  plt.xlabel("coefficient to sigma ratio")
  plt.show()
  plt.clf()
print(" ")

#model 3
for i in [0,1,499]:
  upper=np.zeros(8)
  lower=np.zeros(8)
  J=np.linspace(0.01,2.5,8)
  for j in range(8):
    beta=J[j]
    Y=np.empty(n2)
    for h in range(n2):
      Y[h]=(np.random.rand()<1/(1+np.exp(-beta*sum(X2[h,1:5]))))
    diff=np.empty(100)
    for h in range(100):
      diff[h]=oobvi(X2,Y,i,"classification")
    diff=np.sort(diff)
    upper[j]=(diff[96]+diff[97])/2
    lower[j]=(diff[2]+diff[3])/2
  plt.plot(J,upper)
  plt.plot(J,lower)
  plt.plot(J,np.zeros(8),color="black")
  plt.title("OOB var importance in correlated logit model for X"+str(i+1))
  plt.ylabel("95%CI of variable importance")
  plt.xlabel("coefficient")
  plt.show()
  plt.clf()
print(" ")


