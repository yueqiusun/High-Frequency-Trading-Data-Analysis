import numpy.linalg as la
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime as dt
from pandas import DataFrame
import pandas as pd
from pandas.tools.plotting import scatter_matrix

path = r'/Users/hp/Desktop/code/python/vwap.tbt.20140121'

f = open(path,'r')
lst = list()
with open(path,'r') as myfile:
    for line in myfile:
        a = f.readline()
        a2 = a.strip()
        a3 = a2.split(',')
        b0 = a3[0:1]
        b0.extend(a3[6:11])
        lst.append(tuple(b0))

## build database

conn = sqlite3.connect('my_db1.db')
c = conn.cursor()
c.execute("DROP TABLE IF EXISTS my_db1")
c.execute('''CREATE TABLE my_db1 (SecuCode TEXT, SysTime TEXT, S1P REAL, S1V REAL,B1P REAL, B1V REAL)''')

c.executemany("INSERT INTO my_db1 VALUES (?,?,?,?,?,?)", lst)
conn.commit()
conn.close()

## query

conn = sqlite3.connect('my_db1.db')
c = conn.cursor()
lines = 0
lst = list()
t = ('600196',) # query the information of stock 600000 and order by trading time
# t = ('M000300',)
for row in c.execute('SELECT * FROM my_db1 WHERE \
            SecuCode=? ORDER by SysTime', t):
    lst.append(row)
    lines += 1
conn.close()

# step 1: find the sys time of index
# use pandas dataframe
# transform sys time from string to time

## Pandas processing


l1 = len(lst)

zb1 = np.zeros((l1,1))

t1 = []
t2 = []

for i in range(l1):
    zb1[i] =lst[i][4]
    t1.append(lst[i][1])


t = np.dtype([('datetime', 'O8'), ('close', 'f8')])

dfzb1 = np.zeros(l1,t)

t10 = []
for i in range(l1):
    t10.append(dt.strptime(t1[i], '%Y/%m/%d %H:%M:%S.%f'))

## use t10 of hs300 as index, and use reindex function to obtain price matrix


paths = '/Users/hp/Desktop/Data Mining/data/hs300-2/hs300-2/hs300/'
stock_names = os.listdir(paths)

#load the 300 stock info
paths = '/Users/hp/Desktop/Data Mining/data/hs300-2/hs300-2/hs300/'

stock_names = os.listdir(paths)
stock_names.remove('SH600089.txt')
stock_names.remove('SH600177.txt')
stock_names.remove('SH600958.txt')
stock_names.remove('SH601021.txt')
stock_names.remove('SH601225.txt')
stock_names.remove('SH601969.txt')
stock_names.remove('SH603288.txt')
stock_names.remove('SZ000166.txt')
stock_names.remove('SZ002736.txt')
stock_names.remove('SZ300146.txt')
len_T = len(t10)
price_matx = np.zeros((len_T,300))
cname=['stock','time','pbuy','vbuy','psell'
,'vsale']
for j, name in enumerate(stock_names):
    #strip the stock names

    name=name.strip('SH')
    name=name.strip('.txt')
    name=name.strip('Z')
    
    #read in data
    conn = sqlite3.connect('my_db1.db')
    c = conn.cursor()
    lines = 0
    lst = list()
    t=(name,)
    for row in c.execute('SELECT * FROM my_db1 WHERE \
                SecuCode=? ORDER by SysTime', t):
        lst.append(row)
        lines += 1
    conn.close()
    
    lst=pd.DataFrame(lst)
    lst.columns = cname
    lst.index = lst['time']
    
    lst2 = lst.reindex(t1,\
        method = 'ffill')
    lst3 = lst2.fillna(method = 'bfill')
    price_matx[:,j] =lst3['psell']

price_matx=price_matx[:,0:290]
price_matx=price_matx[:-100,0:290]
# PCA
ret_matx = (price_matx[1:,:]- \
    price_matx[:-1,:])/price_matx[:-1,:]


Sigma = np.cov(ret_matx)


w,v = la.eig(Sigma)

mean_stock=np.mean(ret_matx.T,axis = 0)
score1 = (ret_matx.T-mean_stock).dot(v[:,0:1])
#for the second component
score2 = (ret_matx.T-mean_stock).dot(v[:,1:2])
ss2=np.argsort(score2.ravel())
print '5 largest elements in the principal component directions v2'
for i in ss2[0:5]:
    print stock_names[i]
print '\n'
print '5 smallest elements in the principal component directions v2'
for i in ss2[-5:]:
    print stock_names[i]
print '\n'
#for the third component
score3 = (ret_matx.T-mean_stock).dot(v[:,2:3])
ss3=np.argsort(score3.ravel())
print '5 largest elements in the principal component directions v3'
for i in ss3[0:5]:
    print stock_names[i]
print '5 smallest elements in the principal component directions v3'
for i in ss3[-5:]:
    print stock_names[i]
print '\n'


##k-means
X1 = np.random.randn(200,2)
X2 = np.random.randn(200,2) + [3.5,2.5]

X = np.vstack((X1,X2))

X1 = np.random.randn(200,2)
X2 = np.random.randn(200,2) + [3.5,2.5]

X = np.vstack((X1,X2))

plt.plot(X[:,0],X[:,1],'.')
def ourkmean(X,mu,tol,max_iter):
    # mu is a p*K matrix
    n,p = X.shape
    p,K = mu.shape
    iter = 0
    diff = 100
    dist = np.zeros((n,K))
    VAL_prev = 0
    while iter< max_iter and abs(diff)>tol:
        for j in range(K):
            dist[:,j] = np.sum((X-mu[:,j])**2,axis = 1)
        idx = np.argsort(dist,axis=1)[:,0]
        VAL = 0
        for j in range(K):
            mu[:,j] = np.mean(X[idx==j,:],axis = 0)
            VAL = VAL + np.sum((X[idx==j,:]-mu[:,j])**2)
        diff = VAL - VAL_prev
        VAL_prev = VAL
        iter = iter + 1
        print([iter,VAL])
    return idx
score=np.column_stack((score1,score2))
mu = np.array([[-0.00001,0.00001,0.00001],[0.00001,-0.00001,0.00001]])
tol = 0.00000000000001
max_iter = 1000
idx = ourkmean(score,mu,tol,max_iter)
plt.figure()
plt.hold(1)
plt.plot(X[idx==0,0],X[idx==0,1],'ro')
plt.plot(X[idx==1,0],X[idx==1,1],'bo')
plt.plot(X[idx==2,0],X[idx==2,1],'go')
plt.plot(X[idx==3,0],X[idx==3,1],'yo')
plt.plot(X[idx==4,0],X[idx==4,1],'ko')





#index tracking
y_ret=np.mean(ret_matx,axis=1)

def ourforward(y,x):
    n,p = x.shape
    seq = []
    rest_seq = list(range(p))
    for j in range(p):
        x2 = x[:,seq]
        est_beta = la.inv(x2.T.dot(x2)).dot(x2.T).dot(y) 
        yhat = x2.dot(est_beta)
        Z = y - yhat
        tempx = np.hstack((Z,x[:,rest_seq]))
        corr = np.corrcoef(tempx.T)[0,1:]
        loc = np.argsort(np.abs(corr))[-1]
        seq.append(rest_seq[loc])
        del rest_seq[loc]
    return seq
    
y = y_ret.reshape(2810,1)
x = ret_matx    
seq = ourforward(y,x)    

x2 = x[:,seq[0:50]]
est_beta = la.inv(x2.T.dot(x2)).dot(x2.T).dot(y) 
weight = est_beta/sum(est_beta)
portfolio_ret = x2.dot(weight)

x3 = x[:,seq[-50:]]
est_beta2 = la.inv(x3.T.dot(x3)).dot(x3.T).dot(y) 
weight2 = est_beta2/sum(est_beta2)
portfolio_ret2 = x3.dot(weight2)

ave_ret = np.mean(x,axis = 1)


import matplotlib.pyplot as plt
plt.figure()
plt.hold(1)
plt.plot(np.cumprod(1+y),'--r')
plt.plot(np.cumprod(1+portfolio_ret),'--m')
plt.plot(np.cumprod(1+portfolio_ret2),'--c')

plt.xlim(0,2800)# set axis limits
plt.ylim(1, 1.02)
plt.title('index tracking')
plt.xlabel('x axis')
plt.ylabel('returns')


