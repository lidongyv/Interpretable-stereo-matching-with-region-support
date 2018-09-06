# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-08-30 16:47:51
# @Last Modified by:   yulidong
# @Last Modified time: 2018-08-30 21:13:04
import torch
import torch.multiprocessing as mp
import time
def add(a,b,c):
    start=time.time()
    d=a+b
    c+=d
    print(time.time()-start)
def selfadd(a):
    print('a')
    a+=2
    print(a)
if __name__=='__main__':
    mp.set_start_method('forkserver')
    c=torch.zeros(1,1,5).float().cuda().share_memory_()
    a=torch.arange(5).float().cuda().view_as(c).share_memory_()
    b=torch.arange(5).float().cuda().view_as(c).share_memory_()
    #c=torch.ones(1).share_memory_()
    process=[]
    start=time.time()
    for i in range(5):
        p=mp.Process(target=add,args=[a[:,:,i],b[:,:,i],c[:,:,i],])
        #p=mp.Process(target=selfadd,args=(c))
        p.daemon=True
        p.start()
        process.append(p)
    for p in process:
        p.join()
    # p=mp.Process(target=add,args=[a,b,c,])
    # p.start()
    # p.join()
    print('running')
    print(a,b,c)
    print(time.time()-start)