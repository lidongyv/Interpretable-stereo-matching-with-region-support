# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-08-30 16:47:51
# @Last Modified by:   yulidong
# @Last Modified time: 2018-08-30 19:17:25
import torch
import torch.multiprocessing as mp
def add(a,b,c):
    d=a+b
    c+=d.float()
def selfadd(a):
    print('a')
    a+=2
    print(a)
if __name__=='__main__':
    mp.set_start_method('forkserver')
    c=torch.zeros(1,1,150).share_memory_()
    a=torch.arange(150).view_as(c).share_memory_()
    b=torch.arange(150).view_as(c).share_memory_()
    #c=torch.ones(1).share_memory_()
    process=[]
    for i in range(150):
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