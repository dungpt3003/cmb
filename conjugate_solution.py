#!/usr/bin/env python
# Conjugate Gradient Method

import numpy as np
from base_solution import Solution
from utils import find_neighbors
import threading

class DThread(threading.Thread):
    def __init__(self,tid,x,res,tsum,dsize):
        threading.Thread.__init__(self)
        self.x = x
        self.res = res
        (N,n) = x.shape
        self.N = N
        self.n = n
        self.threadID = tid
        self.tsum = tsum
        self.Dsize = dsize

    def run(self):
        for i in range(self.threadID,self.N,self.tsum):
            neighs = find_neighbors(i,self.Dsize[0],self.Dsize[1])
            neighs_len = len(neighs)
            for j in range(0,self.n):
                for idx in neighs:
                    self.res[i][j] += self.x[idx][j]
                self.res[i][j] -= (neighs_len * self.x[i][j])

class ConjugateSolution(Solution):
    def __init__(self,afile,tfile,m,n,lvl,y,toler,threadnum):
        super().__init__(afile,tfile,m,n,lvl,y)
        self.toler = toler
        self.b = None
        self.ATrans = np.transpose(self.A)
        self.A_TTA = self.ATrans.dot(self.T.dot(self.A))
        self.threadnum = threadnum

    def DmulX(self, x):
        # x can be a N*z matrix,z can be bigger than 1
        (row,col) = x.shape
        res = np.zeros((self.num_N,col))
        threads = []
        for i in range(0,self.threadnum):
            t = DThread(i, x, res, self.threadnum, self.Dsize)
            t.start()
            threads.append(t)
        for th in threads:
            th.join()
        return res

    def getb(self):
        Y = np.reshape(self.y,(self.num_N, self.num_m), order='F')
        # b = reshape(NYTA)
        TA = self.T.dot(self.A)
        YTA = Y.dot(TA)
        #B = self.N.dot(YTA)
        B = YTA

        self.b = np.reshape(B,(self.num_n * self.num_N), order='F')
        return self.b

    def calQx(self, x):
        #return matvec_reshape(self.P, self.DSqure, x, True)
        X = np.reshape(x, (self.num_N,self.num_n),order='F')
        X = self.DmulX(X)
        X = self.DmulX(X)
        return np.reshape(X,(self.num_N*self.num_n),order='F')


    def calBTCBx(self,x):
        X = np.reshape(x, (self.num_N, self.num_n), order='F')
        #X = self.N.dot(X.dot(self.A_TTA))
        # N is identity matrix
        X = X.dot(self.A_TTA)
        return np.reshape(X,(self.num_N * self.num_n),order='F')

    def findSolution(self):
        # init b(Gx = b)
        b = self.getb()
        #self.getDSqure()
        n = len(b)
        x = np.ones(n)

        r = b - self.calQx(x) - self.calBTCBx(x)
        p = r

        r_k_norm = np.dot(r,r)
        origin_r_norm = r_k_norm
        for i in range(2*n):
            #rold = r
            q = self.calQx(p) + self.calBTCBx(p)
            alpha = r_k_norm / np.dot(p,q)
            x += alpha * p
            if i % 50 == 0:
                r = b - self.calQx(x) - self.calBTCBx(x)
            else:
                r -= alpha * q

            r_k1_norm = np.dot(r,r)
            beta = r_k1_norm/r_k_norm
            r_k_norm = r_k1_norm
            #if r_k1_norm < 1e-10 * origin_r_norm:
            if r_k1_norm < 1e-5:
                #newr = b - self.calQx(x) - self.calBTCBx(x)
                #newr_norm = np.dot(newr,newr)
                #print('Itr:', i, r_k1_norm,newr_norm)
                print('Itr:', i, r_k1_norm)
                break
            p = r + beta * p
        return x