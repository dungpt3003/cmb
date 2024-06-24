#!/usr/bin/env python

import numpy as np
from base_solution import Solution

class StandardSolution(Solution):
    def __init__(self, afile, tfile, m, n, lvl, y):
        super().__init__(afile, tfile, m, n, lvl, y)

    def findSolution(self):
        # Q = self.get_Q()
        # B = self.get_B()
        print("BTrans")
        BTrans = np.transpose(self.B)
        C = self.get_C()
        # y = self.y
        print("BTransC")
        BTransC = BTrans.dot(self.C)

        print("BTransCB")
        BTransCB = BTransC.dot(self.B)
        print("Q+BTCB")
        leftmost = self.Q + BTransCB
        with open("output/q_star_n_{0}.npy".format(self.num_lvl), "wb") as f:
            np.save(f, leftmost)
        # # rightmost = BTransC.dot(y)
        # # leftmostInv = np.linalg.inv(leftmost)
        # # self.u = leftmostInv.dot(rightmost)
        
        # #result = np.squeeze(self.u)
        result = np.array([0])
        return result




