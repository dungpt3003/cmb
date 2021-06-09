#!/usr/bin/env python

import numpy as np
from base_solution import Solution

class StandardSolution(Solution):
    def __init__(self, afile, tfile, m, n, lvl, y):
        super().__init__(afile, tfile, m, n, lvl, y)

    def findSolution(self):
        D = self.get_D()
        Q = self.get_Q()

        B = self.get_B()
        BTrans = np.transpose(self.B)
        C = self.get_C()
        y = self.y

        BTransC = BTrans.dot(C)

        BTransCB = BTransC.dot(B)
        leftmost = Q + BTransCB

        rightmost = BTransC.dot(y)
        leftmostInv = np.linalg.inv(leftmost)
        self.u = leftmostInv.dot(rightmost)
        
        result = np.squeeze(self.u)
        return result




