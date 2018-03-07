# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import copy
k = 6

def E_STEP(ini_data, c1, c2, N):
    global W
    global Sample   # 6 given clickstream events
    global SSE
    SSE = 0
    W = np.zeros((2, N))
    Sample = []

    for index in range(N):
        Denom = 0   #denominator
        dist2 = 0   #numerator
        dist1 = 0
        rowD = (ini_data.iloc[index, 1:]).values.tolist() #each row content, float type
        Sample.append(rowD)
        for i in range(k):
            dist2 += (rowD[i]-c2[i])**2
            dist1 += (rowD[i]-c1[i])**2
        Denom = dist1 + dist2
        W[0, index] = dist2/Denom   # Wi1
        W[1, index] = dist1/Denom   # Wi2
        SSE += (W[0, index]**2 * dist1 + W[1, index]**2 * dist2)


def M_STEP(Cen1, Cen2, N):
    global W
    global Sample
    global X
    X = np.zeros((2, k))  #save center
    Denom1 = 0 #calculate the denominator of the ci formula
    Denom2 = 0
    for index in range(N):
        point = Sample[index]
        Denom1 += W[0, index]**2  #denominator of c1
        Denom2 += W[1, index]**2  #denominator of c2
        for i in range(k):
            X[0, i] += point[i] * (W[0, index]**2)
            X[1, i] += point[i] * (W[1, index]**2)

    for i in range(k):
        Cen1[i] = X[0, i]/Denom1
        Cen2[i] = X[1, i]/Denom2
    return Cen1, Cen2


def run(ini_data, c1, c2, N, iter_num, terminSum):
    Cen1 = c1
    Cen2 = c2
    for i in range(iter_num):
        Old_cen1 = copy.copy(Cen1)
        Old_cen2 = copy.copy(Cen2)
        E_STEP(ini_data, Cen1, Cen2, N)
        Cen1, Cen2 = M_STEP(Cen1, Cen2, N)
        print "\nIteration", i+1
        print "Center1:", Cen1
        print "Center2:", Cen2
        print "SSE value is:", SSE
        condition = sum(abs(Old_cen1 - Cen1)) + sum(abs(Old_cen2 - Cen2))
        if condition < terminSum:
            print "\nThe total number of iteration is:", i+1
            print "The final converged centers are:"
            print "Center1:", Cen1
            print "Center2:", Cen2
            break


if __name__ == '__main__':
    ini_data = pd.read_csv("Q2Q3_input.csv")
    iter_num = 50    #maximum iteration
    USER = ini_data.iloc[:, 0]
    c1 = np.array([1, 1, 1, 1, 1, 1], dtype=float)
    c2 = np.array([0, 0, 0, 0, 0, 0], dtype=float)
    N = len(USER)    #length of sample
    terminSum = 0.001  # terminate condition
    run(ini_data, c1, c2, N, iter_num, terminSum)






