# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

#calculate the distance between two points
def distance(index1, index2, K):
    obj1 = Sample[index1]
    obj2 = Sample[index2]
    if K == 2:
        return np.linalg.norm(obj1 - obj2, ord=1)  # manhatten distance
    elif K == 3:
        return np.linalg.norm(obj1 - obj2, ord=2)  # euclidean distance

#get the neighbour point and the corresponding distance for one point
def get_neighbours(num, N, K):
    dists = []
    neighbours = []
    all_neighbour_point = []
    for index in range(N):
        if index == num:
            continue
        else:
            dist = distance(num, index, K)   #num -- index of test, index -- index of train
            dists.append((num, index, dist))
    dists.sort(key=lambda x:x[2]) #sort by the second keyword, which is 'dist'
    neigh = dists[:K]
    distK = neigh[K-1][2]
    for item in dists:
        if item[2] <= distK:
            neighbours.append(item)
            all_neighbour_point.append(item[1])
    return neighbours, distK, all_neighbour_point   # k-distance neighborhood of o

def reachdist_k(point_o, point_o1, K):
    disto_o1 = distance(point_o, point_o1, K)
    two_dist = np.array((disto_o1, DistK[point_o]))
    return np.max(two_dist)   #return the larger one


def lrd_K(Nk, point_o1, neighbour_points):
    numer = Nk[point_o1]  #numerator
    Denom = 0   #denominator
    lrd = 0
    get_list = neighbour_points[point_o1]
    for ele in get_list:
        Denom += reachdist_k(ele, point_o1, K)
    lrd = numer / Denom
    return lrd

#calculate the LOF for one point
def LOF_K(point_o1, neighbour_points, lrd, Nk, K):
    get_list = neighbour_points[point_o1]
    Denom = Nk[point_o1]**2
    numer = 0
    sum_lrd = 0
    for item in get_list:
        numer += reachdist_k(item, point_o1, K)
        sum_lrd += lrd[item]
    value = sum_lrd * numer / Denom
    return value

#run the whole process
def run(K, ini_data):
    global Sample
    global DistK
    N = len(ini_data)
    Sample = []
    Nk = []
    DistK = np.zeros(shape=(N, ))
    neighbours = []
    neighbour_points = []
    lrd = []
    lof_list = []
    Sample = np.array(ini_data.iloc[:, 1:])
    for index in range(N):
        nei, DistK[index], nei_points = get_neighbours(index, N, K)
        neighbours.append(nei)
        neighbour_points.append(nei_points)  #A collection of adjacent points for each point
        Nk.append(len(nei))   #the number of adjacent points for each point

    for i in range(N):
        lrd.append(lrd_K(Nk, i, neighbour_points)) # calculate the lrdk(o)

    for j in range(N):
        lof = LOF_K(j, neighbour_points,lrd, Nk, K)
        lof_list.append((lof, Sample[j], j))
    lof_list.sort(key=lambda x:x[0], reverse=True)
    result = pd.DataFrame(lof_list, columns=['LOF', 'outliers', 'id'])
    print result.iloc[0:5, :]

if __name__ == '__main__':
    ini_data = pd.read_csv("Q2Q3_input.csv")
    K = int(raw_input("input the value of K:\n"))
    if K == 2:
        print "(Manhattan Distance) Top 5 outliers are:"
        run(K, ini_data)
    elif K == 3:
        print "(Euclidean Distance) Top 5 outliers are:"
        run(K, ini_data)
    else:
        print "Wrong value of K !"


















