# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:06:34 2020

@author: Jamal Moussa
"""

import csv
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import sys

def load_data(filepath):
    dataset = []
    
    with open(filepath) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row.pop("Lat")
            row.pop("Long")
            dataset.append(row)
            
    return dataset

def calculate_x_y(time_series):
    x = 0
    y = 0
    dates = []
    for day in time_series:
        if day == 'Country/Region' or day == 'Province/State':
            continue
        dates.append(day)
        
    t = len(dates) - 1
    
    if int(time_series[dates[t]]) == 0:
        x = None
        y = None
        return (x,y)
    
    #day with 10 times less cases
    n10 = int(time_series[dates[t]])/10
   
    i = max([day for day in range(len(dates)) if int(time_series[dates[day]]) <= n10])
    
    if i == None:
        x = None
    else:
        x = t - i
        
    #day with 100 times less cases
    n100 = int(time_series[dates[t]])/100
    j = max([day for day in range(len(dates)) if int(time_series[dates[day]]) <= n100])
    
    if j == None:
        y = None
    else:
        y = i - j
        
    return (x,y)

def hac(dataset):
    #number each data point 0-len(dataset)
    cluster = enumerate(dataset)
    dist = pairwise_distances(dataset, metric = 'euclidean')
    np.fill_diagonal(dist, sys.maxsize)
    
    for k in range(1, dist.shape[0]):
        min_val = sys.maxsize
        
        for i in range(0, dist.shape[0]):
            for j in range(0, dist.shape[1]):
                if(dist[i][j] <= min_val):
                    min_val = dist[i][j]
                    rowidx = i 
                    colidx = j
                    
        for i in range(0, dist.shape[0]):
            if(i != colidx):
                temp = min(dist[colidx][i], dist[rowidx][i])
                dist[colidx][i] = temp
                dist[i][colidx] = temp
                
        for i in range(0, dist.shape[0]):
            dist[rowidx][i] = sys.maxsize
            dist[i][rowidx] = sys.maxsize
            