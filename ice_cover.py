# -*- coding: utf-8 -*-
"""
P8 Ice REgression
CS 540
April 16, 2020
@author: Jamal Moussa
"""
import math
import random

def get_dataset():
    dataset = []
    x = [i for i in range(1855, 2020)]
    y = [118, 151, 121, 96, 110, 117, 132, 104, 125, 118, 125, 123, 110, 127,
         131, 99, 126, 144, 136, 126, 91, 130, 62, 112, 99, 161, 78, 124, 119, 
         124, 128, 131, 113, 88, 75, 111, 97, 112, 101, 101, 91, 110, 100, 130, 
         111, 107, 105, 89, 126, 108, 97, 94, 83, 106, 98, 101, 108, 99, 88, 
         115, 102, 116, 115, 82, 110, 81, 96, 125, 104, 105, 124, 103, 106, 96, 
         107, 98, 65, 115, 91, 94, 101, 121, 105, 97, 105, 96, 82, 116, 114, 92, 
         98, 101, 104, 96, 109, 122, 114, 81, 85, 92, 114, 111, 95, 126, 105, 
         108, 117, 112, 113, 120, 65, 98, 91, 108, 113, 110, 105, 97, 105, 107,
         88, 115, 123, 118, 99, 93, 96, 54, 111, 85, 107, 89, 87, 97, 93, 88, 
         99, 108, 94, 74, 119, 102, 47, 82, 53, 115, 21, 89, 80, 101, 95, 66, 
         106, 97, 87, 109, 57, 87, 117, 91, 62, 65, 94, 86, 70]
    
    dataset = [[x[i],y[i]] for i in range(len(x))]
    
    return dataset

def print_stats(dataset):
    size = len(dataset)
    mean = sum([dataset[i][1] for i in range(size)]) / size
    diff = sum([(dataset[i][1] - mean)**2 for i in range(size)])
    diff = diff / (size - 1)
    sdeviation = math.sqrt(diff)
    
    size = str(round(size, 2))
    mean = str(round(mean, 2))
    sdev = str(round(sdeviation, 2))
    
    print(size)
    print(mean)
    print(sdev)
    
def regression(beta_0, beta_1):
    data = get_dataset()
    n = len(data)
    mse = sum([(beta_0 + beta_1 * data[i][0] - data[i][1])**2 for i in range(n)]) / n
    
    mse = round(mse, 2)
    return mse

def gradient_descent(beta_0, beta_1):
    data = get_dataset()
    n = len(data)
    dbeta_0 = 2 * sum([(beta_0 + beta_1 * data[i][0] - data[i][1]) for i in range(n)]) / n
    dbeta_1 = 2 * sum([data[i][0] * (beta_0 + beta_1 * data[i][0] - data[i][1]) for i in range(n)]) / n
    
    dbeta_0 = round(dbeta_0, 2)
    dbeta_1 = round(dbeta_1, 2)
    
    return (dbeta_0, dbeta_1)

def iterate_gradient(T, eta):
    beta_0 = 0
    beta_1 = 0
    beta_vector = []
    
    for t in range(T):
        gradient = gradient_descent(beta_0, beta_1)
        beta_0 = beta_0 - eta * (gradient[0])
        beta_1 = beta_1 - eta * (gradient[1])
        reg = regression(beta_0, beta_1)
        beta_vector.append((t+1, beta_0, beta_1, reg))
        
    for value in beta_vector:
        line = str(round(value[0], 2)) + " " + str(round(value[1], 2)) + " " + str(round(value[2], 2)) + " " + str(round(value[3], 2)) 
        print(line)
        
def compute_betas():
    data = get_dataset()
    n = len(data)
    av_x = sum([data[i][0] for i in range(n)]) / n
    av_y = sum([data[i][1] for i in range(n)]) / n
    
    beta_1 = sum([(data[i][0] - av_x) * (data[i][1] - av_y) for i in range(n)]) / sum([(data[i][0] - av_x)**2 for i in range(n)])
    beta_0 = av_y - beta_1 * av_x
    
    mse = regression(beta_0, beta_1)
    
    return (beta_0, beta_1, mse)

def predict(year):
    beta = compute_betas()
    
    return round((beta[0] + beta[1] * year), 2)

def regression_normalized(beta_0, beta_1):
    data = get_dataset()
    n = len(data)
    
    x = [data[i][0] for i in range(n)]
    av_x = sum(x) / n
    
    std = math.sqrt(sum([(x[i] - av_x)**2 for i in range(n)]) / (n - 1))
    
    for i in range(n):
        x[i] = (x[i] - av_x) / std
        
    mse = sum([(beta_0 + beta_1 * x[i] - data[i][1])**2 for i in range(n)]) / n
    
    
    return mse

def gradient_normalized(beta_0, beta_1):
    data = get_dataset()
    n = len(data)
    
    x = [data[i][0] for i in range(n)]
    av_x = sum(x) / n
    
    std = math.sqrt(sum([(x[i] - av_x)**2 for i in range(n)]) / (n - 1))
    
    for i in range(n):
        x[i] = (x[i] - av_x) / std
    
    dbeta_0 = 2 * sum([(beta_0 + beta_1 * x[i] - data[i][1]) for i in range(n)]) / n
    dbeta_1 = 2 * sum([x[i] * (beta_0 + beta_1 * x[i] - data[i][1]) for i in range(n)]) / n
    
    return (dbeta_0, dbeta_1)

def iterate_normalized(T, eta):
    beta_0 = 0
    beta_1 = 0
    beta_vector = []
    
    for t in range(T):
        gradient = gradient_normalized(beta_0, beta_1)
        beta_0 = beta_0 - eta * (gradient[0])
        beta_1 = beta_1 - eta * (gradient[1])
        reg = regression_normalized(beta_0, beta_1)
        beta_vector.append((t+1, beta_0, beta_1, reg))
        
    for value in beta_vector:
        line = str(round(value[0], 2)) + " " + str(round(value[1], 2)) + " " + str(round(value[2], 2)) + " " + str(round(value[3], 2)) 
        print(line)

def sgd(T, eta):
    data = get_dataset()
    n = len(data)
    
    x = [data[i][0] for i in range(n)]
    av_x = sum(x) / n
    
    std = math.sqrt(sum([(x[i] - av_x)**2 for i in range(n)]) / (n - 1))
    
    for i in range(n):
        x[i] = (x[i] - av_x) / std
        
    rand = random.randint(0,n)
    xjt = x[rand]
    yjt = data[rand][1]
    
    beta_0 = 0
    beta_1 = 0
    beta_vector = []
    
    for t in range(T):
        dbeta_0 = 2 * (beta_0 + beta_1 * xjt - yjt)
        dbeta_1 = 2 * (beta_0 + beta_1 * xjt - yjt) * xjt
        beta_0 = beta_0 - eta * dbeta_0
        beta_1 = beta_1 - eta * dbeta_1
        reg = regression_normalized(beta_0, beta_1)
        beta_vector.append((t+1, beta_0, beta_1, reg))
        
    for value in beta_vector:
        line = str(round(value[0], 2)) + " " + str(round(value[1], 2)) + " " + str(round(value[2], 2)) + " " + str(round(value[3], 2)) 
        print(line)