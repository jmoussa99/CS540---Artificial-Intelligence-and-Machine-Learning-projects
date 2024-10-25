# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:56:51 2020

@author: Jamal Moussa
"""
from collections import Counter
import os
import math

def create_vocabulary(training_directory, cutoff):
    
    vocab= Counter()
    for _ in ['2016/', '2020/']:
        path = training_directory + _
        temp = os.listdir(path)
        for i, file in enumerate(temp):
            file = path + file
            with open(file, 'r', encoding='utf8') as line:
                word = Counter(line.read().split('\n'))
                word = {key: word[key] for key in word if word[key] >= cutoff}
                vocab += word
        
    return sorted(vocab.keys())

def create_bow(vocab, filepath):
    
    bow = Counter()
    
    with open(filepath, 'r', encoding= 'utf8') as line:
        words = Counter(line.read().split('\n'))
        for i in words:
            if vocab and str(i) in vocab:
                bow += {i: words[i]}
            else:
                bow += {None: 1}
    
    return {key: bow[key] for key in bow}
        
def load_training_data(vocab, directory):
    load = []
    
    for label in ['2016', '2020']:
        path = directory + label + '/'
        temp = os.listdir(path) #list of all files in year
        
        for i, file in enumerate(temp):
            file = path + file
            entry = {'label': label, 'bow':create_bow(vocab, file)}
            load.append(entry)
    
    return load
    
def prior(training_data, label_list):
    sum2016 = 0
    sum2020 = 0
    total = 0
    
    for docs in training_data:
        if label_list[0] == docs.get('label'):
            sum2020 += 1
            total += 1
        elif label_list[1] == docs.get('label'):
            sum2016 += 1
            total += 1
            
    p2016 = math.log(sum2016) - math.log(total)
    p2020 = math.log(sum2020) - math.log(total)
    
    return {'2020': p2020, '2016': p2016}
            
def p_word_given_label(vocab, training_data, label):
    
    wordcnt = Counter(vocab) + Counter({None: 1}) #counter for all words w/ -1 smoothing
    sumw = 0
    
    for docs in training_data:
        if label == docs.get('label'):
            words = docs.get('bow')
            for key in words: # words type dict, key = word w/count per doc
                if key in wordcnt:
                    wordcnt.update({key: words[key]})
                    sumw += wordcnt[key] 
                else:
                    wordcnt.update({None: words[key]})
                    sumw += wordcnt[key] 
    
    for w in wordcnt:
        wordcnt[w] = math.log(wordcnt[w]) - math.log(sumw + len(vocab))
        
    return {key: wordcnt[key] for key in wordcnt}

def train(training_directory, cutoff):
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)
    
    priorlog = prior(training_data, ['2020', '2016'])
    pw2016 = p_word_given_label(vocab, training_data, '2016')
    pw2020 = p_word_given_label(vocab, training_data, '2020')
    
    trained = {
            'vocabulary': vocab,
            'log prior': priorlog,
            'log p(w|y=2016)': pw2016,
            'log p(w|y=2020)': pw2020
            }
    
    return trained

def classify(model, filepath):
    data = create_bow(model['vocabulary'], filepath)
    classify2016 = 0
    classify2020 = 0
    
    for word in data:
        classify2016 += model['log p(w|y=2016)'][word]
        classify2020 += model['log p(w|y=2020)'][word]
        
    classify2016 += model['log prior']['2016']
    classify2020 += model['log prior']['2020']
    
    predict = ''
    if classify2020 > classify2016:
        predict = '2020'
    else:
        predict = '2016'
        
    classifier = {
            'predicted y': predict,
            'log p(y=2016|x)': classify2016,
            'log p(y=2020|x)': classify2020
            }
    
    return classifier

    