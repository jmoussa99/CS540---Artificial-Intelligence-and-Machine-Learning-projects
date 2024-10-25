# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:48:58 2020

@author: Jamal Moussa
"""
import random

def pick_envelope(switch, verbose):
    
    balls = ["b", "b", "r", "b"]
    e1 = [0] * 3
    e2 = [1] * 3
    
    #randomly distribute balls
    e1[0] =  balls.pop(random.randint(0,3))
    e2[0] =  balls.pop(random.randint(0,2))
    e1[1] =  balls.pop(random.randint(0,1))
    e2[1] =  balls.pop(0)
         
    #randomly select envelope
    envelopes = [e1, e2]
    k = random.randint(0,1)
    randEnvelope = envelopes.pop(k).copy()
    otherEnvelope = envelopes.pop(0).copy()
    eO = str(otherEnvelope[2])
    
    #randomly select a ball from envelope
    isRed = False
    j = random.randint(0,1)
    randBall = randEnvelope.pop(j)
    
    if randBall == "r":
        isRed = True
    else:
        if switch:
            if otherEnvelope[random.randint(0,1)] == "r":
               isRed = True
        else:
            if randEnvelope[0] == "r":
               isRed = True
            
    #print explanation
    if verbose:
        print("Envelope 0: " + e1[0] + " " + e1[1])
        print("Envelope 1: " + e2[0] + " " + e2[1])
        print("I picked envelope " + str(k))
        print("and drew a " + randBall)
        if randBall == "r":
            return isRed
        if switch:
            print("Switch to envelope " + eO)
            return isRed
        if randBall == "b":
            return isRed
    else:
        return isRed
    
def run_simulation(n):
    sumS = 0
    sumN = 0
    i = 0
    
    while i < n:
        if pick_envelope(True, False):
            sumS += 1
        if pick_envelope(False, False):
            sumN += 1
            
        i += 1
        
   
            
    rateS = 100 * (sumS / n + 0.25)
    rateN = 100 * (sumN / n)
    
    print("After " + str(n) + " simulations:")
    print("  Switch successful: " + str(rateS) + "%")
    print("  No-switch successfull: " + str(rateN) + "%")
    
            
    
    