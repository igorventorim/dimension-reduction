#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:27:55 2017

@author: thomas
"""
import numpy as np
import matplotlib.pyplot as plt


from LabelBinarizer2 import LabelBinarizer2
lb = LabelBinarizer2()


#from keras.utils import to_categorical

def printall(obj):
    np.set_printoptions(threshold='inf') # print all elements of array
    print(obj)
    np.set_printoptions(threshold=1000) # back to default

def halfvec(v):
    return v[0:len(v)/2]

def code1outofc(y):
    #print 'code1outofc>\n'
    
    # convert the string labels to numerical class labels
    c = np.unique(y, return_inverse=True)
    labels = c[0]
    numclasses = len(labels)
    y = c[1]
    # code the numerical class labels as multidimensional 1-out-of-c vector
    #YY = to_categorical(y, num_classes=numclasses)
    
    lb.fit(y)
    Y = lb.transform(y)
    #assert((Y==YY).all()); printall(Y); printall(YY)

    return y, Y, numclasses

def pause(comment=''):
    raw_input(comment + 'Press the <ENTER> key to continue...')

def customxlabels(xmin, xmax, numticks, scale=1.0, formstr='%.2f'):
    x = np.linspace(xmin, xmax, numticks+1)
    delta = (xmax-xmin)/numticks
    ticks = []
    
    for i in xrange(0,numticks+1):
        #print 'i=', i, 'of', numticks+1, 'tick=', i*delta*scale
        tick = formstr % (i*delta*scale)
        #tick = str(i*delta*scale)
        ticks.append(tick)
    #print 'x=', x, 'customxlabels>', ticks
    plt.xticks(x, ticks, rotation=45)


def axisscale(start=0, stop=1, num=None, delta=None):
    ''' returns a scaled x-axis'''
    print('num=', num, 'delta=', delta)
    if num == None:
        if delta == None:
            print('axisscale> Error: at least num of delta must be defined!')
            return None
        else:
            num = int(round((stop-start+1)/delta))
            print('num=', num, 'delta=', delta)
            x = np.linspace(start=start, stop=stop, num=num)
    else:
        if delta != None:
            print('axisscale> Error: num and delta defined simultaneously!')
            return None
        else:
            x = np.linspace(start=start, stop=stop, num=num)
    print(x)
    return x


if __name__ == '__main__':
    '''
    y = ['setosa', 'setosa', 'virginica', 'virginica', 'setosa', 'vericolor' ]
    print y
    y, Y, numclasses = code1outofc( y )
    print y
    print Y
    print numclasses
    pause('Test: ')
    '''
    axisscale(start=0, stop=3, num=4)
    axisscale(start=0, stop=3, delta=1)
    axisscale(start=-2, stop=3, delta=1)
    axisscale(start=0, stop=3.56, delta=1)
    axisscale(start=0, stop=3.56, delta=1.3)
    axisscale(start=0.2, stop=3.56, delta=1.3)    