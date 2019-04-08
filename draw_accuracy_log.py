#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import pylab
from pylab import figure, show, legend
from mpl_toolkits.axes_grid1 import host_subplot

# read the log file
fp_noshedule = open('log/train_noshedule.log', 'r')
fp_10000=open('log/train_schedule10000.log','r')
fp_20000=open('log/train_schedule20000.log','r')
fp_5000=open('log/train_schedule5000.log','r')


train_iterations = []
train_acc = []
test_iterations = []

val_iterations=[]
val_acc=[]
val_iterations10000=[]
val_acc10000=[]
val_iterations20000=[]
val_acc20000=[]
val_iterations5000=[]
val_acc5000=[]

# test_accuracy = []

for ln in fp_noshedule:
    # get train_iterations and train_loss
    if 'Train-accuracy=' in ln :
        arr = re.findall(r'\[\b\d+\b\]', ln)
        arr_trainacc=re.findall(r'Train-accuracy=\b\d+\b',ln)
        train_iterations.append(int(arr[0].strip('[').strip(']')))
        train_acc.append(float(ln.split('=')[-1]))

    if 'Validation-accuracy=' in ln:
        arr = re.findall(r'\[\b\d+\b\]', ln)
        arr_valacc=re.findall(r'Validation-accuracy=\b\d+\b',ln)
        val_iterations.append(int(arr[0].strip('[').strip(']')))
        val_acc.append(float(ln.split('=')[-1]))

for ln in fp_10000:
    if 'Validation-accuracy=' in ln:
        arr=re.findall(r'\[\b\d+\b\]',ln)
        arr_valacc=re.findall(r'Validation-accuracy=\b\d+\b',ln)
        val_iterations10000.append(int(arr[0].strip('[').strip(']')))
        val_acc10000.append(float(ln.split('=')[-1]))

for ln in fp_20000:
    if 'Validation-accuracy=' in ln:
        arr=re.findall(r'\[\b\d+\b\]',ln)
        arr_valacc=re.findall(r'Validation-accuracy=\b\d+\b',ln)
        val_iterations20000.append(int(arr[0].strip('[').strip(']')))
        val_acc20000.append(float(ln.split('=')[-1]))

for ln in fp_5000:
    if 'Validation-accuracy=' in ln:
        arr=re.findall(r'\[\b\d+\b\]',ln)
        arr_valacc=re.findall(r'Validation-accuracy=\b\d+\b',ln)
        val_iterations5000.append(int(arr[0].strip('[').strip(']')))
        val_acc5000.append(float(ln.split('=')[-1]))



fp_noshedule.close()
plt.title('schedule_test')
plt.plot(val_iterations10000,val_acc10000,color='green',label='schedule 10000')
plt.plot(val_iterations20000,val_acc20000,color='red',label='schedule 20000')
plt.plot(val_iterations,val_acc,color='blue',label='no schedule')
plt.plot(val_iterations5000,val_acc5000,color='pink',label='schedule 5000')
plt.legend()

plt.xlabel('epoch times')
plt.ylabel('acc')
plt.savefig('schedule_test.png')
plt.show()