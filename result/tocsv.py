import os
import shutil
import subprocess
import pandas as pd

accesses = []
temp=[]
power=[]
number=[]
prediction=[]
a=0
with open('data.trace', 'r') as f:
    lines=f.readlines()
    for i in range(len(lines)-1):
        l=lines[i].split("    ")
        l2=lines[i+1].split("    ")
        for j in range(1,33):
            current=l[j].split(" ")
            next=l2[j].split(" ")
            accesses.append(current[0])
            temp.append(current[1])
            power.append(current[2])
            number.append(j-1)
            prediction.append(next[1])
            a=a+1
        print(a)
with open('data2.trace', 'r') as f:
    lines=f.readlines()
    for i in range(len(lines)-1):
        l=lines[i].split("    ")
        l2=lines[i+1].split("    ")
        for j in range(1,33):
            current=l[j].split(" ")
            next=l2[j].split(" ")
            accesses.append(current[0])
            temp.append(current[1])
            power.append(current[2])
            number.append(j-1)
            prediction.append(next[1])
            a=a+1
        print(a)
with open('data3.trace', 'r') as f:
    lines=f.readlines()
    for i in range(len(lines)-1):
        l=lines[i].split("    ")
        l2=lines[i+1].split("    ")
        for j in range(1,33):
            current=l[j].split(" ")
            next=l2[j].split(" ")
            accesses.append(current[0])
            temp.append(current[1])
            power.append(current[2])
            number.append(j-1)
            prediction.append(next[1])
            a=a+1
        print(a)
df=pd.DataFrame({'accesses':accesses,'temp':temp,'power':power,'number':number,'prediction':prediction})
df.to_csv('result.csv',index=False)