import os
import shutil
import subprocess
import pandas as pd

find_command = 'find result -type f -name "final.trace"'
find_process = subprocess.Popen(find_command, shell=True, stdout=subprocess.PIPE, universal_newlines=True)

accesses = []
temp=[]
power=[]
number=[]
prediction=[]

for file in find_process.stdout:
    file=file.strip()
    with open(file, 'r') as f:
        lines=f.readlines()
        for i in range(len(lines)-1):
            l=lines[i].split("    ")
            l2=lines[i+1].split("    ")
            for j in range(2,34):
                current=l[j].split(" ")
                next=l2[j].split(" ")
                accesses.append(current[0])
                temp.append(current[1])
                power.append(current[2])
                number.append(j-2)
                prediction.append(next[1])
df=pd.DataFrame({'accesses':accesses,'temp':temp,'power':power,'number':number,'prediction':prediction})
df.to_csv('result.csv',index=False)