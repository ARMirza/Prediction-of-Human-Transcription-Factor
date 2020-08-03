import os
import csv
import numpy as np
import pandas as pd
dataframelist=[]
fasta=""
files=os.listdir('data')
aminoacid=list(np.zeros(26,dtype = int))
columns=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','Length','Factor'] 
finallist=[]
k = 0 

for i in range(0,len(files)):
    try:
        aminoacid=list(np.zeros(26,dtype = int))
        fasta=""
        data=""
        count=0
        activator = False
        repressor = False
        with open ('data/'+files[i], "r") as myfile:
            data=myfile.readlines()
        fasta="" 
        for j in range(0,len(data)):
            if j==0:
                temp=data[j]
                temp=temp[0:len(temp)-1]
                if str(temp)=="Activator":
                    k+=1
                    if(k<100):
                        activator = True
                    else:
                        activator = False
                if str(temp)=="Repressor":
                    repressor = True
                    activator  =  False
                if str(temp)=="No transcription":
                    activator = False
                    repressor = False
                    break
                
            elif j>=4:   
                fasta+=data[j]
        aminoacid.append(len(fasta))
        aminoacid.append(data[0])
        if(activator):
            for q in fasta:
                if(ord(q)!=10):
                    aminoacid[ord(q)-65]+=1
            dataframelist.append(aminoacid)    
        elif(repressor):
            for q in fasta:
                if(ord(q)!=10):
                    aminoacid[ord(q)-65]+=1
            dataframelist.append(aminoacid)
    except Exception as e:
        continue
    
df=pd.DataFrame(dataframelist,columns=columns)
df.to_csv('finaldata.csv')