import numpy as np
import random

class dataset:
    def __init__(self,no):
        self.number=no
        self.data0=[]
        self.data1=[]

    def get(self,add_noise=False):
        n=self.number
        self.data0=[]
        self.data1=[]
        for i in range(n):
            if add_noise==False:
                if random.randint(0,1)==0:
                    x=random.uniform(-1,1)
                    sign=random.choice([-1,1])
                    y=sign*np.sqrt(1-x**2)
                    self.data0.append(np.array([x,y]))
                else:
                    x=random.uniform(-1,1)
                    sign=random.choice([-1,1])
                    y=3+sign*np.sqrt(1-x**2)
                    self.data1.append(np.array([x,y]))
            else:
                if random.randint(0,1)==0:
                    x=random.uniform(-1,1)
                    sign=random.choice([-1,1])
                    y=sign*np.sqrt(1-x**2)
                    self.data0.append(np.array([x+random.gauss(0,0.1),y+random.gauss(0,0.1)]))
                else:
                    x=random.uniform(-1,1)
                    sign=random.choice([-1,1])
                    y=3+sign*np.sqrt(1-x**2)
                    self.data1.append(np.array([x+random.gauss(0,0.1),y+random.gauss(0,0.1)]))
        self.data0=np.array(self.data0)
        self.data1=np.array(self.data1)

class perceptron_Training_Algorithm_customed:
    def __init__(self):
        self.weight=[]
        self.bias=0
    
    def training(self,data0,data1,bias):
        self.bias=0
        self.weight=np.zeros(2)
        converged=True
        if bias:
            for i in range(10000):
                converged=True
                for j in range(len(data1)):
                    if np.dot(self.weight,data1[j])+self.bias<=0:
                        self.weight+=data1[j]
                        self.bias+=1
                        converged=False
                for j in range(len(data0)):
                    if np.dot(self.weight,data0[j])+self.bias>=0:
                        self.weight-=data0[j]
                        self.bias-=1
                        converged=False
                if converged==True:
                    break
        else:
            for i in range(10000):
                converged=True
                for j in range(len(data1)):
                    if np.dot(self.weight,data1[j])<=0:
                        self.weight+=data1[j]
                        converged=False
                for j in range(len(data0)):
                    if np.dot(self.weight,data0[j])>=0:
                        self.weight-=data0[j]
                        converged=False
                if converged==True:
                    break
        return self.weight,self.bias
    
    def training_dataset(self,data,number,bias):
        self.bias=0
        self.weight=np.zeros(2)
        converged=True
        if bias:
            for i in range(10000):
                converged=True
                for j in range(len(data)):
                    label=number[j]
                    if label>0:
                        if np.dot(self.weight,data[j])+self.bias<=0:
                            self.weight+=data[j]
                            self.bias+=1
                            converged=False
                    else:
                        if np.dot(self.weight,data[j])+self.bias>=0:
                            self.weight-=data[j]
                            self.bias-=1
                            converged=False
                if converged==True:
                    break
        else:
            for i in range(10000):
                converged=True
                for j in range(len(data)):
                    label=number[j]
                    if label>0:
                        if np.dot(self.weight,data[j])<=0:
                            self.weight+=data[j]
                            converged=False
                    else:
                        if np.dot(self.weight,data[j])>=0:
                            self.weight-=data[j]
                            converged=False
                if converged==True:
                    break
        return self.weight,self.bias