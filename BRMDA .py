# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
from math import exp 
from openpyxl import Workbook

nm = 495 
nd = 383 
nc = 5430 

disease_number = np.genfromtxt(r'disease number.txt',dtype=str,delimiter='\t') 
miRNA_number=np.loadtxt(r'miRNA number.txt',dtype=bytes).astype(str)

DS1 =np.loadtxt(r'disease semantic similarity matrix 1.txt') 
DS2 =np.loadtxt(r'disease semantic similarity matrix 2.txt')
DS = (DS1 + DS2) / 2  
DSweight = np.loadtxt(r'weight matrix of disease semantic similarity.txt') 

FS = np.loadtxt(r'miRNA functional similarity matrix.txt')
FSweight = np.loadtxt(r'weight matrix of miRNA functional similarity.txt')


A = np.zeros((nd,nm),dtype=float) 
ConnectDate = np.loadtxt(r'known disease mirna association.txt',dtype=int)-1 
for i in range(nc):    
    A[ConnectDate[i,1], ConnectDate[i,0]] = 1 
def Getgauss_miRNA(adjacentmatrix,nm):
      

       KM = np.zeros((nm,nm))

       gamaa=1
       sumnormm=0
       for i in range(nm):
           normm = np.linalg.norm(adjacentmatrix[:,i])**2
           sumnormm = sumnormm + normm  
       gamam = gamaa/(sumnormm/nm)


       for i in range(nm):
              for j in range(nm):
                      KM[i,j]= exp (-gamam*(np.linalg.norm(adjacentmatrix[:,i]-adjacentmatrix[:,j])**2))
       return KM
       
def Getgauss_disease(adjacentmatrix,nd):


       KD = np.zeros((nd,nd))

       gamaa=1
       sumnormd=0
       for i in range(nd):
              normd = np.linalg.norm(adjacentmatrix[i])**2
              sumnormd = sumnormd + normd
       gamad=gamaa/(sumnormd/nd)


       for i in range(nd):
           for j in range(nd):
               KD[i,j]= exp(-(gamad*(np.linalg.norm(adjacentmatrix[i]-adjacentmatrix[j])**2)))
       return KD
   
def get_nearest_neighbors(S, size=5):     
    S -= np.eye(S.shape[0])  
    m, n = S.shape
    X = np.zeros((m, n))
    for i in range(m):
        ii = np.argsort(-S[i, :])[:min(size, n)]
         #   ii = ii[0:size]           
        X[i, ii] = S[i, ii]
    return X   

class BRDTI(object):
    
    def __init__(self,args):                
        self.D = args["D"]
        self.learning_rate = args["learning_rate"]       
        self.max_iters = args["max_iters"]
        self.intMat = args["intMat"]
        self.lambda_R = args["lambda_R"]
        self.lambda_C = args["lambda_C"]
        self.FS_m = args["FS_m"]
        self.DS_d = args["DS_d"]
        self.simple_predict = args["simple_predict"]
               
    def init(self):
       
        self.miRNA_bias = np.zeros(nm)
       
       
        self.miRNA_factors = np.sqrt(1/float(self.D)) * np.random.normal(size=(nm,self.D))
        self.disease_factors = np.sqrt(1/float(self.D)) * np.random.normal(size=(nd,self.D))

    def _uniform_user_sampling(self):
        """
          Creates `n_samples` random samples from training data for performing Stochastic
          Gradient Descent. We start by uniformly sampling miRNAs, 
          and then sample a positive and a negative disease for each 
          miRNA sample.
        """  
        self.pos_pairs = np.argwhere(self.intMat==1)

        data = sp.csr_matrix(self.intMat)
        np.random.shuffle(self.pos_pairs)
        sgd_diseases = list(self.pos_pairs[:,0])
        sgd_pos_miRNAs = list(self.pos_pairs[:,1])

        sgd_neg_miRNAs = []
        for sgd_disease in sgd_diseases:
            neg_miRNA = np.random.choice(list(set(range(495)) - set(data[sgd_disease].indices)))
            sgd_neg_miRNAs.append(neg_miRNA)
            
        return sgd_diseases, sgd_pos_miRNAs, sgd_neg_miRNAs
        
       
     
        
       
    
    def update_factors(self,u,i,j):
        """apply SGD update"""        
        
        x = self.miRNA_bias[i] - self.miRNA_bias[j] + np.dot(self.disease_factors[u,:],self.miRNA_factors[i,:]-self.miRNA_factors[j,:])
        
        if x > 200:
            z = 0
        if x < -200:
            z = 1
        else:    
            ex = exp(-x)
            z = ex/(1.0 + ex)
        update_u=True
        update_i=True
        update_j = True
        # update bias terms
        if update_i:
            d = z - self.lambda_R * self.miRNA_bias[i]
            self.miRNA_bias[i] = self.miRNA_bias[i]+ self.learning_rate * d
        if update_j:
            d = -z - self.lambda_R * self.miRNA_bias[j]
            self.miRNA_bias[j] = self.miRNA_bias[j]+ self.learning_rate * d  
           
        if update_u:                         
            d = (self.miRNA_factors[i,:]-self.miRNA_factors[j,:])*z - self.lambda_R*self.disease_factors[u,:] 
           # if self.user_cb_alignment_regularization > 0:
                #code for updating content alingment - based on similarity matrix                          
            alignmentVectorU = np.dot(self.DS_d[u,:], self.disease_factors)   
            alignmentSumU = np.sum(self.DS_d[u,:])
            d = d - 2*self.lambda_R *self.lambda_C* ((alignmentSumU * self.disease_factors[u,:])-alignmentVectorU )
            self.disease_factors[u,:] += self.learning_rate * np.asarray(d)

        if update_i:                                    
            d = self.disease_factors[u,:]*z - self.lambda_R*self.miRNA_factors[i,:] 
         #   if self.item_cb_alignment_regularization > 0:
                #code for updating content alingment - based on similarity matrix               
            alignmentVectorI = np.dot(self.FS_m[i,:], self.miRNA_factors)   
            alignmentSumI = np.sum(self.FS_m[i,:])
            d = d - 2*self.lambda_R *self.lambda_C* ((alignmentSumI * self.miRNA_factors[i,:])-alignmentVectorI )
            self.miRNA_factors[i,:] += self.learning_rate * np.asarray(d)

        if update_j:                         
            d = -self.disease_factors[u,:]*z - self.lambda_R *self.miRNA_factors[j,:]
           # if self.user_cb_alignment_regularization > 0:
                #code for updating content alingment - based on similarity matrix               
            alignmentVectorJ = np.dot(self.FS_m[j,:], self.miRNA_factors)   
            alignmentSumJ = np.sum(self.FS_m[j,:])
            d = d - 2*self.lambda_R *self.lambda_C* ((alignmentSumJ * self.miRNA_factors[j,:])-alignmentVectorJ )            
            self.miRNA_factors[j,:] += self.learning_rate * np.asarray(d)
 
    def train(self):
        """train model
        data: miRNA-disease matrix as a scipy sparse matrix
              miRNAs and diseases are zero-indexed
        mSim: matrix of miRNA similarities
        dSim: matrix of disease similarities
        """
        self.init()   
    
        for it in range(self.max_iters):
            diseases, pos_miRNAs, neg_miRNAs = self._uniform_user_sampling()
            for u,i,j in zip(diseases, pos_miRNAs, neg_miRNAs):
                self.update_factors(u,i,j)

    

    def predict(self,u,i):
                               
            #predict only from learned factors, ignore neighbouring diseases and miRNAs even for novel ones
            if self.simple_predict:
                return self.miRNA_bias[i]  + np.dot(self.disease_factors[u],self.miRNA_factors[i])
    
            elif (u not in self.pos_pairs[:,0]) & (i in self.pos_pairs[:,1]):            
                alignmentMatrixU = np.dot(self.DS_d[u,:], self.disease_factors)              
                alignmentVectorU = alignmentMatrixU/np.sum(self.DS_d[u,:]) 
                return self.miRNA_bias[i]  + np.sum(np.array(alignmentVectorU[:]).flatten() * np.array(self.miRNA_factors[i,:]).flatten()) 
    
            elif (i not in self.pos_pairs[:,1]) & (u in self.pos_pairs[:,0]):            
                alignmentMatrixI = np.dot(self.FS_m[i,:], self.miRNA_factors)           
                alignmentVectorI = alignmentMatrixI/np.sum(self.FS_m[i,:])              
                return np.mean(self.miRNA_bias)  +  np.sum(np.array(alignmentVectorI[:]).flatten() * np.array(self.disease_factors[u,:]).flatten())
            
            elif (i not in self.pos_pairs[:,1]) & (u not in self.pos_pairs[:,0]):
                alignmentMatrixI = np.dot(self.FS_m[i,:], self.miRNA_factors)    
                alignmentVectorI = alignmentMatrixI/np.sum(self.FS_m[i,:])            
                alignmentMatrixU = np.dot(self.DS_d[u,:], self.disease_factors)    
                alignmentVectorU = alignmentMatrixU/np.sum(self.DS_d[u,:]) 
                
                return np.mean(self.miRNA_bias)  + np.sum(np.array(alignmentVectorU[:]).flatten() * np.array(alignmentVectorI[:]).flatten())
            
            else:
                return self.miRNA_bias[i]  + np.dot(self.disease_factors[u],self.miRNA_factors[i])
        
    def predict_score(self):
        score = []
        self.neg_pairs = np.argwhere(self.intMat==0)
        for u,i in zip (self.neg_pairs[:,0],self.neg_pairs[:,1]):
            score.append([disease_number[u][1],miRNA_number[i][1],self.predict(u,i)])
        return score
            

KM = Getgauss_miRNA(A,nm)  
KD = Getgauss_disease(A,nd)  

  
FS_integration=np.zeros((nm,nm))
for i in range(nm):
    for j in range(nm):
        if  FSweight[i,j] == 1:
            FS_integration[i,j] = FS[i,j]
        else:
            FS_integration[i,j] = KM[i,j]
  

DS_integration = np.zeros((nd,nd))
for i in range(nd):
    for j in range(nd):
        if  DSweight[i,j] == 1:
            DS_integration[i,j] = DS[i,j]
        else:
            DS_integration[i,j] = KD[i,j]


FS_integration = get_nearest_neighbors(FS_integration, 5)
DS_integration = get_nearest_neighbors(DS_integration, 5)  
   


args = {
            'D':50,
            'learning_rate':0.05,
            'max_iters' : 100,   
            'lambda_R':10**(-2),  
            'lambda_C' :1, 
            'FS_m': FS_integration,
            'DS_d': DS_integration,
            'intMat': A,
            'simple_predict': False}
    
brdti_model= BRDTI(args)
brdti_model.train()
globalscore = brdti_model.predict_score()

wb1=Workbook()
ws1=wb1.active
ws1.title = 'prediction result'

ws1.append(['disease name','miRNA name','score'])

for i in globalscore:

    ws1.append(i)
wb1.save(filename=r'predcition_result.xlsx')












