import copy
import random
import math
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import numpy.linalg as LA
from tqdm import tqdm

            
class FKLICM:
    def __init__(self, error=1e-5):
            super().__init__()
            self.u, self.z, self.n_clusters = None, None, None
            self.max_iter = 100
            self.m = 2
            self.error = error
            
    def fit(self,data):
        #A function must to determine initial cluster
        z = np.concatenate(([[191.60092378594294, 70.57337492039093, 24.899843407682727]], [[234.90884173097305, 160.9799293504796, 128.9331846930514]],
                            [[231.22398401191802, 247.03505240586207, 188.78770596814024]], [[230.6665933403778, 228.32536033677724, 106.95308018971896]], 
                            [[134.66454242382565, 217.4160018156643, 180.52683245243756]], [[250.12562937710385, 179.2448574071007, 235.06592211620836]], 
                            [[160.28617426554362, 139.58720728556315, 48.05828590393067]], [[237.8435262044271, 109.45404052734375, 43.556009928385414]], 
                            [[170.078125, 75.4609375, 105.609375]]),axis=0)
        
        #Number of clusters
        self.n_clusters = z.shape[0]
        
        #Initialize the membership matrix u
        height, width, channels = data.shape
        u = np.zeros((height, width, self.n_clusters))
        

        #Get coordinates and neighbor pixels value
        results = self.get_coordinates(data)
        
        for t in tqdm(range(self.max_iter)):
            #Calculate mt, fuzzy membership and learning rate
            mt = self.m + (t * (self.m - 1))/self.max_iter
            alpha = self.fuzzy_factor(u, z, data, results)
            u = self.update_membership(data, u, z, mt, alpha)
            rate = u ** mt
            
            #Calculate weight
            z_old = z.copy()
            z = self.update_weight(z_old, data, rate)
            
            #Update learning rate
            rate = u ** mt

            if self.end_conditon(z,z_old):
                print ("Finished Clustering")
                break

        self.u = u
        self.z = z
        # Write centers
        with open('centers.txt', 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(self.z.shape))
            for data_slice in self.z:
                np.savetxt(outfile, data_slice, fmt='%-7.4f')
                outfile.write('# New slice\n')

        # Write membership
        with open('u.txt', 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(self.u.shape))
            for data_slice in self.u:
                np.savetxt(outfile, data_slice, fmt='%-7.4f')
                outfile.write('# New slice\n')
    
        output = np.argmax(self.u, axis=-1)
        print("output u argmax ",output.shape)
        
        return output, self.z
    
    def update_membership(self, data, u, z, mt, alpha):
    
        u_new = u.copy()
        
        for i in range(z.shape[0]):
            for kx in tqdm(range(data.shape[0])):
                for ky in tqdm(range(data.shape[1])):
                    current = 0.0
                    norm1 = 0.0
                    norm2 = 0.0
                    for l in range(z.shape[0]):
                        norm1 = (LA.norm(data[kx,ky,:] - z[i,:])) ** 2
                        factor1 = alpha[kx,ky,i]

                        norm2 = (LA.norm(data[kx,ky,:] - z[l,:])) ** 2
                        factor2 = alpha[kx,ky,l]
                        #Analysis division by 0
                        current += ((np.divide((norm1 + factor1),(norm2 + factor2),out=np.zeros_like(norm1 + factor1), where=(norm2 + factor2)!=0)) ** (1/(mt-1)))

                    u_new[kx,ky,i] = np.divide(1,current,out=np.zeros_like(current),where=current!=0)
        
        return u_new
    
    def fuzzy_factor(self, u, center, data, results):

        alpha = np.zeros((u.shape[0], u.shape[1], u.shape[2]))

        for i in range(center.shape[0]):
            for kx in range(data.shape[0]):
                for ky in range(data.shape[1]):
                    temp = 0.0
                    w = (data.shape[1] * kx) + ky
                    for neighbor,v in results[w].items():
                        a = [x for x,y in [neighbor]]
                        b = [y for x,y in [neighbor]]
                        a = a[0]
                        b = b[0]
                    
                        norm = (LA.norm(center[i] - v)) ** 2
                        dist = LA.norm(data[kx,ky,:] - v)
                        temp += ((((1 - u[a,b,i]) ** self.m) * norm) / (dist + 1))
                    
                    alpha[kx,ky,i] = temp

        return alpha
    
    def get_coordinates(self, data):
   
        d = {(i, c):data[i][c] for i in range(len(data)) for c in range(len(data[0]))}
   
        results = []     
        for kx in tqdm(range(data.shape[0])):
            for ky in tqdm(range(data.shape[1])):
                results.append({k:v for k,v in d.items() if k in [(kx, ky-1), (kx+1, ky-1), (kx+1, ky),(kx+1, ky+1), (kx, ky+1), (kx-1, ky+1), (kx-1, ky), (kx-1,ky-1)]})

        results = np.array(results)

        return results

    def update_weight(self, z_old, data, rate):

        z = []
        for i in range(len(z_old)):
            current1 = 0.0
            current2 = 0.0
            temp = 0.0
            for kx in range(data.shape[0]):
                for ky in range(data.shape[1]):
                    current1 += rate[kx,ky,i] * (data[kx,ky,:] - z_old[i,:])
                    current2 += rate[kx,ky,i]
            temp = z_old[i,:] + current1/current2
            z.append(temp)

        z = np.array(z)
        
        return z

    def end_conditon(self, z, z_old):
        
        if (LA.norm(z - z_old)) > self.error:
            return False

        return True
