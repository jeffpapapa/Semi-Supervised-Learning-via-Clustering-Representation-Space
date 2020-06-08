# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:38:49 2020

@author: jeffp
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

class DBI(nn.Module):
    def __init__(self, num_of_clus, point_dim, gpu_stat):
        super().__init__()
        
        if(gpu_stat):
            self.clus_count = torch.ones(num_of_clus).cuda()
            self.num_of_clus = torch.tensor(num_of_clus).cuda()
            self.Ai = torch.ones(num_of_clus, point_dim).cuda()*0.001
            self.Si_sum = torch.ones(num_of_clus, point_dim).cuda()*0.001
            self.Si = torch.ones(num_of_clus).cuda()*0.001
            self.Mij = torch.ones(num_of_clus, num_of_clus).cuda()*0.001
            self.Rij = torch.ones(num_of_clus, num_of_clus).cuda()*0.1
            self.Di = torch.ones(num_of_clus).cuda()*0.001
            self.D = torch.tensor(0).cuda()
        else:
            self.clus_count = torch.ones(num_of_clus)
            self.num_of_clus = torch.tensor(num_of_clus)
            self.Ai = torch.ones(num_of_clus, point_dim)*0.001
            self.Si_sum = torch.ones(num_of_clus, point_dim)*0.001
            self.Si = torch.ones(num_of_clus)*0.001
            self.Mij = torch.ones(num_of_clus, num_of_clus)*0.001
            self.Rij = torch.ones(num_of_clus, num_of_clus)*0.1
            self.Di = torch.ones(num_of_clus)*0.001
    
    def forward(self, data_points, clustering):
        
        ### Compute Ai
        for data, num in zip(data_points, clustering):
            self.clus_count[num] = self.clus_count[num] + 1
            self.Ai[num] = self.Ai[num] + data
            
        for count in range(self.num_of_clus):
            if(self.clus_count[count]!=0):
                value_div = self.Ai[count].clone().detach()
                self.Ai[count] = torch.div(value_div, self.clus_count[count])
#         print(data_points)
#         print(self.clus_count)
#         print(self.Ai)
        ### Compute Si
        for data, num in zip(data_points, clustering):
            self.Si[num] = self.Si[num] + torch.sum(torch.pow(data-self.Ai[num],2))
        
        for count in range(self.num_of_clus):
            value_div = self.Si[count].clone().detach()
            self.Si[count] = torch.pow(torch.div(value_div,self.clus_count[count]),0.5)
        #print(self.Si)
        ### Compute Mij
        for i in range(self.num_of_clus):
            for j in range(self.num_of_clus):
                ### upper triangle
                if(i<j):
                    self.Mij[i][j] = torch.norm(self.Ai[i]-self.Ai[j],2)
        
        
        ### Compute Rij
        for i in range(self.num_of_clus):
            for j in range(self.num_of_clus):
                ### upper triangle
                if(i<j):
                    #print(i,j)
                    if(self.Mij[i][j]==0):
                        pass
                    else:
                        value_sum = self.Si[i]+self.Si[j]
                        div_num = self.Mij[i][j].clone().detach()
                        value_div = torch.div(value_sum, div_num)
                        self.Rij[i][j] = value_div
                    self.Rij[j][i] = self.Rij[i][j].clone()
                if(i==j):
                    self.Rij[i][j] = self.Rij[i][j].clone()*0
        #print(self.Rij)
        
        ### Compute Di
#        for num in range(self.num_of_clus):
#            self.Di[num] = torch.max(self.Rij[num])
        
        for num in range(self.num_of_clus):
            value, idx = torch.max(self.Rij[num], 0)
            self.D = self.D + value
        #print(self.Di)
            
        ### Compute DB_loss
        
        #DB_loss = torch.div(torch.sum(self.Di),self.num_of_clus.float())
        DB_loss = torch.div(self.D, self.num_of_clus.float())
        #print(DB_loss)
        
        return DB_loss
        

#class margin(nn.Module):
#    def __init__(self, num_of_clus, k, gpu_stat):
#        super().__init__()
#        self.gpustat = gpu_stat
#        if(gpu_stat):
#            self.k = torch.tensor(k).cuda()
#            self.num_of_clus = torch.tensor(range(0,num_of_clus)).cuda()
#            self.margin_loss = Variable(torch.tensor(0.0).cuda(), requires_grad=True)
#            self.zero_loss = torch.tensor(50000).float().cuda()
#            self.dim = torch.tensor(1.0).cuda()
#            
#        else:
#            self.k = torch.tensor(k)
#            self.num_of_clus = torch.tensor(range(0,num_of_clus))
#            self.margin_loss = torch.tensor(0.0)
#            self.zero_loss = torch.tensor(50000).float()
#            self.dim = torch.tensor(1.0)
#    
#    def pairwise_dist(self, xyz1, xyz2, gpustat):
#        r_xyz1 = torch.sum(xyz1 * xyz1, dim=2, keepdim=True)  # (B,N,1)
#        r_xyz2 = torch.sum(xyz2 * xyz2, dim=2, keepdim=True)  # (B,M,1)
#        mul = torch.matmul(xyz2, xyz1.permute(0,2,1))         # (B,M,N)
#        dist = r_xyz2 - 2 * mul + r_xyz1.permute(0,2,1)       # (B,M,N)
#        try:
#            top_k, indices = torch.topk(dist.flatten(), self.k, largest=False, sorted=False, out=None)
#        except:
#            tmp_k = self.k-1
#            while(tmp_k>0):
#                try:
#                    top_k, indices = torch.topk(dist.flatten(), tmp_k, largest=False, sorted=False, out=None)
#                    break
#                except:
#                    tmp_k = tmp_k-1
##             top_k, indices = torch.topk(dist.flatten(), 1, largest=False, sorted=False, out=None)
#
#
#        # return torch.transpose(top_k[0], 1, 0)
#        
#        #top_k = top_k+1
#        #top_k = torch.pow(top_k,-1)
#    
#        return top_k
#
#    def forward(self, x, clustering):
#        #self.dim = self.dim*len(x[0])
#        devide_rate = -0.5
##         print(x, clustering)
#        for num in (self.num_of_clus):
#
#            points_a = x[(clustering == num.item())]
#            if(len(points_a)==0):
#                value_div = self.margin_loss.clone().detach()
#                self.margin_loss = torch.div(value_div,1)
#            else:
#                points_a = points_a.view(1,len(points_a),-1)
#                for another_num in (self.num_of_clus):
#                    if(another_num != num):
#                        points_b = x[(clustering == another_num)]
#                        if(len(points_b)==0):
#                            value_div = self.margin_loss.clone().detach()
#                            self.margin_loss = torch.div(value_div,1)
#                        else:
#                            points_b = points_b.view(1,len(points_b),-1)
#                            if(another_num>num):
#                                self.margin_loss = (self.margin_loss + 
#                                                    torch.pow(torch.div(torch.sum(self.pairwise_dist(points_a, points_b, self.gpustat)),(self.k.float())),devide_rate))
##                                 self.margin_loss = (self.margin_loss + 
##                                                     torch.div(torch.sum(self.pairwise_dist(points_a, points_b, self.gpustat)),(self.k.float())))
##                                 self.margin_loss = (self.margin_loss + 
##                                                     torch.sum(self.pairwise_dist(points_a, points_b, self.gpustat)))
##        self.margin_loss = torch.pow( self.margin_loss, devide_rate)
#        value_div = self.margin_loss.clone().detach()
#        self.margin_loss = torch.div(value_div,len(self.num_of_clus))
##        self.margin_loss = torch.div(value_div,self.k.float())         
##        self.margin_loss = torch.pow((self.margin_loss),devide_rate)
#        #self.margin_loss = torch.div(value_div,len(x))
#            
#        return self.margin_loss
    
class margin(nn.Module):
    def __init__(self, num_of_clus, k, gpu_stat):
        super().__init__()
        self.gpustat = gpu_stat
        if(gpu_stat):
            self.k = torch.tensor(k).cuda()
            self.num_of_clus = torch.tensor(range(0,num_of_clus)).cuda()
            self.margin_loss = Variable(torch.tensor(0.0).cuda(), requires_grad=True)
            self.zero_loss = torch.tensor(50000).float().cuda()
            self.dim = torch.tensor(1.0).cuda()
            
        else:
            self.k = torch.tensor(k)
            self.num_of_clus = torch.tensor(range(0,num_of_clus))
            self.margin_loss = torch.tensor(0.0)
            self.zero_loss = torch.tensor(50000).float()
            self.dim = torch.tensor(1.0)
    
#     def pairwise_dist(self, xyz1, xyz2, gpustat):
#         r_xyz1 = torch.sum(xyz1 * xyz1, dim=2, keepdim=True)  # (B,N,1)
#         r_xyz2 = torch.sum(xyz2 * xyz2, dim=2, keepdim=True)  # (B,M,1)
#         mul = torch.matmul(xyz2, xyz1.permute(0,2,1))         # (B,M,N)
#         dist = r_xyz2 - 2 * mul + r_xyz1.permute(0,2,1)       # (B,M,N)
#         try:
#             top_k, indices = torch.topk(dist.flatten(), self.k, largest=False, sorted=False, out=None)
#         except:
#             tmp_k = self.k-1
#             while(tmp_k>0):
#                 try:
#                     top_k, indices = torch.topk(dist.flatten(), tmp_k, largest=False, sorted=False, out=None)
#                     break
#                 except:
#                     tmp_k = tmp_k-1
# #             top_k, indices = torch.topk(dist.flatten(), 1, largest=False, sorted=False, out=None)


#         # return torch.transpose(top_k[0], 1, 0)
# #         top_k = top_k+1
# #         top_k = torch.pow(top_k,-1)
#         return top_k

    def pairwise_dist(self, xyz1, xyz2, gpustat):
        top_k_sum = torch.tensor(0.0).cuda()
        r_xyz1 = torch.sum(xyz1 * xyz1, dim=2, keepdim=True)  # (B,N,1)
        r_xyz2 = torch.sum(xyz2 * xyz2, dim=2, keepdim=True)  # (B,M,1)
        mul = torch.matmul(xyz2, xyz1.permute(0,2,1))         # (B,M,N)
        dist = r_xyz2 - 2 * mul + r_xyz1.permute(0,2,1)       # (B,M,N)
        
        dist = dist[0]
        points = len(dist[0])
        for _ in range(self.k):
            if(len(dist.flatten())==0):
                break
            top_k, idx = torch.topk(dist.flatten(), 1, largest=False, sorted=False, out=None)
            dist = torch.cat((dist[:int(idx/(points-_))], dist[int(idx/(points-_))+1:]), 0)
            dist = torch.cat((dist[:,:(idx%(points-_))], dist[:,(idx%(points-_))+1:]),1)
            top_k_sum = top_k_sum + torch.pow(top_k[0],-1)
#             top_k, indices = torch.topk(dist.flatten(), 1, largest=False, sorted=False, out=None)


        # return torch.transpose(top_k[0], 1, 0)
#         top_k = top_k+1
#         top_k = torch.pow(top_k,-1)
        return top_k_sum

    def forward(self, x, clustering):
        #self.dim = self.dim*len(x[0])
        devide_rate = -0.25
#         print(x, clustering)
        for num in (self.num_of_clus):
            points_a = x[(clustering == num)]
            if(len(points_a)==0):
                self.margin_loss = torch.div(self.margin_loss,1)
            else:
                points_a = points_a.view(1,len(points_a),-1)
                for another_num in (self.num_of_clus):
                    if(another_num != num):
                        points_b = x[(clustering == another_num)]
                        if(len(points_b)==0):
                            self.margin_loss = torch.div(self.margin_loss,1)
                        else:
                            points_b = points_b.view(1,len(points_b),-1)
                            if(another_num>num):
                                
#                                 self.margin_loss = (self.margin_loss + 
#                                                     torch.pow(torch.div(torch.sum(self.pairwise_dist(points_a, points_b, self.gpustat)),(self.k.float())),devide_rate))
#                                 self.margin_loss = (self.margin_loss + 
#                                                     torch.pow(torch.div((self.pairwise_dist(points_a, points_b, self.gpustat)),(self.k.float())),devide_rate))

#                                 self.margin_loss = (self.margin_loss + 
#                                                     torch.div(torch.sum(self.pairwise_dist(points_a, points_b, self.gpustat)),(self.k.float())))
#                                 self.margin_loss = (self.margin_loss + 
#                                                     torch.sum(self.pairwise_dist(points_a, points_b, self.gpustat)))
                                self.margin_loss = (self.margin_loss + 
                                                    self.pairwise_dist(points_a, points_b, self.gpustat))

        self.margin_loss = torch.div(self.margin_loss,len(self.num_of_clus))
        self.margin_loss = torch.div((self.margin_loss),self.k.float())         
        #self.margin_loss = torch.pow((self.margin_loss),(-0.25))
        #del self.k,self.num_of_clus
        return self.margin_loss