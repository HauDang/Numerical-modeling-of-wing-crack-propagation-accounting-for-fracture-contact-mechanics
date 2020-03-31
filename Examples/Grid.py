import numpy as np 
import math
import auxiliary
aux = auxiliary.AuxiFu()
class ReMesh():
    def __init__(self, p, t, tipcra, moucra, roscra, nodcra, nodaro, dc):
        self.p = p
        self.t = t
        self.tipcra = tipcra
        self.moucra = moucra
        self.roscra = roscra
        self.nodcra = nodcra
        self.nodaro = nodaro
        self.dc = dc
    def refinement(self, tipava, crapro, error, A0, lmin, option):
        """ Mesh refinement around crack tips"""
        numtip = len(self.tipcra)
        eleref1 = [] # find elements at tips
        for i in range(numtip):
            if tipava[i] == True and crapro[i] == True:
                tip = self.tipcra[i]
                tipind = aux.p2index(self.p,tip,0)
                eleclo = np.where(self.t == tipind)[0]
                eleref1 = np.concatenate((eleref1,eleclo))
        if len(eleref1) > 0:
            eleref1 = eleref1.astype(int)
            xcoord = np.zeros((len(eleref1),3))
            xcoord[:,0] = self.p[self.t[eleref1,0],0]
            xcoord[:,1] = self.p[self.t[eleref1,1],0]
            xcoord[:,2] = self.p[self.t[eleref1,2],0]
            ycoord = np.zeros((len(eleref1),3))
            ycoord[:,0] = self.p[self.t[eleref1,0],1]
            ycoord[:,1] = self.p[self.t[eleref1,1],1]
            ycoord[:,2] = self.p[self.t[eleref1,2],1]
            areele1 = aux.area(xcoord,ycoord)
            eleref1 = eleref1[np.where(areele1 > 4*A0)[0]]
        
        eleref2 = [] # find elements around rossete
        xc = (self.p[self.t[:,0],:] + self.p[self.t[:,1],:] + self.p[self.t[:,2],:])/3
        for i in range(len(crapro)):
            if tipava[i] == True and crapro[i] == True:
                tip = self.tipcra[i]
                dt = np.sqrt((xc[:,0] - tip[0])**2 + (xc[:,1] - tip[1])**2)
                eleclo = np.where(dt < lmin*4)[0]
                eleref2 = np.concatenate((eleref2,eleclo))
            
        if len(eleref2) > 0:
            eleref2 = eleref2.astype(int)    
            xcoord = np.zeros((len(eleref2),3))
            xcoord[:,0] = self.p[self.t[eleref2,0],0]
            xcoord[:,1] = self.p[self.t[eleref2,1],0]
            xcoord[:,2] = self.p[self.t[eleref2,2],0]
            ycoord = np.zeros((len(eleref2),3))
            ycoord[:,0] = self.p[self.t[eleref2,0],1]
            ycoord[:,1] = self.p[self.t[eleref2,1],1]
            ycoord[:,2] = self.p[self.t[eleref2,2],1]
            areele2 = aux.area(xcoord,ycoord)
            eleref2 = eleref2[np.where(areele2 > 6*A0)[0]]     
        if option == 'based on tip':
            eleref = np.int32(np.concatenate((eleref1,eleref2)))
        elif option == 'based on error':
            eleref0 = [] # find highest error elements
            if sum(error) != 0:
                ind = error.ravel().argsort()
                eleref0 = ind[len(error)-numtip*6::]
                xcoord = np.zeros((len(eleref0),3))
                xcoord[:,0] = self.p[self.t[eleref0,0],0]
                xcoord[:,1] = self.p[self.t[eleref0,1],0]
                xcoord[:,2] = self.p[self.t[eleref0,2],0]
                ycoord = np.zeros((len(eleref0),3))
                ycoord[:,0] = self.p[self.t[eleref0,0],1]
                ycoord[:,1] = self.p[self.t[eleref0,1],1]
                ycoord[:,2] = self.p[self.t[eleref0,2],1]
                areele0 = aux.area(xcoord,ycoord)
                eleref0 = eleref0[np.where(areele0 > 8*A0)[0]]                
            eleref = np.int32(np.concatenate((eleref0,eleref1,eleref2)))
        if len(eleref) > 0:
            eleref = np.unique(eleref)
            print('remeshing', len(eleref), 'elements')
            self.divideelement(eleref)
            nodfix = aux.p2index(self.p,self.nodaro,0)
            for i in range(numtip):
                tip = self.tipcra[i]
                fixtip = []
                if len(tip) > 0:
                    tipindi = aux.p2index(self.p,tip,0)  
                    fixtip = np.unique(self.t[np.where(self.t == tipindi)[0],:])
                craindi = aux.p2index(self.p,self.nodcra[i],0)
                nodfix = np.concatenate((nodfix,craindi,fixtip))
            self.smoothing(nodfix)
        return self.p, self.t

    def quaterelement(self, tipava, crapro, lmin):
        """ Replace region around all crack tips by new mesh including the quater point elements """
        numtip = len(self.tipcra)
        for i in range(numtip):
            if tipava[i] == True and crapro[i] == True: 
                tip = self.tipcra[i]
                pvcr = self.nodcra[i]
                self.p,self.t,pb,pt,paround,indexqpe = self.tipmesh(tip,pvcr,lmin)
                self.moucra[i] = np.concatenate(([pb.reshape(1,2), pt.reshape(1,2)]))
                self.roscra[i] = paround 
        self.removeduplicateelement()
        return self.p, self.t, self.moucra, self.roscra
    def adjustmesh(self, tipava, KIC, Gi, keq, lmin, iniang, gloang, craang ):
        """ Update the fracture propagation by determination new crack tip in the mesh
        then split privious crack tip into two parts """
        ladv = np.zeros(len(self.tipcra))
        ladv[np.int32(np.where(abs(keq) >= KIC)[0])] = lmin*(Gi[np.int32(np.where(abs(keq) >= KIC)[0])]/max(Gi[np.int32(np.where(abs(keq) >= KIC)[0])]))**0.35
        consif = abs(keq) >=  KIC; conlen = ladv >= 0.5*lmin
        crapro = consif*conlen
        for i in range(len(crapro)):
            if tipava[i] == True and crapro[i] == True:
                tip = self.tipcra[i]
                if len(tip) == 1:
                    tip = tip[0]
                pvcr = self.nodcra[i]
                if len(pvcr) == 1:
                    pvcr = pvcr[0]
                pa = self.roscra[i]
                tipx = ladv[i]*np.cos(craang[i] + gloang[i] + iniang[i])
                tipy = ladv[i]*np.sin(craang[i] + gloang[i] + iniang[i])
                tipnew = np.array(([tip[0] + tipx, tip[1] + tipy]))
                discra = np.zeros(10) + 10**3
                for j in range(len(crapro)):
                    if i != j:
                        discra[j] = aux.p2segment(tipnew.reshape(1,2), self.nodcra[j])
                if min(discra) > 3.0*max(ladv):      
                    dis = np.sqrt((pa[:,0] - tipnew[0])**2 + (pa[:,1] - tipnew[1])**2)
                    ind = np.argmin(dis)
                    local = aux.p2index(self.p, pa[ind,:], 0)
                    self.p[local,0] = tipnew[0]; self.p[local,1] = tipnew[1]
                    # draw = PlotModel.draw_model(t, p, 3, 1)    
                    self.p, self.t, self.tipcra, self.moucra, self.nodcra = self.splitelement(i, tipnew)
                    gloang[i] = gloang[i] + craang[i]
                else:
                    dis = np.sqrt((pa[:,0] - tipnew[0])**2 + (pa[:,1] - tipnew[1])**2)
                    ind = np.argmin(dis)
                    local = aux.p2index(self.p, pa[ind,:], 0)
                    self.p[local,0] = tipnew[0]; self.p[local,1] = tipnew[1]   
                    self.p, self.t, self.tipcra, self.moucra, self.nodcra = self.splitelement(i, tipnew)
                    self.p, self.t, self.tipcra, self.moucra, self.nodcra = self.connectcrack(i, self.nodcra[np.argmin(discra)])
                    tipava[i] = False
        return self.p, self.t, self.tipcra, self.nodcra, self.moucra, gloang, tipava, crapro
    def divideelement(self, eleref):
        """ Divide elements that need to be refined by four sub-elements"""
        numnod = self.p.shape[0]
        tnew = np.zeros((1,3),np.int32)
        pnew = np.zeros((1,2))
        for j in range(len(eleref)):
            elei = eleref[j]
            index = self.t[elei,:]
            p1 = (self.p[index[0],:] + self.p[index[1],:])/2
            p2 = (self.p[index[0],:] + self.p[index[2],:])/2
            p3 = (self.p[index[1],:] + self.p[index[2],:])/2
            pi = np.array(([[p1[0],p1[1]],
                            [p2[0],p2[1]],
                            [p3[0],p3[1]]]))
            newele = np.array(([[index[0],numnod+j*3,numnod+j*3+1],
                                [index[1],numnod+j*3,numnod+j*3+2],
                                [index[2],numnod+j*3+1,numnod+j*3+2],
                                [numnod+j*3,numnod+j*3+1,numnod+j*3+2]]))
            tnew = np.append(tnew, newele, axis=0)
            pnew = np.append(pnew, pi, axis=0)
        tnew = np.delete(tnew, 0, axis = 0)
        pnew = np.delete(pnew, 0, axis = 0)
        self.t = np.delete(self.t,eleref,axis = 0)
        self.t = np.append(self.t, tnew, axis=0)
        self.p = np.append(self.p, pnew, axis=0)
        self.removeduplicatenode()
        poi, local = self.nodeedge()
        layer = 0
        while len(local) != 0:
            layer = 1
            self.removehangingnode(poi, local,layer)  
            self.removeduplicatenode() 
            poi, local = self.nodeedge()
        return self.p, self.t, len(eleref)
    def removehangingnode(self, poi, local, layer):
        """ Remove hanging nodes on the mesh"""
        numnod = self.p.shape[0]
        tnew = np.zeros((1,3),np.int32)
        pnew = np.zeros((1,2))
        eledel = np.zeros((1,1),np.int32)
        if layer <= 2:
            cou = 0
            for i in range(len(local)):
                pi = poi[i,:]
                x,y = pi
                for e in range(self.t.shape[0]):
                    pv = self.p[self.t[e,:],:]
                    pv = np.append(pv,pv[:1,:], axis = 0)
                    ds = aux.p2segment(pi.reshape(1,2), pv)
                    dv = min((pi[0] - pv[:,0])**2 + (pi[1] - pv[:,1])**2)
                    
                    if ds <= np.finfo(float).eps*1e5 and dv != 0:
                        eledel = np.append(eledel, e)
                        d01 = aux.p2segment(pi.reshape(1,2), self.p[self.t[e,[0,1]],:])
                        d02 = aux.p2segment(pi.reshape(1,2), self.p[self.t[e,[0,2]],:])
                        l01 = (self.p[self.t[e,0],0] - self.p[self.t[e,1],0])**2 + (self.p[self.t[e,0],1] - self.p[self.t[e,1],1])**2
                        l02 = (self.p[self.t[e,0],0] - self.p[self.t[e,2],0])**2 + (self.p[self.t[e,0],1] - self.p[self.t[e,2],1])**2
                        l12 = (self.p[self.t[e,1],0] - self.p[self.t[e,2],0])**2 + (self.p[self.t[e,1],1] - self.p[self.t[e,2],1])**2
                        if d01 <= np.finfo(float).eps*1e5:
                            if l01 >= max(l02,l12):
                                te = np.array(([[local[i], self.t[e,2], self.t[e,0]],
                                                [local[i], self.t[e,1], self.t[e,2]]]))
                                tnew = np.append(tnew,te,axis = 0)
                            elif l02 >= max(l01,l12):
                                p02 = (self.p[self.t[e,0],:] + self.p[self.t[e,2],:])/2
                                pe = np.array(([[p02[0],p02[1]]]))
                                te = np.array(([[local[i], numnod + cou, self.t[e,0]],
                                                [local[i], self.t[e,1], numnod + cou],
                                                [self.t[e,1], self.t[e,2], numnod + cou]]))
                                tnew = np.append(tnew,te,axis = 0)
                                pnew = np.append(pnew,pe,axis = 0)
                                cou = cou + 1 
                            else:
                                p12 = (self.p[self.t[e,1],:] + self.p[self.t[e,2],:])/2
                                pe = np.array(([[p12[0],p12[1]]]))
                                te = np.array(([[local[i], numnod + cou, self.t[e,0]],
                                                [local[i], self.t[e,1], numnod + cou],
                                                [self.t[e,0], numnod + cou, self.t[e,2]]]))
                                tnew = np.append(tnew,te,axis = 0)
                                pnew = np.append(pnew,pe,axis = 0)
                                cou = cou + 1 
                        elif d02 <= np.finfo(float).eps*1e5:
                            if l02 >= max(l01,l12):
                                te = np.array(([[local[i], self.t[e,0], self.t[e,1]],
                                                [local[i], self.t[e,1], self.t[e,2]]]))
                                tnew = np.append(tnew,te,axis = 0)
                            elif l01 >= max(l02,l12):
                                p01 = (self.p[self.t[e,0],:] + self.p[self.t[e,1],:])/2
                                pe = np.array(([[p01[0],p01[1]]]))
                                te = np.array(([[local[i], self.t[e,0], numnod + cou],
                                                [local[i], numnod + cou, self.t[e,2]],
                                                [self.t[e,1], self.t[e,2], numnod + cou]]))
                                tnew = np.append(tnew,te,axis = 0)
                                pnew = np.append(pnew,pe,axis = 0)
                                cou = cou + 1
                            else:
                                p12 = (self.p[self.t[e,1],:] + self.p[self.t[e,2],:])/2
                                pe = np.array(([[p12[0],p12[1]]]))
                                te = np.array(([[local[i], self.t[e,0], numnod + cou],
                                                [local[i], numnod + cou, self.t[e,2]],
                                                [self.t[e,0], self.t[e,1], numnod + cou]]))
                                tnew = np.append(tnew,te,axis = 0)
                                pnew = np.append(pnew,pe,axis = 0)
                                cou = cou + 1
                        else:
                            if l12 >= max(l01,l02):
                                te = np.array(([[local[i], self.t[e,0], self.t[e,1]],
                                                [local[i], self.t[e,2], self.t[e,0]]]))
                                tnew = np.append(tnew,te,axis = 0)
                            elif l01 >= max(l02,l12):
                                p01 = (self.p[self.t[e,0],:] + self.p[self.t[e,1],:])/2
                                pe = np.array(([[p01[0],p01[1]]]))
                                te = np.array(([[local[i], numnod + cou, self.t[e,1]],
                                                [local[i], self.t[e,2], numnod + cou],
                                                [self.t[e,2], self.t[e,0], numnod + cou]]))
                                tnew = np.append(tnew,te,axis = 0)
                                pnew = np.append(pnew,pe,axis = 0)
                                cou = cou + 1
                            else:
                                p02 = (self.p[self.t[e,0],:] + self.p[self.t[e,2],:])/2
                                pe = np.array(([[p02[0],p02[1]]]))
                                te = np.array(([[local[i], numnod + cou, self.t[e,1]],
                                                [local[i], self.t[e,2], numnod + cou],
                                                [self.t[e,0], self.t[e,1], numnod + cou]]))
                                tnew = np.append(tnew,te,axis = 0)
                                pnew = np.append(pnew,pe,axis = 0)
                                cou = cou + 1
                        break
        else: # layer = 2,3,4
            cou = 0
            for i in range(len(local)):
                pi = poi[i,:]
                x,y = pi
                for e in range(self.t.shape[0]):
                    pv = self.p[self.t[e,:],:]
                    pv = np.append(pv,pv[:1,:], axis = 0)
                    ds = aux.p2segment(pi.reshape(1,2), pv)
                    dv = min((pi[0] - pv[:,0])**2 + (pi[1] - pv[:,1])**2)
                    
                    if ds <= np.finfo(float).eps*1e5 and dv != 0:
                        eledel = np.append(eledel, e)
                        d01 = aux.p2segment(pi.reshape(1,2), self.p[self.t[e,[0,1]],:])
                        d02 = aux.p2segment(pi.reshape(1,2), self.p[self.t[e,[0,2]],:])
                        l01 = (self.p[self.t[e,0],0] - self.p[self.t[e,1],0])**2 + (self.p[self.t[e,0],1] - self.p[self.t[e,1],1])**2
                        l02 = (self.p[self.t[e,0],0] - self.p[self.t[e,2],0])**2 + (self.p[self.t[e,0],1] - self.p[self.t[e,2],1])**2
                        l12 = (self.p[self.t[e,1],0] - self.p[self.t[e,2],0])**2 + (self.p[self.t[e,1],1] - self.p[self.t[e,2],1])**2
                        if d01 <= np.finfo(float).eps*1e5:
                            te = np.array(([[local[i], self.t[e,0], self.t[e,2]],
                                            [local[i], self.t[e,1], self.t[e,2]]]))
                            tnew = np.append(tnew,te,axis = 0)
                            cou = cou + 1 
                        elif d02 <= np.finfo(float).eps*1e5:
                            te = np.array(([[local[i], self.t[e,0], self.t[e,1]],
                                            [local[i], self.t[e,2], self.t[e,1]]]))
                            tnew = np.append(tnew,te,axis = 0)
                            cou = cou + 1
                        else:
                            te = np.array(([[local[i], self.t[e,0], self.t[e,1]],
                                            [local[i], self.t[e,0], self.t[e,2]]]))
                            tnew = np.append(tnew,te,axis = 0)
                            cou = cou + 1
                        break
        tnew = np.delete(tnew,0,axis = 0)   
        pnew = np.delete(pnew,0,axis = 0)           
        self.t = np.append(self.t, tnew, axis=0)  
        self.p = np.append(self.p, pnew, axis=0)            
        self.t = np.delete(self.t,eledel[1::],axis = 0) 
    def removeduplicatenode(self):
        """ Check and remove duplicate nodes"""
        index = np.zeros((1,2),np.int32)
        for k in range(self.p.shape[0],1,-1):
            pk = self.p[k-1,:]
            dis = np.sqrt( (self.p[:k-1,0] - pk[0])**2 + (self.p[:k-1,1] - pk[1])**2)
            local = np.where(dis < np.finfo(float).eps*1e5)[0]
            if len(local) != 0:
                index = np.append(index, np.array(([k-1, local[0]])).reshape(1,2), axis = 0)
        index = np.delete(index, 0, axis = 0)
        if len(index) > 0:
            self.p = np.delete(self.p,index[:,0],axis = 0)
            for ni in range(index.shape[0]):
                id1,id2 = np.where(self.t == index[ni,0])
                for mi in range(len(id1)):
                    self.t[id1[mi],id2[mi]] = index[ni,1]
        tca = np.unique(self.t)
        tcb = np.unique(self.t)
        while max(tca) > len(tca)-1:
            t1 = tca[1::]
            t2 = tca[:-1]
            t0 = t1 - t2
            t0 = np.insert(t0,0,0)
            index = np.where(t0>1)[0]
            tca[index] = tca[index] - 1
        for ni in range(len(tca)):
            id1,id2 = np.where(self.t == tcb[ni])
            for mi in range(len(id1)):
                self.t[id1[mi],id2[mi]] = tca[ni]   
    def removeduplicateelement(self):
        xc = (self.p[self.t[:,0],:] + self.p[self.t[:,1],:] + self.p[self.t[:,2],:])/3
        t0 = np.sort(self.t, axis = 1)
        t0 = np.unique(self.t, axis = 0)
        if t0.shape[0] < self.t.shape[0]:
            eledel = []
            for i in range(xc.shape[0]):
                xci = xc[i,:]
                for j in range(i+1,xc.shape[0]):
                    xcj = xc[j,:]
                    if abs(xci[0] - xcj[0]) < np.finfo(float).eps*1e5 and abs(xci[1] - xcj[1]) < np.finfo(float).eps*1e5:
                        eledel.append(j)
            self.t = np.delete(self.t,eledel, axis = 0) 
    def nodeedge(self):
        """ Check if a node belong to a edge in the mesh"""
        poi = np.zeros((1,2))
        local = np.zeros((1,1),np.int32)
        for e in range (self.t.shape[0]):
            pv = self.p[self.t[e,:],:]
            pv = np.append(pv,pv[:1,:], axis = 0)
            ds = aux.p2segment(self.p, pv)
            ind = np.where(ds < np.finfo(float).eps*1e5)[0]
            if len(ind) > 3:
                indp = np.setdiff1d(ind,self.t[e,:])
                poi = np.append(poi,self.p[indp,:].reshape(len(indp),2), axis = 0)
                local = np.append(local, indp)
        poi = np.delete(poi, 0, axis = 0)
        local = np.delete(local, 0, axis = 0)
        return poi, local
    def tipmesh(self, tip, pvcr, lmin): 
        """ Replace region around a crack tip by new mesh including the quater point elements """
        if len(tip) == 1:
            tip = tip[0]
        if len(pvcr) == 1:
            pvcr = pvcr[0]
        id1 = np.where(np.isclose(self.p[:,0],tip[0]))[0]
        id2 = np.where(np.isclose(self.p[:,1],tip[1]))[0]
        tipind = np.intersect1d(id2,id1)[0]
        bcInd = aux.p2index(self.p,pvcr,0)
        elearotip = np.where(self.t == tipind)[0]
        indarotip = self.t[elearotip,:]
        indarotip = np.setdiff1d(np.unique(indarotip),tipind)
        id0 = np.intersect1d(indarotip,bcInd)
        P1 = self.p[tipind,:]
        P2 = (self.p[id0[0],:] + self.p[id0[1],:])/2
        P31 = self.p[id0[0],:]
        d01 = (P2[0] - P1[0])*(P31[1] - P1[1]) - (P2[1] - P1[1])*(P31[0] - P1[0]) 
        if d01 > 0:
            P1Ind = id0[0]
            P2Ind = id0[1]
        else:
            P1Ind = id0[1]
            P2Ind = id0[0]
        l1 = np.sqrt(sum((self.p[P1Ind,:] - self.p[tipind,:])**2))
        l2 = np.sqrt(sum((self.p[P2Ind,:] - self.p[tipind,:])**2))
        if l1 > 1.8*l2:
            pnew = 1/2*(tip + P1).reshape(1,2)
            indnew = self.p.shape[0]
            elein = np.intersect1d(np.where(self.t == tipind)[0], np.where(self.t == P1Ind)[0])
            id0 = np.setdiff1d(self.t[elein,:],[tipind,P1Ind])[0]
            tnew = np.array(([[id0, tipind, indnew],
                                [id0, P1Ind, indnew]]))
            self.t = np.delete(self.t, elein, axis = 0)
            self.t = np.append(self.t, tnew, axis=0)  
            self.p = np.append(self.p, pnew, axis=0)  
            P1Ind = indnew            
        elif l2 > 1.8*l1:
            pnew = 1/2*(tip + P2).reshape(1,2)
            indnew = self.p.shape[0]
            elein = np.intersect1d(np.where(self.t == tipind)[0], np.where(self.t == P2Ind)[0])
            id0 = np.setdiff1d(self.t[elein,:],[tipind,P2Ind])[0]
            tnew = np.array(([[id0, tipind, indnew],
                                [id0, P2Ind, indnew]]))
            self.t = np.delete(self.t, elein, axis = 0)
            self.t = np.append(self.t, tnew, axis=0)  
            self.p = np.append(self.p, pnew, axis=0)  
            P2Ind = indnew       
        pbel0 = self.p[P1Ind,:]
        pupp0 = self.p[P2Ind,:]
        tip = self.p[tipind,:]
        p1, t1, inOut, InCen, r = self.rosette(pbel0,pupp0,tip,lmin)
        pbel = p1[inOut[-1],:]
        pupp = p1[inOut[0],:]
        NodArouP1 = self.t[np.where(self.t == P1Ind)[0],:]
        NodArouP1 = np.unique(NodArouP1)
        A1Ind = np.setdiff1d(np.intersect1d(NodArouP1,bcInd),np.array([P1Ind,tipind]))[0]
        NodArouP2 = self.t[np.where(self.t == P2Ind)[0],:]
        NodArouP2 = np.unique(NodArouP2)
        A2Ind = np.setdiff1d(np.intersect1d(NodArouP2,bcInd),np.array([P2Ind,tipind]))[0]
        IndOut,nodedelete,eledelete = self.regionremesh(tipind,np.max(np.array(([r,lmin]))),A1Ind,A2Ind)
        pv1 = p1[inOut[-1],:]
        pv1 = pv1.reshape(1,len(pv1))
        pv2 = self.p[IndOut,:]
        pv3 = p1[inOut,:]
        pv = np.concatenate((pv1,pv2,pv3), axis = 0)
        p2, t2 = self.fillgap(pv)
        p12, t12 = self.assemblymesh(p1,t1,p2,t2)
        self.p = np.delete(self.p, nodedelete, 0)
        self.t = np.delete(self.t, eledelete, 0)
        p0 = np.copy(self.p)
        t0 = np.copy(self.t)
        for k in range(len(nodedelete),0,-1):
            t0[t0 > nodedelete[k-1]] = t0[t0 > nodedelete[k-1]] - 1
        self.p, self.t = self.assemblymesh(p12,t12,p0,t0)  
        Paround = p1[1:,:] 
        id0 = np.intersect1d(np.where(tip[0] == self.p[:,0])[0],np.where(tip[1] == self.p[:,1])[0])
        idb = np.intersect1d(np.where(pbel[0] == self.p[:,0])[0],np.where(pbel[1] == self.p[:,1])[0])
        idu = np.intersect1d(np.where(pupp[0] == self.p[:,0])[0],np.where(pupp[1] == self.p[:,1])[0])
        indexCQPE = np.array([idb, id0, idu])
        self.p[idb,0] = pbel0[0]; self.p[idb,1] = pbel0[1]
        self.p[idu,0] = pupp0[0]; self.p[idu,1] = pupp0[1]
        return self.p, self.t, pbel0,pupp0,Paround,indexCQPE
    def rosette(self, P1, P2, O, lmin):
        """ Create a rosette at a crack tip mouth P1-O-P2 """
        angle0 = aux.angle(P1,O,P2)
        P11 = np.array([1,0])
        P00 = np.array([0,0])
        P22 = (P1+P2)/2-O
        angle = aux.angle(P11,P00,P22)
        x1 = 1
        y1 = 0
        x2 = 0
        y2 = 0
        x3 = P1[0] - O[0]
        y3 = P1[1] - O[1]
        d0 = (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1)
        rota = -np.sign(d0)
        n = 6
        alpha = [i*(2*math.pi - angle0)/n + angle0/2 for i in range(n+1)] 
        if lmin == 0:
            r = np.sqrt(sum((O - P1)**2))
        else:
            r = lmin
        xin = r*np.cos(alpha + rota*angle)
        yin = r*np.sin(alpha + rota*angle)
        xin = xin + O[0]
        yin = yin + O[1]
        xin = xin.reshape(len(xin),1)
        yin = yin.reshape(len(yin),1)
        coord1 = np.concatenate((xin,yin), axis=1)
        coord = np.concatenate((O.reshape(1,len(O)),coord1), axis=0)
        id1 = np.arange(0,n)+1
        id2 = np.arange(1,n+1) +1
        id1 = id1.reshape(len(id1),1)
        id2 = id2.reshape(len(id2),1)
        IndIn1 = np.concatenate((id1,id2), axis=1)
        nodes = np.ones((IndIn1.shape[0],3),np.int32)*0
        nodes[:,1] = IndIn1[:,0]
        nodes[:,2] = IndIn1[:,1]
        inOut = np.arange(n+1,0,-1) 
        InCen = 0
        return coord, nodes, inOut, InCen, r
    def regionremesh(self, tipind, r, A1Ind, A2Ind): 
        """ Determine elements around a crack tip that need to be replaced """
        EleIn = np.where(self.t == tipind)[0]
        IndIn = self.t[EleIn,:]
        IndIn = np.setdiff1d(np.unique(IndIn),tipind)
        tt = np.copy(self.t)
        p0 = self.p[tipind,:]
        nodescloset = np.where(np.sqrt((self.p[:,0] - p0[0])**2 + (self.p[:,1] - p0[1])**2) <= 1.2*r)
        nodescloset = np.union1d(nodescloset,IndIn)
        
        elemecloset = np.where(np.array([aux.ismember(self.t[:,0],nodescloset), 
                                         aux.ismember(self.t[:,1],nodescloset), 
                                         aux.ismember(self.t[:,2],nodescloset)]) == 1)[1]
        elemecloset = np.unique(elemecloset)
        nodesaround = self.t[elemecloset,:]
        nodesaround = np.unique(nodesaround)
        Elei = np.array([aux.ismember(self.t[:,0],nodesaround), 
                         aux.ismember(self.t[:,1],nodesaround), 
                         aux.ismember(self.t[:,2],nodesaround)])
        
        eledelete = np.where(np.sum(Elei,axis = 0) == 3)[0]
        tt = np.delete(tt, eledelete, 0)
        tt = np.unique(tt)
        nodesoutside = np.intersect1d(nodesaround,tt)
        nodedelete = np.setdiff1d(nodesaround,nodesoutside)
        IndOut = A1Ind
        nodesoutside = np.setdiff1d(nodesoutside,np.array([A1Ind,A2Ind]))
        i = 0
        while len(nodesoutside) > 0:
            i = i + 1
            elearou = eledelete[np.where(self.t[eledelete,:] == A1Ind)[0]]
            A1Ind = np.intersect1d(np.unique(self.t[elearou,:]),nodesoutside)
            IndOut = np.append(IndOut,A1Ind)
            nodesoutside = np.setdiff1d(nodesoutside,A1Ind)
        IndOut = np.append(IndOut,A2Ind)
        return IndOut,nodedelete,eledelete
    def fillgap(self, pv):
        """ Fill a region that has been deleted and only was replaced by rosette """
        from shapely.geometry import Polygon
        polygon = Polygon(pv)
        arepol = polygon.area
        areele = 0
        numnod = pv.shape[0] - 1
        edge1 = pv[:2,:]
        i = 0
        j = 1
        edgind = [0, 1]
        t = np.zeros((1,3),np.int32)
        
        while arepol > np.sum(areele):
            ang1 = aux.angle(edge1[0,:],pv[i+2,:],edge1[1,:])
            ang2 = aux.angle(edge1[0,:],pv[numnod-j,:],edge1[1,:])
            if ang1 > ang2:
                ti = np.array([edgind[0], edgind[1], i+2],np.int32)
                
                p13 = (pv[ti[0],:] + pv[ti[2],:])/2 ; p13 = p13.reshape(1,2)
                d13 = aux.p2segment(p13, pv)
                
                p23 = (pv[ti[1],:] + pv[ti[2],:])/2 ; p23 = p23.reshape(1,2)
                d23 = aux.p2segment(p23, pv)
        
                    
                if d13 > np.finfo(float).eps*1e5:
                    edgind = [ti[0], ti[2]]
                    
                if d23 > np.finfo(float).eps*1e5:
                    edgind = [ti[1], ti[2]]
                i = i + 1
            else:
                ti = np.array([edgind[0], edgind[1], numnod-j])
        
                p13 = (pv[ti[0],:] + pv[ti[2],:])/2 ; p13 = p13.reshape(1,2)
                d13 = aux.p2segment(p13, pv)
                
                p23 = (pv[ti[1],:] + pv[ti[2],:])/2 ; p23 = p23.reshape(1,2)
                d23 = aux.p2segment(p23, pv)
                    
                if d13 > np.finfo(float).eps*1e5:
                    edgind = [ti[0], ti[2]]
                    
                if d23 > np.finfo(float).eps*1e5:
                    edgind = [ti[1], ti[2]]
                j = j + 1
            t = np.concatenate((t,ti.reshape(1,3)),axis = 0)
            xcoord = np.zeros((t.shape[0]-1,3))
            xcoord[:,0] = pv[t[1:,0],0]
            xcoord[:,1] = pv[t[1:,1],0]
            xcoord[:,2] = pv[t[1:,2],0]
            
            ycoord = np.zeros((t.shape[0]-1,3))
            ycoord[:,0] = pv[t[1:,0],1]
            ycoord[:,1] = pv[t[1:,1],1]
            ycoord[:,2] = pv[t[1:,2],1]
            
            areele = aux.area(xcoord,ycoord)
            edge1 = pv[edgind,:]
        t2 = np.delete(t,0,axis = 0)
        p2 = pv[:-1,:]
        """The code bellow is more simple, but some time it dose not work"""
        # p2 = pv[0:-1,:]
        # tri = Delaunay(p2)
        # t2 = tri.simplices
        # celcen = (p2[t2[:,0],:] + p2[t2[:,1],:] + p2[t2[:,2],:])/3
        # inpol = aux.inpolygon(celcen,pv)
        # t2 = t2[inpol,:]
        return p2,t2
    def assemblymesh(self, p1, t1, p2, t2):
        """ assembly mesh p1-t1 to mesh p2-t2"""
        p12 = np.concatenate((p2,p1), axis = 0)
        t12 = np.concatenate((t2,t1+t2.max(1).max(0)+1), axis = 0)
        indica = np.empty((p12.shape[0],2),np.int32)
        for i in range(p12.shape[0]):
            indica[i,0] = i
            for j in range(p12.shape[0]):
                if p12[i,0] == p12[j,0] and p12[i,1] == p12[j,1]:
                    indica[i,1] = j
                    break
        NodDup = indica[indica[:,0] != indica[:,1],:]
        NodNor = indica[indica[:,0] == indica[:,1],:]
        p12 = p12[NodNor[:,0],:]    
            
        for k in range(NodDup.shape[0]):
            t12[t12 == NodDup[k,0]] = NodDup[k,1]
        deleno = NodDup[:,0]
        for k in range(len(deleno),0,-1):
            t12[t12 > deleno[k-1]] = t12[t12 > deleno[k-1]] - 1
        return p12,t12
    def smoothing(self, nodfix):
        """ Smooth a mesh by using the Laplacian smoothing"""
        nodall = np.int32([i for i in range(self.p.shape[0])])
        nodche = np.setdiff1d(nodall,nodfix)
        for j in range(2):
            for i in range(len(nodche)):
                nodmov = self.p[nodche[i],:]
                elearo = np.where(self.t == nodche[i])[0]
                indaro = self.t[elearo,:]
                X, Y = np.transpose(self.p[indaro,:])
                Ae = aux.area(X,Y); Ae = Ae.reshape(len(Ae),1)
                totare = sum(Ae)
                indaro1 = np.setdiff1d(np.unique(indaro), nodche[i])
                p0 = np.sum(self.p[indaro1,:],axis = 0)/len(indaro1)
                self.p[nodche[i],0] = p0[0];self.p[nodche[i],1] = p0[1]
                
                X, Y = np.transpose(self.p[indaro,:])
                Ae = aux.area(X,Y); Ae = Ae.reshape(len(Ae),1)
                totare1 = sum(Ae)
                if totare1 > totare:
                    self.p[nodche[i],0] = nodmov[0];self.p[nodche[i],1] = nodmov[1]
    def splitelement(self, tipi, tipnew):
        tip = self.tipcra[tipi]
        pb = self.moucra[tipi][0]
        pt = self.moucra[tipi][1]
        pvcra = self.nodcra[tipi]
        
        bb = aux.p2index(self.p, pb, 0)
        bt = aux.p2index(self.p, pt, 0)
        b1 = aux.p2index(self.p, tip, 0)
        b2 = aux.p2index(self.p, tipnew, 0)
        
        elearo = np.where(self.t == b1)[0]
        indout = self.t[elearo,:]; indout = np.unique(indout)
        indout = np.setdiff1d(indout,[bb,bt,b1])
        indouti = np.copy(bb)
        A0 = (self.p[bb,:] +  self.p[bt,:])/2
        A1 = self.p[b2,:]
        A2 = self.p[indout,:]
        d0 = (A1[0,0] - A0[0,0])*(A2[:,1] - A0[0,1]) - (A1[0,1] - A0[0,1])*(A2[:,0] - A0[0,0])                
        posnod = np.where(d0 <= 0)[0]
        bb = indout[posnod]
        indouti = np.append(bb,indouti[-1])   
        sumind = np.array(aux.ismember(self.t[elearo,0],indouti)) + np.array(aux.ismember(self.t[elearo,1],indouti)) + np.array(aux.ismember(self.t[elearo,2],indouti))
        group1 = elearo[np.where(sumind == 2)[0]]
        p0 = (pb + pt)/2
        vt1 = tip - p0
        nt1 = np.array(([-vt1[1], vt1[0]]))/aux.norm(vt1)
        
        vt2 = tipnew - p0
        nt2 = np.array(([-vt2[1], vt2[0]]))/aux.norm(vt2)
        
        nt = nt1 + nt2; nt = nt/aux.norm(nt)
        pbnew = tip - nt*self.dc/2
        ptnew = tip + nt*self.dc/2
        totnod = self.p.shape[0]
        self.p[b1,0] = ptnew[0]; self.p[b1,1] = ptnew[1]
        self.p = np.concatenate((self.p,pbnew.reshape(1,2)),axis = 0)
        for j in range(len(group1)):
            idecha = np.where(self.t[group1[j],:] == b1)[0]
            self.t[group1[j],idecha] = totnod
        tipincra = np.intersect1d(np.where(pvcra[:,0] == tip[0])[0], np.where(pvcra[:,1] == tip[1])[0])[0]
        nodcranew = np.zeros(shape = (pvcra.shape[0]+4,2))
        nodcranew[:tipincra,:] = pvcra[:tipincra,:]
        nodcranew[tipincra,:] = pt
        nodcranew[tipincra+1,:] = ptnew
        nodcranew[tipincra+2,:] = tipnew
        nodcranew[tipincra+3,:] = pbnew
        nodcranew[tipincra+4,:] = pb
        nodcranew[tipincra+5::,:] = pvcra[tipincra+1::,:]
        self.tipcra[tipi] = tipnew
        self.moucra[tipi] = np.concatenate(([pbnew.reshape(1,2), ptnew.reshape(1,2)]))
        self.nodcra[tipi] = nodcranew
        return self.p, self.t, self.tipcra, self.moucra, self.nodcra
    def connectcrack(self, tipi, confac):
        tip = self.tipcra[tipi]
        pb = self.moucra[tipi][0]
        pt = self.moucra[tipi][1]
        
        bb = aux.p2index(self.p, pb, 0)
        bt = aux.p2index(self.p, pt, 0)
        b1 = aux.p2index(self.p, tip, 0)
        bc = aux.p2index(self.p, confac, 0)
        dis0 = 1E20
        for i in range(len(bc)):
            poi = self.p[bc[i],:]
            dis = np.sqrt(sum((tip - poi)**2))
            if dis < dis0:
                dis0 = np.copy(dis)
                pnew = poi
        bar = np.concatenate((self.t[:,[0,1]],self.t[:,[0,2]],self.t[:,[1,2]]),axis = 0)
        bar = np.unique(bar,axis = 0)
        for i in range(bar.shape[0]):
            if bar[i,0] > bar[i,1]:
                bari = bar[i,0]
                bar[i,0] = bar[i,1]
                bar[i,1] = bari
        bar = np.unique(bar,axis = 0)
        lmin = np.sqrt(sum((tip - pb)**2))
        for i in range(self.p.shape[0]):
            M = self.p[i,:]
            N, flag, dis = aux.projection(tip,pnew,M)
            if flag == 2 and dis < lmin/8 and dis > 0:
                self.p[i,:] = N.reshape(1,2)
        pnew0 = np.zeros((1,2),np.int32)
        for i in range(bar.shape[0]):
            M1 = self.p[bar[i,0],:]
            M2 = self.p[bar[i,1],:]
            N, flag = aux.intersection(tip,pnew,M1,M2)
            d1 = (pnew[0] - tip[0])*(M1[1] - tip[1]) - (pnew[1] - tip[1])*(M1[0] - tip[0])
            d2 = (pnew[0] - tip[0])*(M2[1] - tip[1]) - (pnew[1] - tip[1])*(M2[0] - tip[0])
            if abs(d1) > np.finfo(float).eps*1E5 and abs(d2) > np.finfo(float).eps*1E5 and np.sign(d1) != np.sign(d2) and flag == 2:
                pnew0 = np.concatenate((pnew0,bar[i,:].reshape(1,2)), axis = 0)
        pnew0 = np.delete(pnew0, 0, axis=0)        
        for i in range(pnew0.shape[0]):
            M = pnew0[i,:]
            N1, flag1, dis1 = aux.projection(tip,pnew,self.p[M[0],:])
            N2, flag2, dis2 = aux.projection(tip,pnew,self.p[M[1],:])
            if dis1 < dis2:
                self.p[M[0],:] = N1.reshape(1,2)
            else:
                self.p[M[1],:] = N2.reshape(1,2)
        
        poi = aux.p2index(self.p,np.array(([tip, pnew])),1) 
        cutpoi = self.p[poi,:]
        for i in range(len(poi)):
            if i < len(poi)-1:
                O2 = cutpoi[i+1,:]
            else:
                O2 = []
            if len(O2) == 2:
                O1 = self.tipcra[tipi]
                B1 = self.moucra[tipi][0] # pb
                B2 = self.moucra[tipi][1] # pt
                pvcra = self.nodcra[tipi]
      
                bb = aux.p2index(self.p, B1, 0)
                bt = aux.p2index(self.p, B2, 0)
                b1 = aux.p2index(self.p, O1, 0)
                b2 = aux.p2index(self.p, O2, 0)
                elearo = np.where(self.t == b1)[0]
                indout = np.unique(self.t[elearo,:])
                indout = np.setdiff1d(indout,[bb,bt,b1])
                indouti = np.copy(bb)
                A0 = (self.p[bb,:] +  self.p[bt,:])/2
                A1 = self.p[b2,:]
                A2 = self.p[indout,:]
                d0 = (A1[0,0] - A0[0,0])*(A2[:,1] - A0[0,1]) - (A1[0,1] - A0[0,1])*(A2[:,0] - A0[0,0])                
                posnod = np.where(d0 <= 0)[0]
        
                bb = indout[posnod]
                indouti = np.append(bb,indouti[-1])  
                sumind = np.array(aux.ismember(self.t[elearo,0],indouti)) + np.array(aux.ismember(self.t[elearo,1],indouti)) + np.array(aux.ismember(self.t[elearo,2],indouti))
                group1 = elearo[np.where(sumind == 2)[0]]
                
                p0 = (B1 + B2)/2
                vt1 = O1 - p0
                nt1 = np.array(([-vt1[1], vt1[0]]))/aux.norm(vt1)
                
                vt2 = O2 - O1
                nt2 = np.array(([-vt2[1], vt2[0]]))/aux.norm(vt2)
                
                nt = nt1 + nt2; nt = nt/aux.norm(nt)

                pbnew = O1 - nt*self.dc/2
                ptnew = O1 + nt*self.dc/2
                totnod = self.p.shape[0]
                self.p[b1,0] = ptnew[0]; self.p[b1,1] = ptnew[1]
                self.p = np.concatenate((self.p,pbnew.reshape(1,2)),axis = 0)
                for j in range(len(group1)):
                    idecha = np.where(self.t[group1[j],:] == b1)[0]
                    self.t[group1[j],idecha] = totnod
                
                tipincra = np.intersect1d(np.where(pvcra[:,0] == O1[0])[0], np.where(pvcra[:,1] == O1[1])[0])[0]
                nodcranew = np.zeros(shape = (pvcra.shape[0]+4,2))
                nodcranew[:tipincra,:] = pvcra[:tipincra,:]
                nodcranew[tipincra,:] = B2
                nodcranew[tipincra+1,:] = ptnew
                nodcranew[tipincra+2,:] = O2
                nodcranew[tipincra+3,:] = pbnew
                nodcranew[tipincra+4,:] = B1
                nodcranew[tipincra+5::,:] = pvcra[tipincra+1::,:]
                self.tipcra[tipi] = O2
                self.moucra[tipi] = np.concatenate(([pbnew.reshape(1,2), ptnew.reshape(1,2)]))
                self.nodcra[tipi] = nodcranew
            else:
                O1 = self.tipcra[tipi]
                B1 = self.moucra[tipi][0] # pb
                B2 = self.moucra[tipi][1] # pt
                pvcra = self.nodcra[tipi]
      
                bb = aux.p2index(self.p, B1, 0)
                bt = aux.p2index(self.p, B2, 0)
                b1 = aux.p2index(self.p, O1, 0)
                
                elearo = np.where(self.t == b1)[0]
                indout = np.unique(self.t[elearo,:])
                indout = np.setdiff1d(indout,[bb,bt,b1])
                indouti = np.copy(bb)
                A0 = (self.p[bb,:] +  self.p[bt,:])/2
                A1 = self.p[b2,:]
                A2 = self.p[indout,:]
                d0 = (A1[0,0] - A0[0,0])*(A2[:,1] - A0[0,1]) - (A1[0,1] - A0[0,1])*(A2[:,0] - A0[0,0])                
                posnod = np.where(d0 <= 0)[0]
        
                bb = indout[posnod]
                indouti = np.append(bb,indouti[-1])  
                sumind = np.array(aux.ismember(self.t[elearo,0],indouti)) + np.array(aux.ismember(self.t[elearo,1],indouti)) + np.array(aux.ismember(self.t[elearo,2],indouti))
                group1 = elearo[np.where(sumind == 2)[0]]
                
                p0 = (B1 + B2)/2
                bc = aux.p2index(self.p, confac, 0)
                bar = np.concatenate((self.t[:,[0,1]],self.t[:,[0,2]],self.t[:,[1,2]]),axis = 0)
                bar = np.unique(bar,axis = 0)
                ii, jj = np.where(bar == b1)
                nodex = np.unique(bar[ii,:])
                id2 = np.intersect1d(bc,nodex)
                id2 = np.setdiff1d(id2, b1)
                poi1 = self.p[id2[0],:]
                u1 = poi1 - O1
                v = O1 - p0
                dir1 = u1[0]*v[1] - u1[1]*v[0]
                if dir1 > 0:
                    nt = u1/aux.norm(u1)
                    pbnew = O1 + nt*self.dc
                    ptnew = O1 - nt*self.dc
                else:
                    nt = u1/aux.norm(u1)
                    pbnew = O1 - nt*self.dc
                    ptnew = O1 + nt*self.dc
                    
                totnod = self.p.shape[0]
                self.p[b1,0] = ptnew[0]; self.p[b1,1] = ptnew[1]
                self.p = np.concatenate((self.p,pbnew.reshape(1,2)),axis = 0)
                for j in range(len(group1)):
                    idecha = np.where(self.t[group1[j],:] == b1)[0]
                    self.t[group1[j],idecha] = totnod
                tipincra = np.intersect1d(np.where(pvcra[:,0] == O1[0])[0], np.where(pvcra[:,1] == O1[1])[0])[0]
                nodcranew = np.zeros(shape = (pvcra.shape[0]+3,2))
                nodcranew[:tipincra,:] = pvcra[:tipincra,:]
                nodcranew[tipincra,:] = B2
                nodcranew[tipincra+1,:] = ptnew
                nodcranew[tipincra+2,:] = pbnew
                nodcranew[tipincra+3,:] = B1
                nodcranew[tipincra+4::,:] = pvcra[tipincra+1::,:]
                
                self.tipcra[tipi] = O2
                self.moucra[tipi] = np.concatenate(([pbnew.reshape(1,2), ptnew.reshape(1,2)]))
                self.nodcra[tipi] = nodcranew
        
        return self.p, self.t, self.tipcra, self.moucra, self.nodcra