import numpy as np 
from scipy import linalg
import math
import numpy.matlib
import auxiliary
aux = auxiliary.AuxiFu()

class Discretization():
    def stiffness(self, p6, t6, material, qpe):
        CQPE = []
        for i in range(len(qpe)):
            if len(qpe[i]) > 0:
                CQPE.append(qpe[i])
        CQPE = np.unique(CQPE)
        poi3, wei3 = GaussPointRule.gausspoint(7, 'T3')
        poi4, wei4 = GaussPointRule.gausspoint(7, 'Q4')
        ne = t6.shape[0]
        sdof = p6.shape[0]*2
        edof = 12
        K = np.zeros((sdof,sdof))
        for e in range(ne):
            if len(material) == 3:
                D = material
            else:
                D = material[e]
            X = p6[t6[e,:],0]
            Y = p6[t6[e,:],1]
            if (X[1] - X[0])*(Y[-1] - Y[0]) - (X[-1] - X[0])*(Y[1] - Y[0]) < 0:
                indinv = t6[e,::-1]
                t6[e,1::] = indinv[0:-1:1]
                X = p6[t6[e,:],0]
                Y = p6[t6[e,:],1] 
            index = [t6[e,0]*2, t6[e,0]*2+1,
                     t6[e,1]*2, t6[e,1]*2+1,
                     t6[e,2]*2, t6[e,2]*2+1,
                     t6[e,3]*2, t6[e,3]*2+1,
                     t6[e,4]*2, t6[e,4]*2+1,
                     t6[e,5]*2, t6[e,5]*2+1]
            if np.sum(e == CQPE) == 0:
                xis = poi3[:,0]
                eta = poi3[:,1]
                N, dNdx, dNdy, detJac = ShapeFunction.T6element(X, Y, xis, eta)
                Ke = np.zeros((edof,edof))
                for i in range(poi3.shape[0]):
                    B = np.zeros((3,edof))
                    B[0,:] = [dNdx[i,0], 0, dNdx[i,1], 0, dNdx[i,2], 0, dNdx[i,3], 0, dNdx[i,4], 0, dNdx[i,5], 0]
                    B[1,:] = [0, dNdy[i,0], 0, dNdy[i,1], 0, dNdy[i,2], 0, dNdy[i,3], 0, dNdy[i,4], 0, dNdy[i,5]]
                    B[2,:] = [dNdy[i,0], dNdx[i,0], dNdy[i,1], dNdx[i,1], dNdy[i,2], dNdx[i,2], dNdy[i,3], dNdx[i,3], dNdy[i,4], dNdx[i,4], dNdy[i,5], dNdx[i,5]]     
                    Ke = Ke + np.dot(np.dot(np.transpose(B),D),B)*detJac[i]*wei3[i]    
            else:
                Ke = np.zeros((edof,edof))
                xis = poi4[:,0]
                eta = poi4[:,1]
                N, dNdx, dNdy, detJac = ShapeFunction.Q8element(X, Y, xis, eta)
                Ke = np.zeros((edof,edof))
                
                for i in range(poi4.shape[0]):
                    B = np.zeros((3,edof))
                    B[0,:] = [dNdx[i,0] + dNdx[i,6] + dNdx[i,7], 0, dNdx[i,1], 0, dNdx[i,2], 0, dNdx[i,3], 0, dNdx[i,4], 0, dNdx[i,5], 0]
                    B[1,:] = [0, dNdy[i,0] + dNdy[i,6] + dNdy[i,7], 0, dNdy[i,1], 0, dNdy[i,2], 0, dNdy[i,3], 0, dNdy[i,4], 0, dNdy[i,5]]
                    B[2,:] = [dNdy[i,0] + dNdy[i,6] + dNdy[i,7], dNdx[i,0] + dNdx[i,6] + dNdx[i,7], dNdy[i,1], dNdx[i,1], dNdy[i,2], dNdx[i,2], dNdy[i,3], dNdx[i,3], dNdy[i,4], dNdx[i,4], dNdy[i,5], dNdx[i,5]]     
                    Ke = Ke + np.dot(np.dot(np.transpose(B),D),B)*detJac[i]*wei4[i]  
            for j in range(edof):
                for i in range(edof):
                    kj = index[j]
                    ki = index[i]
                    K[ki,kj] = K[ki,kj] + Ke[i,j]
        return K
    def loadsegment(self, p6, NeuDof, NeuVal):
        sdof = p6.shape[0]*2
        F = np.zeros((sdof,1))
        for e in range(NeuDof.shape[0]):
            index = NeuDof[e,:]
            dx = np.sqrt((p6[index[2],0] - p6[index[0],0])**2 + (p6[index[2],1] - p6[index[0],1])**2)
            Fex = np.array([1/6,4/6,1/6])*NeuVal[e,0]*dx
            Fey = np.array([1/6,4/6,1/6])*NeuVal[e,1]*dx
            F[index*2,0] = F[index*2,0] + Fex
            F[index*2+1,0] = F[index*2+1,0] + Fey
        return F
    def loadpoint(self, p6, t6, thick, NeuDof, NeuVal):
        sdof = p6.shape[0]*2
        F = np.zeros((sdof,1))
        for e in range(NeuDof.shape[0]):
            F[NeuDof[e,0],0] = NeuVal[e,0]
            F[NeuDof[e,1],0] = NeuVal[e,1]*thick
        return F         
class SetupModel():
    def circle (self, p6, t6, fy):
        indtop = np.where(p6[:,1] == max(p6[:,1]))[0]
        indbot = np.where(p6[:,1] == min(p6[:,1]))[0]
        dirdof = np.concatenate((indbot*2, indbot*2+1))
        dirval = dirdof*0
        neudof = np.array(([indtop*2, indtop*2+1])); neudof = neudof.reshape(1,2)
        neuval = np.array(([0, -fy])); neuval = neuval.reshape(1,2)
        return dirdof, dirval, neudof, neuval
    def shearslip (self, p6, t6, nodaro, tor):
        lx = max(p6[:,0]) - min(p6[:,0])
        topind = aux.p2index(p6,nodaro[[3,2],:],1) 
        botind = aux.p2index(p6,nodaro[[0,1],:],1)  
        
        dirdof = np.concatenate((botind*2+1, topind*2+1))
        dirval = dirdof*0
        
        neudof = np.zeros((np.int((len(topind) - 1)/2) + np.int((len(botind) - 1)/2),3),np.int32)
        neuval = np.zeros((np.int((len(topind) - 1)/2) + np.int((len(botind) - 1)/2),2))
        cou = 0
        for i in range(np.int((len(topind) - 1)/2)):
            index = [2*i, 2*i+1, 2*i+2]
            neudof[cou,:] = topind[index].reshape(1,3)
            neuval[cou,:] = [-tor/lx,0]
            cou = cou + 1
        for i in range(np.int((len(botind) - 1)/2)):
            index = [2*i, 2*i+1, 2*i+2]
            neudof[cou,:] = botind[index].reshape(1,3)
            neuval[cou,:] = [tor/lx,0]   
            cou = cou + 1
        return dirdof, dirval, neudof, neuval
    def hertcontact (self, p6, t6, R, N, boucir, numcir, bourec, f0, phi):
        # material for circle
        young = 7E3
        poisson = 0.3
        matcir = young/(1-poisson**2)*np.array(([[1,poisson,0],
                                            [poisson,1,0],
                                            [0,0,(1-poisson)/2]]))
        # material for rectangle
        young = 1E6
        poisson = 0.45
        matrec = young/(1-poisson**2)*np.array(([[1,poisson,0],
                                            [poisson,1,0],
                                            [0,0,(1-poisson)/2]]))
        material = []
        for i in range(t6.shape[0]):
            if i < numcir:
                material.append(matcir)
            else:
                material.append(matrec)

        bouindcir = aux.p2index(p6,boucir,0) 
        bar = np.concatenate((t6[:numcir,[0,1]],t6[:numcir,[1,2]],t6[:numcir,[2,3]],t6[:numcir,[3,4]],t6[:numcir,[4,5]],t6[:numcir,[5,0]]),axis = 0)
        bar = np.sort(bar, axis = 1)
        bar = np.unique(bar,axis = 0)
        bounodcir = np.array([bouindcir[0]]); m = len(bouindcir)
        while len(bounodcir) != m:
            ind1 = bounodcir[-1]
            bouindcir = np.setdiff1d(bouindcir,ind1)
            indclo = np.unique(bar[np.where(bar == ind1)[0],:])
            ind2 = np.setdiff1d(indclo,ind1)
            ind2 = np.intersect1d(ind2,bouindcir)
            if len(ind2) == 1:
                bounodcir = np.concatenate((bounodcir, ind2))
            else:
                bounodcir = np.concatenate((np.array([ind2[1]]), bounodcir, np.array([ind2[0]])))     
        
        posnodcir = np.concatenate((bounodcir[7*np.int(np.sqrt(N))+1::], bounodcir[0:3*np.int(np.sqrt(N))+2]))   
        posnodrec = aux.p2index(p6,bourec[3:1:-1,:],1)
        
        gap = np.zeros(shape = (p6.shape[0]))
        gap[posnodcir] = -np.sqrt(np.sum((p6[posnodcir,:] - p6[posnodrec,:])**2, axis = 1))    
        
        topnod = bounodcir[np.where(p6[bounodcir,1] >=  R - np.finfo(float).eps*1E3)[0]]
        lefcir = bounodcir[np.where(p6[bounodcir,0] <=  -R + np.finfo(float).eps*1E3)[0]]
        rigcir = bounodcir[np.where(p6[bounodcir,0] >=  R - np.finfo(float).eps*1E3)[0]]
        ind = np.where(bounodcir == topnod)[0]
        numeleloa = 2
        neunod = bounodcir[np.int(ind - numeleloa):np.int(ind + numeleloa + 1)]
        
        lenloa = numeleloa/(boucir.shape[0] - 1)*2*math.pi*R
        neudof = np.zeros((np.int((len(neunod) - 1)/2),3),np.int32)
        neuval = np.zeros((np.int((len(neunod) - 1)/2),2))
        cou = 0
        for i in range(np.int((len(neunod) - 1)/2)):
            index = [2*i, 2*i+1, 2*i+2]
            neudof[cou,:] = neunod[index].reshape(1,3)
            neuval[cou,:] = [f0*math.sin(phi)/lenloa, -f0*math.cos(phi)/lenloa]
            cou = cou + 1
                
        dirnod = np.concatenate((aux.p2index(p6,bourec[0:2,:],0), aux.p2index(p6,bourec[1:3,:],0), aux.p2index(p6,bourec[3:5,:],0)))
        dirdof = np.concatenate((dirnod*2, dirnod*2+1, lefcir*2, lefcir*2+1, rigcir*2, rigcir*2+1))
        dirval = dirdof*0  
        return material, dirdof, dirval, neudof, neuval, posnodcir, posnodrec, gap
    def edgcraextension(self, p6, t6, nodaro, nodmid, sig):
        topind = aux.p2index(p6,nodaro[[3,2],:],1) 
        botind = aux.p2index(p6,nodaro[[0,1],:],1)  
        rigind = aux.p2index(p6,nodaro[[1,2],:],1)  
        midind = aux.p2index(p6,nodmid[[0,-1],:],1)
        
        dirdof = np.concatenate((rigind*2, midind*2+1))
        dirval = 0*dirdof
        neudoft = np.zeros((np.int((len(topind) - 1)/2),3), np.int32)
        neuvalt = np.zeros((np.int((len(topind) - 1)/2),2))
        for i in range(np.int((len(topind) - 1)/2)):
            index = [2*i, 2*i + 1, 2*i + 2]
            neudoft[i,:] = topind[index]
            neuvalt[i,:] = [0, sig]
        
        neudofb = np.zeros((np.int((len(botind) - 1)/2),3), np.int32)
        neuvalb = np.zeros((np.int((len(botind) - 1)/2),2))
        for i in range(np.int((len(botind) - 1)/2)):
            index = [2*i, 2*i + 1, 2*i + 2]
            neudofb[i,:] = botind[index]
            neuvalb[i,:] = [0, -sig]
        
        neudof = np.concatenate((neudoft,neudofb), axis=0)
        neuval = np.concatenate((neuvalt,neuvalb), axis=0)
        return dirdof, dirval, neudof, neuval
class PostProcessing():
    def solver(self, K, F, udof, uval, sdof):
        import scipy.sparse
        import scipy.sparse.linalg
        K[udof,:] = 0
        F[udof,0] = uval  
        for i in range(len(udof)):
            K[udof[i],udof[i]] = 1
        # U = np.linalg.solve(K, F)    
        
        K = scipy.sparse.csr_matrix(K)
        F = scipy.sparse.csr_matrix(F)
        U = scipy.sparse.linalg.spsolve(K, F).reshape(K.shape[0],1)
        
        U = U[:sdof]
        return U
    def energy( self, p6, t6, U, material, qpe):
        CQPE = []
        for i in range(len(qpe)):
            if len(qpe[i]) > 0:
                CQPE.append(qpe[i])
        CQPE = np.unique(CQPE)
        poi3, wei3 = GaussPointRule.gausspoint(4, 'T3')
        poi4, wei4 = GaussPointRule.gausspoint(4, 'Q4')
        ne = t6.shape[0]
        edof = 12
        Ene = 0
        for e in range(ne):
            if len(material) == 3:
                D = material
            else:
                D = material[e]
            X = p6[t6[e,:],0]
            Y = p6[t6[e,:],1]
            if (X[1] - X[0])*(Y[-1] - Y[0]) - (X[-1] - X[0])*(Y[1] - Y[0]) < 0:
                indinv = t6[e,::-1]
                t6[e,1::] = indinv[0:-1:1]
                X = p6[t6[e,:],0]
                Y = p6[t6[e,:],1]
            index = [t6[e,0]*2, t6[e,0]*2+1,
                 t6[e,1]*2, t6[e,1]*2+1,
                 t6[e,2]*2, t6[e,2]*2+1,
                 t6[e,3]*2, t6[e,3]*2+1,
                 t6[e,4]*2, t6[e,4]*2+1,
                 t6[e,5]*2, t6[e,5]*2+1]
            disp = U[index,0]
            if np.sum(e == CQPE) == 0:
                xis = poi3[:,0]
                eta = poi3[:,1]
                N, dNdx, dNdy, detJac = ShapeFunction.T6element(X, Y, xis, eta) # này k có, nó không tạo ôối tơợng
                Ke = np.zeros((edof,edof))
                for i in range(poi3.shape[0]):
                    B = np.zeros((3,edof))
                    B[0,:] = [dNdx[i,0], 0, dNdx[i,1], 0, dNdx[i,2], 0, dNdx[i,3], 0, dNdx[i,4], 0, dNdx[i,5], 0]
                    B[1,:] = [0, dNdy[i,0], 0, dNdy[i,1], 0, dNdy[i,2], 0, dNdy[i,3], 0, dNdy[i,4], 0, dNdy[i,5]]
                    B[2,:] = [dNdy[i,0], dNdx[i,0], dNdy[i,1], dNdx[i,1], dNdy[i,2], dNdx[i,2], dNdy[i,3], dNdx[i,3], dNdy[i,4], dNdx[i,4], dNdy[i,5], dNdx[i,5]]     
                    Ke = np.dot(np.dot(np.transpose(B),D),B)*detJac[i]*wei3[i]  
                    Ene = Ene + 1/2*np.dot(np.dot(np.transpose(disp),Ke),disp)
                
            else:
                Ke = np.zeros((edof,edof))
                xis = poi4[:,0]
                eta = poi4[:,1]
                N, dNdx, dNdy, detJac = ShapeFunction.Q8element(X, Y, xis, eta)
                Ke = np.zeros((edof,edof))
                
                for i in range(poi4.shape[0]):
                    B = np.zeros((3,edof))
                    B[0,:] = [dNdx[i,0] + dNdx[i,6] + dNdx[i,7], 0, dNdx[i,1], 0, dNdx[i,2], 0, dNdx[i,3], 0, dNdx[i,4], 0, dNdx[i,5], 0]
                    B[1,:] = [0, dNdy[i,0] + dNdy[i,6] + dNdy[i,7], 0, dNdy[i,1], 0, dNdy[i,2], 0, dNdy[i,3], 0, dNdy[i,4], 0, dNdy[i,5]]
                    B[2,:] = [dNdy[i,0] + dNdy[i,6] + dNdy[i,7], dNdx[i,0] + dNdx[i,6] + dNdx[i,7], dNdy[i,1], dNdx[i,1], dNdy[i,2], dNdx[i,2], dNdy[i,3], dNdx[i,3], dNdy[i,4], dNdx[i,4], dNdy[i,5], dNdx[i,5]]     
                    Ke = np.dot(np.dot(np.transpose(B),D),B)*detJac[i]*wei4[i]  
                    Ene = Ene + 1/2*np.dot(np.dot(np.transpose(disp),Ke),disp)
        return Ene
        
    def stresses( self, p6, t6, U, material, qpe):
        CQPE = []
        for i in range(len(qpe)):
            if len(qpe[i]) > 0:
                CQPE.append(qpe[i])
        CQPE = np.unique(CQPE)
        snode = np.max(t6[:,[0, 2, 4]]) + 1
        Sigxx = np.zeros((t6.shape[0],snode))
        Sigyy = np.zeros((t6.shape[0],snode))
        Sigxy = np.zeros((t6.shape[0],snode))
        Sigid = np.zeros((t6.shape[0],snode))
        ne = t6.shape[0]
        edof = 12
        for e in range(ne):
            if len(material) == 3:
                D = material
            else:
                D = material[e]
            X = p6[t6[e,:],0]
            Y = p6[t6[e,:],1]
            if (X[1] - X[0])*(Y[-1] - Y[0]) - (X[-1] - X[0])*(Y[1] - Y[0]) < 0:
                indinv = t6[e,::-1]
                t6[e,1::] = indinv[0:-1:1]
                X = p6[t6[e,:],0]
                Y = p6[t6[e,:],1]
            index = [t6[e,0]*2, t6[e,0]*2+1,
                 t6[e,1]*2, t6[e,1]*2+1,
                 t6[e,2]*2, t6[e,2]*2+1,
                 t6[e,3]*2, t6[e,3]*2+1,
                 t6[e,4]*2, t6[e,4]*2+1,
                 t6[e,5]*2, t6[e,5]*2+1]
            disp = U[index,0]
            if np.sum(e == CQPE) == 0:
                xis = np.array([0, 1, 0])
                eta = np.array([0, 0, 1])
                N, dNdx, dNdy, detJac = ShapeFunction.T6element(X, Y, xis, eta)
                Sige = np.zeros((3,3))
                for i in range(3):
                    B = np.zeros((3,edof))
                    B[0,:] = [dNdx[i,0], 0, dNdx[i,1], 0, dNdx[i,2], 0, dNdx[i,3], 0, dNdx[i,4], 0, dNdx[i,5], 0]
                    B[1,:] = [0, dNdy[i,0], 0, dNdy[i,1], 0, dNdy[i,2], 0, dNdy[i,3], 0, dNdy[i,4], 0, dNdy[i,5]]
                    B[2,:] = [dNdy[i,0], dNdx[i,0], dNdy[i,1], dNdx[i,1], dNdy[i,2], dNdx[i,2], dNdy[i,3], dNdx[i,3], dNdy[i,4], dNdx[i,4], dNdy[i,5], dNdx[i,5]]     
                    Sige = np.dot(np.dot(D,B),disp)
                    Sigxx[e,t6[e,2*i]] = Sige[0]
                    Sigyy[e,t6[e,2*i]] = Sige[1]
                    Sigxy[e,t6[e,2*i]] = Sige[2]
                    Sigid[e,t6[e,2*i]] = 1
                
            else:
                xis = np.array([-1, 1, 1])
                eta = np.array([0, -1, 1])
                N, dNdx, dNdy, detJac = ShapeFunction.Q8element(X, Y, xis, eta)
                for i in range(3):
                    if detJac[i] == 0:
                        Sigxx[e,t6[e,2*i]] = 1e20
                        Sigyy[e,t6[e,2*i]] = 1e20
                        Sigxy[e,t6[e,2*i]] = 1e20
                        Sigid[e,t6[e,2*i]] = 1
                    else:
                        B = np.zeros((3,edof))
                        B[0,:] = [dNdx[i,0] + dNdx[i,6] + dNdx[i,7], 0, dNdx[i,1], 0, dNdx[i,2], 0, dNdx[i,3], 0, dNdx[i,4], 0, dNdx[i,5], 0]
                        B[1,:] = [0, dNdy[i,0] + dNdy[i,6] + dNdy[i,7], 0, dNdy[i,1], 0, dNdy[i,2], 0, dNdy[i,3], 0, dNdy[i,4], 0, dNdy[i,5]]
                        B[2,:] = [dNdy[i,0] + dNdy[i,6] + dNdy[i,7], dNdx[i,0] + dNdx[i,6] + dNdx[i,7], dNdy[i,1], dNdx[i,1], dNdy[i,2], dNdx[i,2], dNdy[i,3], dNdx[i,3], dNdy[i,4], dNdx[i,4], dNdy[i,5], dNdx[i,5]]     
                        Sige = np.dot(np.dot(D,B),disp)
                        Sigxx[e,t6[e,2*i]] = Sige[0]
                        Sigyy[e,t6[e,2*i]] = Sige[1]
                        Sigxy[e,t6[e,2*i]] = Sige[2]
                        Sigid[e,t6[e,2*i]] = 1
        X = p6[t6[:,[0, 2, 4]],0]
        Y = p6[t6[:,[0, 2, 4]],1]
        Ae = aux.area(X,Y)
        Ae = Ae.reshape(len(Ae),1)
        Sigxxi = np.dot(np.transpose(Sigxx),Ae)/np.dot(np.transpose(Sigid),Ae)
        Sigyyi = np.dot(np.transpose(Sigyy),Ae)/np.dot(np.transpose(Sigid),Ae)
        Sigxyi = np.dot(np.transpose(Sigxy),Ae)/np.dot(np.transpose(Sigid),Ae)
        
        sigxx = np.zeros(shape = (p6.shape[0]))
        sigxx[t6[:,0]] = Sigxxi[t6[:,0],0]
        sigxx[t6[:,2]] = Sigxxi[t6[:,2],0]
        sigxx[t6[:,4]] = Sigxxi[t6[:,4],0]
        sigxx[t6[:,1]] = (Sigxxi[t6[:,0],0] + Sigxxi[t6[:,2],0])/2
        sigxx[t6[:,3]] = (Sigxxi[t6[:,2],0] + Sigxxi[t6[:,4],0])/2
        sigxx[t6[:,5]] = (Sigxxi[t6[:,4],0] + Sigxxi[t6[:,0],0])/2
        
        sigyy = np.zeros(shape = (p6.shape[0]))
        sigyy[t6[:,0]] = Sigyyi[t6[:,0],0]
        sigyy[t6[:,2]] = Sigyyi[t6[:,2],0]
        sigyy[t6[:,4]] = Sigyyi[t6[:,4],0]
        sigyy[t6[:,1]] = (Sigyyi[t6[:,0],0] + Sigyyi[t6[:,2],0])/2
        sigyy[t6[:,3]] = (Sigyyi[t6[:,2],0] + Sigyyi[t6[:,4],0])/2
        sigyy[t6[:,5]] = (Sigyyi[t6[:,4],0] + Sigyyi[t6[:,0],0])/2
        
        sigxy = np.zeros(shape = (p6.shape[0]))
        sigxy[t6[:,0]] = Sigxyi[t6[:,0],0]
        sigxy[t6[:,2]] = Sigxyi[t6[:,2],0]
        sigxy[t6[:,4]] = Sigxyi[t6[:,4],0]
        sigxy[t6[:,1]] = (Sigxyi[t6[:,0],0] + Sigxyi[t6[:,2],0])/2
        sigxy[t6[:,3]] = (Sigxyi[t6[:,2],0] + Sigxyi[t6[:,4],0])/2
        sigxy[t6[:,5]] = (Sigxyi[t6[:,4],0] + Sigxyi[t6[:,0],0])/2
        
        poi3, wei3 = GaussPointRule.gausspoint(7, 'T3')
        Error = np.zeros((t6.shape[0],1))
        for e in range(t6.shape[0]):
            Sigxxh = Sigxx[e,t6[e,[0, 2, 4]]];Sigxxh = Sigxxh.reshape(len(Sigxxh),1)
            Sigxxr = Sigxxi[t6[e,[0, 2, 4]]]
            
            Sigyyh = Sigyy[e,t6[e,[0, 2, 4]]];Sigyyh = Sigyyh.reshape(len(Sigyyh),1)
            Sigyyr = Sigyyi[t6[e,[0, 2, 4]]]
            
            Sigxyh = Sigxy[e,t6[e,[0, 2, 4]]];Sigxyh = Sigxyh.reshape(len(Sigxyh),1)
            Sigxyr = Sigxyi[t6[e,[0, 2, 4]]]
            
            A = 0
            for i in range(poi3.shape[0]):
                xis = poi3[i,0]
                eta = poi3[i,1]
                N = np.array([1 - xis - eta, xis, eta]); N = N.reshape(1,len(N))
                Sigxxhf = np.dot(N,Sigxxh); Sigyyhf = np.dot(N,Sigyyh); Sigxyhf = np.dot(N,Sigxyh);
                Sigxxrf = np.dot(N,Sigxxr); Sigyyrf = np.dot(N,Sigyyr); Sigxyrf = np.dot(N,Sigxyr);
                ErrSig = np.array(([Sigxxhf - Sigxxrf, Sigyyhf - Sigyyrf, Sigxyhf - Sigxyrf]))
                ErrSig = ErrSig.reshape(len(ErrSig),1)
                A = A + np.dot(np.transpose(ErrSig),ErrSig)*2*Ae[e]*wei3[i]
            Error[e,0] = np.sqrt(A)
            # Error[e,1] = e

        return sigxx,sigyy,sigxy,Error        
    def SIF(self, tipava, p6, t6, numtip, disp, young, poisson, k, qpe ):
        import math
        Gi = np.zeros(numtip)
        ki = []
        keq = np.zeros(numtip)
        craang = np.zeros(numtip)
        for i in range(numtip):
            if tipava[i] == True:
                qpei = qpe[i]
                Ux = disp[0::2]
                Uy = disp[1::2]
                Uxy = np.concatenate((Ux, Uy), axis=1) 
                
                e = t6[qpei[0],2]
                d = t6[qpei[0],1]
                a = t6[qpei[0],0]
                b = t6[qpei[-1],5]
                c = t6[qpei[-1],4]
                
                L1 = np.sqrt((p6[c,0] - p6[a,0])**2 + (p6[c,1] - p6[a,1])**2)
                L2 = np.sqrt((p6[e,0] - p6[a,0])**2 + (p6[e,1] - p6[a,1])**2)
                L = 1/2*(L1 + L2)
                
                B1 = p6[e,:]
                B2 = p6[c,:]
                O = p6[a,:]
                B = 1/2*(B1+B2)
                
                P11 = np.array([1,0])
                P00 = np.array([0,0])
                P22 = O - B
                angle = aux.angle(P11,P00,P22)
                
                x1 = 1; y1 = 0;
                x2 = 0; y2 = 0;
                x3 = O[0] - B[0]; y3 = O[1] - B[1];
                d0 = (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1)
                if abs(angle) > np.finfo(float).eps*1000 and abs(abs(angle) - math.pi) > np.finfo(float).eps*1000:
                    angle0 = 2*math.pi - np.sign(d0)*angle
                else:
                    angle0 = angle
                
                Trans = np.array(([[math.cos(angle0), math.sin(angle0)],[-math.sin(angle0),  math.cos(angle0)]]))
                
                pc = np.dot(Trans,Uxy[c,:])
                pb = np.dot(Trans,Uxy[b,:])
                pe = np.dot(Trans,Uxy[e,:])
                pd = np.dot(Trans,Uxy[d,:])
                K1 = young/3/(1+k)/(1+poisson)*np.sqrt(2*math.pi/L)*(4*(pb[1] - pd[1]) - 0.5*(pc[1] - pe[1]))
                K2 = young/3/(1+k)/(1+poisson)*np.sqrt(2*math.pi/L)*(4*(pb[0] - pd[0]) - 0.5*(pc[0] - pe[0]))
                
                alpha = K1/K2
                ang1 = 2*np.arctan(1/4*(alpha + np.sqrt(alpha**2 + 8)))
                ang2 = 2*np.arctan(1/4*(alpha - np.sqrt(alpha**2 + 8)))
                d2s1 = -K1*(np.cos(ang1/2) + 3*np.cos(3*ang1/2)) + K2*(np.sin(ang1/2) + 9*np.sin(3*ang1/2))
                d2s2 = -K1*(np.cos(ang2/2) + 3*np.cos(3*ang2/2)) + K2*(np.sin(ang2/2) + 9*np.sin(3*ang2/2))
                if d2s1 >= 0 and d2s2 >= 0:
                    thetha = 0
                elif d2s1 < 0:
                    thetha = np.sign(ang1)*min(80*np.pi/180,abs(ang1))
                else:
                    thetha = np.sign(ang2)*min(80*np.pi/180,abs(ang2))
                c11 = 3/4*np.cos(thetha/2) + 1/4*np.cos(3*thetha/2)
                c12 = -3/4*np.sin(thetha/2) - 3/4*np.sin(3*thetha/2)
                c21 = 1/4*np.sin(thetha/2) + 1/4*np.sin(3*thetha/2)
                c22 = 1/4*np.cos(thetha/2) + 3/4*np.cos(3*thetha/2)
                kg1 = c11*K1 + c12*K2
                kg2 = c21*K1 + c22*K2
                Gi[i] = (kg1**2 + kg2**2)/young
                ki.append(np.array(([K1,K2])))
                keq[i] = K1*np.cos(thetha/2)**3 - 3/2*K2*np.cos(thetha/2)*np.sin(thetha)
                craang[i] = thetha
            else:
                Gi[i] = 0
                ki.append(np.array(([0,0])))
                keq[i] = 0
                craang[i] = 0    
        return Gi, ki, keq, craang
class ShapeFunction():
    def T6element(X, Y, xis, eta):
        N = np.zeros((len(xis),6))
        N[:,0] = (1 - xis - eta)*(1 - 2*xis - 2*eta)
        N[:,1] = 4*xis*(1 - xis - eta)
        N[:,2] = xis*(2*xis - 1)
        N[:,3] = 4*xis*eta
        N[:,4] = eta*(2*eta - 1)
        N[:,5] = 4*eta*(1 - xis - eta)
        
        dNdxis = np.zeros((len(xis),6))
        dNdxis[:,0] = 4*eta + 4*xis - 3
        dNdxis[:,1] = 4 - 8*xis - 4*eta
        dNdxis[:,2] = 4*xis - 1
        dNdxis[:,3] = 4*eta
        dNdxis[:,4] = 0*xis
        dNdxis[:,5] = -4*eta
        
        dNdeta = np.zeros((len(xis),6))
        dNdeta[:,0] = 4*eta + 4*xis - 3
        dNdeta[:,1] = -4*xis
        dNdeta[:,2] = 0*xis
        dNdeta[:,3] = 4*xis
        dNdeta[:,4] = 4*eta - 1
        dNdeta[:,5] = 4 - 4*xis - 8*eta
        
        dxdxis = np.dot(dNdxis,X)
        dxdeta = np.dot(dNdeta,X)
        dydxis = np.dot(dNdxis,Y)
        dydeta = np.dot(dNdeta,Y)
        
        detJac = dxdxis*dydeta - dxdeta*dydxis
        dxisdx =  dydeta/detJac
        dxisdy = -dxdeta/detJac
        detadx = -dydxis/detJac
        detady =  dxdxis/detJac
        
        dNdx = np.zeros((len(dxisdx),dNdxis.shape[1]))
        dNdy = np.zeros((len(dxisdx),dNdxis.shape[1]))
        for i in range(len(dxisdx)):
            dNdx[i,:] = dNdxis[i,:]*dxisdx[i] + dNdeta[i,:]*detadx[i]
            dNdy[i,:] = dNdxis[i,:]*dxisdy[i] + dNdeta[i,:]*detady[i]
            
        return N, dNdx, dNdy, detJac
    def Q8element(X, Y, xis, eta):
        corx = np.array(([-1, 0, 1,1,1,0,-1,-1]))
        cory = np.array(([-1,-1,-1,0,1,1, 1, 0]))
        N = np.zeros((len(xis),8))
        dNdxis = np.zeros((len(xis),8))
        dNdeta = np.zeros((len(xis),8))
        for i in range(8):
            xisi = corx[i]
            etai = cory[i]
            N[:,i] = ( (1 + xis*xisi)*(1 + eta*etai) - (1 - xis**2)*(1 + eta*etai) - (1 - eta**2)*(1 + xis*xisi) )*xisi**2*etai**2/4 + (1 - xis**2)*(1 + eta*etai)*(1 - xisi**2)*etai**2/2 + (1 - eta**2)*(1 + xis*xisi)*(1 - etai**2)*xisi**2/2
            dNdxis[:,i] = ( xisi*(1 + eta*etai) + 2*xis*(1 + eta*etai) - xisi*(1 - eta**2) )*xisi**2*etai**2/4 - 2*xis*(1 + eta*etai)*(1 - xisi**2)*etai**2/2 + xisi*(1 - eta**2)*(1 - etai**2)*xisi**2/2
            dNdeta[:,i] = ( etai*(1 + xis*xisi) - etai*(1 - xis**2) + 2*eta*(1 + xis*xisi) )*xisi**2*etai**2/4 + etai*(1 - xis**2)*(1 - xisi**2)*etai**2/2 - 2*eta*(1 + xis*xisi)*(1 - etai**2)*xisi**2/2
        
        dxdxis = (dNdxis[:,0] + dNdxis[:,6] + dNdxis[:,7])*X[0] + dNdxis[:,1]*X[1] + dNdxis[:,2]*X[2] + dNdxis[:,3]*X[3] + dNdxis[:,4]*X[4] + dNdxis[:,5]*X[5] 
        dxdeta = (dNdeta[:,0] + dNdeta[:,6] + dNdeta[:,7])*X[0] + dNdeta[:,1]*X[1] + dNdeta[:,2]*X[2] + dNdeta[:,3]*X[3] + dNdeta[:,4]*X[4] + dNdeta[:,5]*X[5] 
        dydxis = (dNdxis[:,0] + dNdxis[:,6] + dNdxis[:,7])*Y[0] + dNdxis[:,1]*Y[1] + dNdxis[:,2]*Y[2] + dNdxis[:,3]*Y[3] + dNdxis[:,4]*Y[4] + dNdxis[:,5]*Y[5] 
        dydeta = (dNdeta[:,0] + dNdeta[:,6] + dNdeta[:,7])*Y[0] + dNdeta[:,1]*Y[1] + dNdeta[:,2]*Y[2] + dNdeta[:,3]*Y[3] + dNdeta[:,4]*Y[4] + dNdeta[:,5]*Y[5] 
        
        detJac = dxdxis*dydeta - dxdeta*dydxis
        if min(detJac) == 0:
            dNdx = np.zeros((len(dydeta),dNdxis.shape[1]))
            dNdy = np.zeros((len(dydeta),dNdxis.shape[1]))
        else:
            dxisdx =  dydeta/detJac
            dxisdy = -dxdeta/detJac
            detadx = -dydxis/detJac
            detady =  dxdxis/detJac
            dNdx = np.zeros((len(dxisdx),dNdxis.shape[1]))
            dNdy = np.zeros((len(dxisdx),dNdxis.shape[1]))
    
            for i in range(len(dxisdx)):
                dNdx[i,:] = dNdxis[i,:]*dxisdx[i] + dNdeta[i,:]*detadx[i]
                dNdy[i,:] = dNdxis[i,:]*dxisdy[i] + dNdeta[i,:]*detady[i]   
        return N, dNdx, dNdy, detJac 
class GaussPointRule():
    def gausspoint(N, Ele):
        n = np.array([i for i in range(N)]) + 1
        nnk = 2*n + 1
        A = np.zeros(N+1)
        A[0] = 1/3
        A[1:] = 1/(nnk*(nnk + 2))
        n = np.array([i for i in range(1,N)]) + 1
        nnk0 = nnk[n-1]
        nk = n + 1
        # B = np.zeros(len(n))
        nnk2 = nnk0*nnk0
        B = 4*(n*nk)**2/(nnk2*nnk2-nnk2)
        abx = A.reshape(len(A),1)
        aby = np.zeros((len(A),1))
        aby[0,0] = 2
        aby[1,0] = 2/9
        aby[2:,0] = B
        ab = np.concatenate((abx,aby), axis = 1)
        s = np.sqrt(ab[1:N,1])
        
        X, V = linalg.eig(np.diag(ab[:N,0], k=0) + np.diag(s, k=-1) + np.diag(s, k=1))
        X = X.reshape(len(X),1)
        ind = sorted(range(len(X)), key=lambda k: X[k])
        ind = np.array(ind)
        x = (X[ind] + 1)/2
        wx = ab[0,1]*V[0,ind]**2/4
        
        N = N - 1
        N1 = N + 1
        N2 = N + 2
        
        yi = np.array([i for i in range(N,-1,-1)])
        y = np.cos((2*yi+1)*math.pi/(2*N+2))
        
        L = np.zeros((N1,N2))
        y0 = 2
        while (abs(y-y0)).max(0) > np.finfo(float).eps:
            L[:,0] = 1
            L[:,1] = y
            for k in range(2,N1+1):
                L[:,k] = ( (2*k-1)*y*L[:,k-1] - (k-1)*L[:,k-2] )/k
            Lp = N2*( L[:,N1-1] - y*L[:,N2-1] )/(1-y**2)
            y0 = y
            y = y0 - L[:,N2-1]/Lp   
        if Ele == '1D':
            a = -1
            b = 1
            poi = (a*(1-y)+b*(1+y))/2
            wei = (b-a)/((1-y**2)*Lp**2)*(N2/N1)**2
        if Ele == 'T3':
            v = np.array([[0, 0],[0, 1],[1, 0]])
            cd1 = np.array([[ 1, 0, 0],[-1, 0, 1],[0, 1,-1]])
            cd = np.dot(cd1,v)
            t = (1+y)/2
            Wx = abs(np.linalg.det(cd[1:3,:]))*wx
            Wy = 1/((1-y**2)*Lp**2)*(N2/N1)**2
            tt,xx = np.meshgrid(t,x)
            yy = tt*xx
            X = cd[0,0] + cd[1,0]*xx + cd[2,0]*yy
            Y = cd[0,1] + cd[1,1]*xx + cd[2,1]*yy
            
            poi = np.zeros(((N+1)*(N+1),2))
            
            poi[:,0] = np.ravel(np.transpose(X.real))
            poi[:,1] = np.ravel(np.transpose(Y.real))
            Wx = Wx.reshape(1,len(Wx))
            Wy = Wy.reshape(1,len(Wy))
            wei = np.dot(np.transpose(Wx),Wy)
            wei = np.ravel(np.transpose(wei))
        if Ele == 'Q4':    
            a = -1
            b = 1
            y = y.reshape(len(y),1)
            x1d = (a*(1-y)+b*(1+y))/2
            X = np.matlib.repmat(x1d,N + 1, 1)
            Lp = Lp.reshape(len(Lp),1)
            w1d = (b-a)/((1-y**2)*Lp**2)*(N2/N1)**2
            poi = np.zeros(((N+1)*(N+1),2))
            poi[:,0] = X[:,0]
            for i in range(N+1):
                poi[i:(N+1)**2:N+1,0] = x1d[i]
                poi[i*(N+1):i*(N+1) + (N+1),1] = x1d[i]
                wei = np.dot(w1d,np.transpose(w1d))
                wei = np.ravel(np.transpose(wei))
        return poi, wei           
    

    