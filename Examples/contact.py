import numpy as np
import math
import auxiliary
aux = auxiliary.AuxiFu()
import FEM
discre = FEM.Discretization()
solution = FEM.PostProcessing()
def norvec(p,nod):
    # compute tangential and norm vectors for each edge
    if nod.ndim == 1:
        boupoi = p[nod,:]
        tanedg = boupoi[1::] - boupoi[0:-1]
        lenedg = np.sqrt(np.sum(tanedg**2,axis = 1))
        noredg = np.copy(tanedg)
        noredg[:,0] = tanedg[:,1]/lenedg; noredg[:,1] = -tanedg[:,0]/lenedg
        # compute norm vector for each point
        bounor = np.zeros(shape = (len(nod),2))
        bounor[ 0,:] = noredg[ 0,:]
        bounor[-1,:] = noredg[-1,:]
        
        bounor[1:-1:1,0] = (noredg[0:-1,0]*lenedg[0:-1] + noredg[1::,0]*lenedg[1::])/(lenedg[0:-1] + lenedg[1::])
        bounor[1:-1:1,1] = (noredg[0:-1,1]*lenedg[0:-1] + noredg[1::,1]*lenedg[1::])/(lenedg[0:-1] + lenedg[1::])
    elif nod.ndim == 2:
        slapoi = p[nod[:,0],:]
        maspoi = p[nod[:,1],:]
        veccon = maspoi - slapoi
        lenvec = np.sqrt(np.sum(veccon**2,axis = 1))
        bounor = np.copy(veccon)
        bounor[:,0] = veccon[:,0]/lenvec; bounor[:,1] = veccon[:,1]/lenvec
    return bounor
def activeset(p6, t6, qpe, disp, gap, possla, posmas, consla, conmas, material ):
    connod = np.concatenate((consla, conmas))
    bounod = np.concatenate((possla, posmas))
    dispx = disp[::2]
    dispy = disp[1::2]
    
    sigxx,sigyy,sigxy,error = solution.stresses(p6, t6, disp, material, qpe)
    sigxx[np.setdiff1d(bounod,connod)] = 0
    sigyy[np.setdiff1d(bounod,connod)] = 0
    sigxy[np.setdiff1d(bounod,connod)] = 0
    
    bounor = norvec(p6,np.concatenate((possla.reshape(len(possla),1), posmas.reshape(len(posmas),1)), axis = 1))
    # bounor = norvec(p6,posnodc)
    trax = (sigxx[possla]*bounor[:,0] + 0.5*sigxy[possla]*bounor[:,1])*bounor[:,0]
    tray = (0.5*sigxy[possla]*bounor[:,0] + sigyy[possla]*bounor[:,1])*bounor[:,1]
    traction = trax + tray    
    nordis = (dispx[possla,0] - dispx[posmas,0])*bounor[:,0] + (dispy[possla,0] - dispy[posmas,0])*bounor[:,1]
    if len(gap) == 1:
        consla = possla[traction + material[2,2]*(-nordis - gap) < 0]
    else:
        consla = possla[traction + material[2,2]*(-nordis - gap[possla]) < 0]
    mylist = aux.ismember(possla, consla)
    idecon = np.where(np.array(mylist) == 1)[0]
    conmas = posmas[idecon]
    return consla, conmas, traction
def contactsolver(p6, t6, material, qpe, K, F, dirdof, dirval, slapoi, maspoi, gap, dc):
    if len(material) == 3:
        D = material
    else:
        D = material[0]
    if dc != 0:
        possla = []; posmas = []
        for craind in range(len(slapoi)):
            posnodsla = aux.p2index(p6,slapoi[craind],1)
            posnodmas = aux.p2index(p6,maspoi[craind],1)       
            indsla = []; indmas = []
            for i in range(1,len(posnodsla)-1):
                xs, ys = p6[posnodsla[i],:]
                for j in range(1,len(posnodmas)-1):
                    xm, ym = p6[posnodmas[j],:]
                    d = np.sqrt((xs - xm)**2 + (ys - ym)**2)
                    if d <= 1.1*dc:
                        indsla.append(i)
                        indmas.append(j)
                        
            possla = np.concatenate((possla, posnodsla[indsla]))
            posmas = np.concatenate((posmas, posnodmas[indmas]))  
        
        possla = np.int32(possla)
        posmas = np.int32(posmas)
    else:
        possla = []; posmas = []
        for craind in range(len(slapoi)):
            posnodsla = aux.p2index(p6,slapoi[craind],1)
            posnodmas = aux.p2index(p6,maspoi[craind],1)       
            possla = np.concatenate((possla, posnodsla))
            posmas = np.concatenate((posmas, posnodmas))   
        possla = np.int32(possla)
        posmas = np.int32(posmas)
    conang = np.zeros(shape = p6.shape[0])
    a = p6[possla,0]-p6[posmas,0]
    b = p6[possla,1]-p6[posmas,1]
    indneg = np.where(a < 0)[0]
    indzer = np.where(a == 0)[0]
    indpos = np.where(a > 0)[0]
    indappzer = np.where(abs(a) < np.finfo(float).eps*1E3)[0]
    conang[possla[indzer]] = 0
    conang[possla[indneg]] = np.arctan(abs(a[indneg]/b[indneg]))
    conang[possla[indpos]] = -np.arctan(abs(a[indpos]/b[indpos]))
    for i in range(len(indappzer)):
        if b[indappzer[i]] > 0:
            conang[possla[indappzer]] = 0
        elif b[indappzer[i]] < 0:
            conang[possla[indappzer]] = math.pi
    conang[posmas] = conang[possla]
    
    consla = []
    conmas = []
    connod = np.concatenate((consla, conmas))
    U = np.zeros(shape = (p6.shape[0],2))
    err = 0;
    conver = 0
    cou = 0
    # import plot
    # import matplotlib.pyplot as plt
    # fig, grid = plt.subplots()
    while conver == 0 and cou < 20:
        cou += 1
        U0 = np.copy(U)
        KR = np.copy(K)
        for i in range(len(connod)):
            k = connod[i]*2
            ai = conang[connod[i]]
            if k == 0:
                KR[k+2::,k]   =  K[k+2::,k]*math.cos(ai) + K[k+2::,k+1]*math.sin(ai)
                KR[k,k+2::]   =  K[k+2::,k]*math.cos(ai) + K[k+2::,k+1]*math.sin(ai)
                KR[k+2::,k+1] = -K[k+2::,k]*math.sin(ai) + K[k+2::,k+1]*math.cos(ai)
                KR[k+1,k+2::] = -K[k+2::,k]*math.sin(ai) + K[k+2::,k+1]*math.cos(ai)
            elif k == K.shape[0]-1-1:
                KR[0:k,k]   =  K[0:k,k]*math.cos(ai) + K[0:k,k+1]*math.sin(ai)
                KR[k,0:k]   =  K[0:k,k]*math.cos(ai) + K[0:k,k+1]*math.sin(ai)
                KR[0:k,k+1] = -K[0:k,k]*math.sin(ai) + K[0:k,k+1]*math.cos(ai)
                KR[k+1,0:k] = -K[0:k,k]*math.sin(ai) + K[0:k,k+1]*math.cos(ai)
            else:
                KR[0:k,k]   =  K[0:k,k]*math.cos(ai) + K[0:k,k+1]*math.sin(ai)
                KR[k,0:k]   =  K[0:k,k]*math.cos(ai) + K[0:k,k+1]*math.sin(ai)
                KR[0:k,k+1] = -K[0:k,k]*math.sin(ai) + K[0:k,k+1]*math.cos(ai)
                KR[k+1,0:k] = -K[0:k,k]*math.sin(ai) + K[0:k,k+1]*math.cos(ai)
                KR[k+2::,k]   =  K[k+2::,k]*math.cos(ai) + K[k+2::,k+1]*math.sin(ai)
                KR[k,k+2::]   =  K[k+2::,k]*math.cos(ai) + K[k+2::,k+1]*math.sin(ai)
                KR[k+2::,k+1] = -K[k+2::,k]*math.sin(ai) + K[k+2::,k+1]*math.cos(ai)
                KR[k+1,k+2::] = -K[k+2::,k]*math.sin(ai) + K[k+2::,k+1]*math.cos(ai)
            KR[k, k+1] = math.sin(ai)*math.cos(ai)*(K[k+1, k+1] - K[k, k]) + math.cos(2*ai)*K[k, k+1]
            KR[k+1, k] = math.sin(ai)*math.cos(ai)*(K[k+1, k+1] - K[k, k]) + math.cos(2*ai)*K[k, k+1]
            KR[k, k]     = math.cos(ai)**2*K[k, k] + math.sin(2*ai)*K[k, k+1] + math.sin(ai)**2*K[k+1, k+1]
            KR[k+1, k+1] = math.sin(ai)**2*K[k, k] - math.sin(2*ai)*K[k, k+1] + math.cos(ai)**2*K[k+1, k+1]     
        Kc = np.zeros(shape = (len(consla), K.shape[0]))
        Fc = np.zeros(shape = (len(consla), 1))
        k22 = np.zeros(shape = (len(consla), len(consla)))
        for i in range(len(consla)):
            Kc[i,consla[i]*2 + 1] = 1 
            Kc[i,conmas[i]*2 + 1] = -1
            if len(gap) == 1:
                Fc[i,0] = 0
            else:
                Fc[i,0] = gap[consla[i]]    
            
        K1 = np.concatenate((KR,np.transpose(Kc)), axis = 1)
        K2 = np.concatenate((Kc, k22), axis = 1)
        KF = np.concatenate((K1, K2), axis = 0)
        FF = np.concatenate((F, Fc), axis = 0)
        disp = solution.solver(KF,FF,dirdof,dirval,p6.shape[0]*2)
        dispnew = np.copy(disp)
        for i in range(len(connod)):
            k = connod[i]*2
            ai = conang[connod[i]]
            dispnew[k] = math.cos(ai)*disp[k] - math.sin(ai)*disp[k+1]
            dispnew[k+1] = math.sin(ai)*disp[k] + math.cos(ai)*disp[k+1]
        disp = np.copy(dispnew)
        consla, conmas, traction = activeset(p6, t6, qpe, disp, gap, possla, posmas, consla, conmas, D )
        connod = np.concatenate((consla, conmas))
        U = np.concatenate((disp[::2], disp[1::2]), axis = 1)
        # pd = p6 + U
        # plot.trisurf2d(fig, grid, pd, t6, eleind = 0, nodind = 0, line = [], point = pd[np.concatenate((consla,conmas)),:])
        # plt.show()
        # plt.pause(1)
        
        err = sum(np.sqrt(np.sum((U0 - U)**2, axis = 1)))/sum(np.sqrt(np.sum(U**2, axis = 1)))*100
        # print('error of active set method', err, '%')
        if err < 10: # relative error (%)
            conver = 1     
    return disp, consla, conmas, traction