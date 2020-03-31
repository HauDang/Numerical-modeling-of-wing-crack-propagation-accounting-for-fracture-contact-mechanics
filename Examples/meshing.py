import math
import numpy as np 
import auxiliary
aux = auxiliary.AuxiFu()
def gmshpy(f):
    """ Call a mesh from GMSH file"""
    f.readline() # '$MeshFormat\n'
    f.readline() # '2.2 0 8\n'
    f.readline() # '$EndMeshFormat\n'
    f.readline() # '$Nodes\n'
    n_nodes = int(f.readline()) # '8\n'
    nodes = np.fromfile(f,count=n_nodes*4, sep=" ").reshape((n_nodes,4))
    p = nodes[:,1:3]
    f.readline() # '$EndNodes\n'
    f.readline() # '$Elements\n'
    elems = np.fromfile(f,sep=" ")[:]
    t0 = elems[5::8]
    t1 = elems[6::8]
    t2 = elems[7::8]
    t = np.zeros((len(t0),3),np.int32)
    t[:, 0] = t0-1
    t[:, 1] = t1-1
    t[:, 2] = t2 - 1
    return p, t
def reprocessing(p, t, numcra, example):
    """ Determine these cracks in the mesh"""
    """ NOte: this function depends on the setup of model """
    if example == 'circle crack': 
        dis = np.sqrt(p[:,0]**2 + p[:,1]**2)
        indnod = np.arange(0,p.shape[0])
        indaro = indnod[dis >= max(dis) - np.finfo(float).eps]
        nodaro = p[indaro,:]
        lx = max(p[:,0]) - min(p[:,0])
        R = lx/2
        numtip = numcra*2
        possla = [];  posmas = []; nodcra = []; tipcra = []; moucra = []; roscra = []
        for i in range(numcra):
            nn = 12
            id1 = np.arange(nn+1) + i*nn*2
            id2 = np.arange(nn*2,nn-1,-1) + i*nn*2; id2[0] = i*nn*2
            if nn == 6:
                id3 = np.array(([3, 2, 1, 0, 11, 10, 9]),np.int64) + i*nn*2
                id4 = np.array(([9, 8, 7, 6, 5, 4, 3]),np.int64) + i*nn*2
            if nn == 10:
                id3 = np.array(([5, 4, 3, 2, 1, 0, 19, 18, 17, 16, 15]),np.int64) + i*nn*2
                id4 = np.array(([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]),np.int64) + i*nn*2
            if nn == 12:
                id3 = np.array(([6, 5, 4, 3, 2, 1, 0, 23, 22, 21, 20, 19, 18]),np.int64) + i*nn*2
                id4 = np.array(([18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]),np.int64) + i*nn*2
            if nn == 16:
                id3 = np.array(([8, 7, 6, 5, 4, 3, 2, 1, 0, 31, 30, 29, 28, 27, 26, 25, 24]),np.int64) + i*nn*2
                id4 = np.array(([24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8]),np.int64) + i*nn*2
            
            id5 = np.array(([1, nn*2-1]),np.int64) + i*nn*2
            id6 = np.array(([nn+1, nn-1]),np.int64) + i*nn*2
            id7 = 0 + i*nn*2
            id8 = nn + i*nn*2
            possla.append(p[id1,:])
            posmas.append(p[id2,:])
            nodcra.append(p[id3,:])
            nodcra.append(p[id4,:])
            moucra.append(p[id5,:])
            moucra.append(p[id6,:])
            tipcra.append(p[id7,:])
            tipcra.append(p[id8,:])
            roscra.append([])
            roscra.append([])
        # dc = np.sqrt(sum((p[1,:] - p[11,:])**2))
        dc = np.sqrt(sum((p[1,:] - p[2*nn-1,:])**2))
        iniang = np.zeros((numtip))
        for i in range(numtip):
            tip = tipcra[i]
            pv = nodcra[i]
            p0 = 1/2*(pv[0,:] + pv[-1,:])
            px = np.array(([10, p0[1]]))
            a = np.sqrt(sum((tip - p0)**2))
            b = np.sqrt(sum((px - p0)**2))
            c = np.sqrt(sum((px - tip)**2))
            iniang[i] = np.sign(tip[1] - p0[1])*np.arccos((a**2 + b**2 - c**2)/(2*a*b)) 
        xcoord = np.zeros((t.shape[0],3))
        xcoord[:,0] = p[t[:,0],0]
        xcoord[:,1] = p[t[:,1],0]
        xcoord[:,2] = p[t[:,2],0]
        
        ycoord = np.zeros((t.shape[0],3))
        ycoord[:,0] = p[t[:,0],1]
        ycoord[:,1] = p[t[:,1],1]
        ycoord[:,2] = p[t[:,2],1]
        
        areele = aux.area(xcoord,ycoord)
        aremin = min(areele)
        ladv = np.sqrt( (p[0,0] - p[1,0])**2 + (p[0,1] - p[1,1])**2 )
        # thetha = abs(np.pi/2 - iniang[0])
        # om1 = 1 - 4*np.sin(thetha)**2 + 4*np.sin(thetha)**2 * (1 - np.cos(thetha)**2)*(a/R)**2
        # om2 = (2 + (8*np.cos(thetha)**2 - 5) * (a/R)**2) * np.sin(2*thetha)
        # fy = 1E6
        # Kex = fy*np.sqrt(a)/(np.sqrt(np.pi)*R)*np.array(([om1,om2]))

    elif example == 'multi crack':
        cbl = np.intersect1d(np.where(p[:,0] == min(p[:,0]))[0], np.where(p[:,1] == min(p[:,1]))[0])[0]
        cbr = np.intersect1d(np.where(p[:,0] == max(p[:,0]))[0], np.where(p[:,1] == min(p[:,1]))[0])[0]
        ctl = np.intersect1d(np.where(p[:,0] == min(p[:,0]))[0], np.where(p[:,1] == max(p[:,1]))[0])[0]
        ctr = np.intersect1d(np.where(p[:,0] == max(p[:,0]))[0], np.where(p[:,1] == max(p[:,1]))[0])[0]
        nodaro = p[[cbl, cbr, ctr, ctl, cbl],:]
        numtip = numcra*2
        possla = [];  posmas = []; nodcra = []; tipcra = []; moucra = []; roscra = []
        for i in range(numcra):
            nn = 12
            id1 = np.arange(nn+1) + i*nn*2
            id2 = np.arange(nn*2,nn-1,-1) + i*nn*2; id2[0] = i*nn*2
            if nn == 6:
                id3 = np.array(([3, 2, 1, 0, 11, 10, 9]),np.int64) + i*nn*2
                id4 = np.array(([9, 8, 7, 6, 5, 4, 3]),np.int64) + i*nn*2
            if nn == 10:
                id3 = np.array(([5, 4, 3, 2, 1, 0, 19, 18, 17, 16, 15]),np.int64) + i*nn*2
                id4 = np.array(([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]),np.int64) + i*nn*2
            if nn == 12:
                id3 = np.array(([6, 5, 4, 3, 2, 1, 0, 23, 22, 21, 20, 19, 18]),np.int64) + i*nn*2
                id4 = np.array(([18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]),np.int64) + i*nn*2
            if nn == 16:
                id3 = np.array(([8, 7, 6, 5, 4, 3, 2, 1, 0, 31, 30, 29, 28, 27, 26, 25, 24]),np.int64) + i*nn*2
                id4 = np.array(([24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8]),np.int64) + i*nn*2
            
            id5 = np.array(([1, nn*2-1]),np.int64) + i*nn*2
            id6 = np.array(([nn+1, nn-1]),np.int64) + i*nn*2
            id7 = 0 + i*nn*2
            id8 = nn + i*nn*2
            possla.append(p[id1,:])
            posmas.append(p[id2,:])
            nodcra.append(p[id3,:])
            nodcra.append(p[id4,:])
            moucra.append(p[id5,:])
            moucra.append(p[id6,:])
            tipcra.append(p[id7,:])
            tipcra.append(p[id8,:])
            roscra.append([])
            roscra.append([])
        # dc = np.sqrt(sum((p[1,:] - p[11,:])**2))
        dc = np.sqrt(sum((p[1,:] - p[2*nn-1,:])**2))
        iniang = np.zeros((numtip))
        for i in range(numtip):
            tip = tipcra[i]
            pv = nodcra[i]
            p0 = 1/2*(pv[0,:] + pv[-1,:])
            px = np.array(([10, p0[1]]))
            a = np.sqrt(sum((tip - p0)**2))
            b = np.sqrt(sum((px - p0)**2))
            c = np.sqrt(sum((px - tip)**2))
            iniang[i] = np.sign(tip[1] - p0[1])*np.arccos((a**2 + b**2 - c**2)/(2*a*b)) 
    xcoord = np.zeros((t.shape[0],3))
    xcoord[:,0] = p[t[:,0],0]
    xcoord[:,1] = p[t[:,1],0]
    xcoord[:,2] = p[t[:,2],0]
    
    ycoord = np.zeros((t.shape[0],3))
    ycoord[:,0] = p[t[:,0],1]
    ycoord[:,1] = p[t[:,1],1]
    ycoord[:,2] = p[t[:,2],1]
    
    areele = aux.area(xcoord,ycoord)
    aremin = min(areele)
    ladv = np.sqrt( (p[0,0] - p[1,0])**2 + (p[0,1] - p[1,1])**2 )
    if example == 'circle crack':
        return p, t, nodaro, tipcra, moucra, roscra, nodcra, iniang, dc, aremin, ladv, R
    elif example == 'multi crack':
        return p, t, nodaro, tipcra, moucra, roscra, nodcra, iniang, dc, aremin, ladv, possla, posmas
def rectangular(lx, ly, nx, ny):
    numnod = (nx + 1)*(ny + 1)
    p = np.empty((numnod, 2))
    index = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            p[index] = [i*lx/nx, j*ly/ny]
            index += 1
    t = np.empty((nx*ny*2, 3), np.int32)
    index = 0
    for j in range(ny):
        for i in range(nx):
            pcen = [lx/nx/2 + i*lx/nx, ly/ny/2 + j*ly/ny]
            if (pcen[0] <= lx/2 and pcen[1] <= ly/2) or (pcen[0] >= lx/2 and pcen[1] >= ly/2):
                t[index] = [(nx + 1) * j + i, (nx + 1)*j + i + 1, (j + 1) * (nx + 1) + i]
                t[index + 1] = [(nx + 1) * j + i + 1, (nx + 1) * (j + 1) + i + 1, (j + 1) * (nx + 1) + i]
            else:   
                t[index] = [(nx + 1) * j + i, (j + 1) * (nx + 1) + 1 + i, (j + 1) * (nx + 1) + i]
                t[index + 1] = [(nx + 1) * j + i, (nx + 1) * j + 1 + i, (j + 1) * (nx + 1) + 1 + i]
            index += 2
    nodaro = p[[0, nx, (nx+1)*(ny+1) -1, (nx+1)*ny, 0],:]
    return p, t, nodaro
def circle(R, numele):
    nx = np.int32(np.sqrt(numele))
    ny = np.int32(np.sqrt(numele))
    numnod = (nx + 1)**2
    p = np.empty((numnod, 2))
    index = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            p[index] = [-R + i*2*R/nx, -R + j*2*R/ny]
            index += 1

    t = np.empty((nx*ny*2, 3), np.int32)
    index = 0
    for j in range(ny):
        for i in range(nx):
            pcen = [2*R/nx/2 + i*2*R/nx, 2*R/ny/2 + j*2*R/ny]
            if (pcen[0] <= 2*R/2 and pcen[1] <= 2*R/2) or (pcen[0] >= 2*R/2 and pcen[1] >= 2*R/2):
                t[index] = [(nx + 1) * j + i, (j + 1) * (nx + 1) + 1 + i, (j + 1) * (nx + 1) + i]
                t[index + 1] = [(nx + 1) * j + i, (nx + 1) * j + 1 + i, (j + 1) * (nx + 1) + 1 + i]
            else:
                t[index] = [(nx + 1) * j + i, (nx + 1)*j + i + 1, (j + 1) * (nx + 1) + i]
                t[index + 1] = [(nx + 1) * j + i + 1, (nx + 1) * (j + 1) + i + 1, (j + 1) * (nx + 1) + i]
                
            index += 2
    nodaro = p[[0, nx, (nx+1)*(ny+1) -1, (nx+1)*ny, 0],:]
    
    young = 1
    poisson = 0.49
    material = young/(1-poisson**2)*np.array(([[1,poisson,0],
                                               [poisson,1,0],
                                               [0,0,(1-poisson)/2]]))
    sdof = p.shape[0]*2; edof = 6
    K = np.zeros((sdof,sdof))  
    F = np.zeros((sdof,1))  
    for e in range(t.shape[0]):
        x = p[t[e,:],0]
        y = p[t[e,:],1]
        index = [t[e,0]*2, t[e,0]*2+1,
                 t[e,1]*2, t[e,1]*2+1,
                 t[e,2]*2, t[e,2]*2+1]
        dNdx, dNdy, Ae = T3element(x,y)

        B = np.array(([dNdx[0], 0, dNdx[1], 0, dNdx[2], 0], 
                      [0, dNdy[0], 0, dNdy[1], 0, dNdy[2]], 
                      [dNdy[0], dNdx[0], dNdy[1], dNdx[1], dNdy[2], dNdx[2]]))       
        Ke = np.dot(np.dot(np.transpose(B),material),B)*Ae 
        for j in range(edof):
            for i in range(edof):
                kj = index[j]
                ki = index[i]
                K[ki,kj] = K[ki,kj] + Ke[i,j]
    
    topind = aux.p2index(p,nodaro[[3,2],:],1) 
    botind = aux.p2index(p,nodaro[[0,1],:],1) 
    lefind = aux.p2index(p,nodaro[[0,3],:],1) 
    rigind = aux.p2index(p,nodaro[[1,2],:],1) 
    bc = np.concatenate((botind, rigind[1::], topind[-2::-1], lefind[-2:0:-1]))
    udof = []
    for i in range(len(bc)):
        udof.append(bc[i]*2); udof.append(bc[i]*2 + 1)
    uval = np.zeros(len(bc)*2)
    for i in range(len(bc)):
        phi = i*2*math.pi/(len(bc))
        uval[i*2 + 0] = R*math.cos(5*math.pi/4 + phi)
        uval[i*2 + 1] = R*math.sin(5*math.pi/4 + phi)
    for i in range(len(udof)):
        c = udof[i]
        K[c,:] = 0
        K[c,c] = 1
        F[c,0] = uval[i]   
    disp = np.linalg.solve(K, F)    
    dispx = disp[::2]
    dispy = disp[1::2]
    p = np.concatenate((dispx, dispy), axis = 1)
    nodaro = p[bc,:]
    nodaro = np.concatenate((nodaro,nodaro[0,:].reshape(1,2) ))
    return p, t, nodaro
def hertmesh( R, N, ly, ny, d0 ):
    pc, tc, boucir = circle(R, N)
    pr, tr, bourec = rectangular(2*R, ly, np.int(np.sqrt(N)*2), ny)
    pr[:,0] = pr[:,0] - R; pr[:,1] = pr[:,1] - R - ly - d0
    bourec[:,0] = bourec[:,0] - R; bourec[:,1] = bourec[:,1] - R- ly - d0
    boubel = boucir[np.where(boucir[:,1] <= np.finfo(float).eps*1E5)[0],:]
    horcoo = np.unique(boubel[:,0])
    index = 0
    for j in range(ny + 1):
        for i in range(np.int(np.sqrt(N)*2) + 1):
            pr[index,0] = horcoo[i]
            index += 1
    
    numcir = tc.shape[0]
    numrec = tr.shape[0]
    p = np.concatenate((pc,pr),axis = 0)
    t = np.concatenate((tc,tr + pc.shape[0]),axis = 0)
    
    bouindcir = aux.p2index(p,boucir,0) 
    
    bar = np.concatenate((t[:numcir,[0,1]],t[:numcir,[1,2]],t[:numcir,[0,2]]),axis = 0)
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
    indsla = np.arange(2*np.int(np.sqrt(N))+1) - np.int(np.sqrt(N)/2)+1
    slapoi = p[bounodcir[indsla],:]  
    pospoi = p[aux.p2index(p,bourec[3:1:-1,:],1),:]
    
    return p, t, boucir, bourec, slapoi, pospoi, numcir, numrec
def edgecrack(lx, ly, nx, ny, dc):
    p, t, nodaro = rectangular(lx, ly, nx, ny)
    indexc = np.arange(int((nx+1)*ny/2),int((nx+1)*ny/2 + nx*lx/2/lx + 1))
    eleUper = np.arange(int(nx*2*ny/2),int(nx*2*ny/2+nx*2*lx/2/lx))
    pnew = np.empty((len(indexc)-1, 2))
    pnew[:,0] = p[indexc[:-1],0]
    pnew[:,1] = p[indexc[:-1],1] + dc/2
    
    p[indexc[:-1],1] = p[indexc[:-1],1] - dc/2
    nodesnew = np.arange(0,len(indexc)) + p.shape[0]
    nodesnew[-1] = indexc[-1];
    p = np.concatenate((p,pnew), axis=0)
    pt = pnew[-1,:]
    pb = p[indexc[-2],:]
    moucra = []
    roscra = []
    moucrai = np.array(np.concatenate(([pb.reshape(1,2), pt.reshape(1,2)])))
    moucra.append(moucrai)
    roscra.append([])
    cou1 = 0
    cou2 = 0
    innew = np.empty((len(eleUper), 3), np.int32)
    index = 0
    for e in range(len(eleUper)):
        ind = t[eleUper[e],]
        id0 = aux.ismember(ind,indexc)
        if np.sum(id0) == 1:
            indf = list(set(ind) & set(indexc)) 
            indv = list(set(ind) - set(indf)) 
            innew[index] = [nodesnew[cou1],indv[0],indv[1]]
            cou1 = cou1 + 1
        if np.sum(id0) == 2:   
            indf = list(set(ind) & set(indexc)) 
            indv = list(set(ind) - set(indf)) 
            innew[index] = [nodesnew[e - cou2-1],nodesnew[e - cou2],indv[0]]
            cou2 = cou2 + 1
        index = index + 1    
             
    t = np.delete(t, eleUper, axis=0)
    t = np.concatenate((t,innew), axis=0)
    
    for e in range(t.shape[0]):
        x = p[t[e,:],0]
        y = p[t[e,:],1]
        if (x[1] - x[0])*(y[2] - y[0]) - (y[1] - y[0])*(x[2] - x[0]) < 0:
            t[e,:] = t[e,[0, 2, 1]]
    pvcrb = p[indexc[:-1],:]
    pvcrt = p[nodesnew,:]
    tipcra = [p[indexc[-1],:]]
    nodaro = p[[0, nx, (nx+1)*(ny+1) -1, (nx+1)*ny, 0],:]
    nodmid = p[indexc[-1]:np.int((nx+1)*ny/2+nx+1),:]
    nodcra = [np.concatenate((pvcrb,pvcrt[ ::-1,:]), axis=0)]
    return p, t, nodaro, tipcra, moucra, roscra, nodcra, nodmid

def t3tot6(tipava, p, t, tipcra):
    """ Determine a mesh including triangles of 6 nodes"""
    edge = np.concatenate((t[:,[0,1]],t[:,[0,2]],t[:,[1,2]]), axis=0)
    edge = np.sort(edge, axis = 1)
    edge = np.unique(edge, axis = 0)
    
    facecenx = (p[edge[:,0],0] + p[edge[:,1],0])/2
    faceceny = (p[edge[:,0],1] + p[edge[:,1],1])/2
    facecen = np.concatenate((facecenx.reshape(len(facecenx),1), faceceny.reshape(len(faceceny),1)), axis = 1)
    midnode = [i for i in range(p.shape[0],p.shape[0] + edge.shape[0])]
    midnode = np.array(midnode)
    t6 = np.empty((t.shape[0],6), np.int32)
    for e in range(t.shape[0]):
        edgee1 = np.unique(t[e,[0,1]])
        edgee2 = np.unique(t[e,[1,2]])
        edgee3 = np.unique(t[e,[0,2]])
        id1 = np.intersect1d(np.where(edgee1[0] == edge[:,0])[0], np.where(edgee1[1] == edge[:,1])[0])
        id2 = np.intersect1d(np.where(edgee2[0] == edge[:,0])[0], np.where(edgee2[1] == edge[:,1])[0])
        id3 = np.intersect1d(np.where(edgee3[0] == edge[:,0])[0], np.where(edgee3[1] == edge[:,1])[0])
        t6[e,:] = np.array([t[e,0], midnode[id1][0], t[e,1], midnode[id2][0], t[e,2], midnode[id3][0]],np.int32)
    p6 = np.concatenate((p,facecen), axis = 0)
    qpe = []
    if len(tipcra) > 0:
        for i in range(len(tipcra)):
            tip = tipcra[i]
            qpei = []
            if len(tip) == 1:
                tip = tip[0]
            if tipava[i] == True:
                tipind = np.where(np.sqrt((p6[:,0] - tip[0])**2 + (p6[:,1] - tip[1])**2) < np.finfo(float).eps)[0]
                qpei = np.int32(np.where(t6 == tipind)[0])
                
                p6[t6[qpei,1],0] = 3/4*p6[t6[qpei,0],0] + 1/4*p6[t6[qpei,2],0]
                p6[t6[qpei,1],1] = 3/4*p6[t6[qpei,0],1] + 1/4*p6[t6[qpei,2],1]    
                p6[t6[qpei,5],0] = 3/4*p6[t6[qpei,0],0] + 1/4*p6[t6[qpei,4],0]
                p6[t6[qpei,5],1] = 3/4*p6[t6[qpei,0],1] + 1/4*p6[t6[qpei,4],1]   
            qpe.append(qpei)
        
    return p6, t6, qpe
def T3element(x, y):
    dN1dx = (y[1] - y[2])/(x[2]*y[0] - x[1]*y[0] + x[1]*y[2] - x[2]*y[1] + x[0]*(y[1] - y[2]))
    dN1dy =-(x[1] - x[2])/(x[2]*y[0] - x[1]*y[0] + x[1]*y[2] - x[2]*y[1] + x[0]*(y[1] - y[2]))
    dN2dx = (y[0] - y[2])/(x[0]*y[2] - x[0]*y[1] - x[2]*y[0] + x[2]*y[1] + x[1]*(y[0] - y[2]))
    dN2dy = (x[0] - x[2])/(x[0]*y[1] - x[1]*y[0] - x[0]*y[2] + x[2]*y[0] + x[1]*y[2] - x[2]*y[1])
    dN3dx = (y[0] - y[1])/(x[2]*y[0] - x[1]*y[0] + x[1]*y[2] - x[2]*y[1] + x[0]*(y[1] - y[2]))
    dN3dy =-(x[0] - x[1])/(x[0]*y[1] - x[1]*y[0] - x[0]*y[2] + x[2]*y[0] + x[1]*y[2] - x[2]*y[1])
    l1 = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
    l2 = np.sqrt((x[2] - x[0])**2 + (y[2] - y[0])**2)
    l3 = np.sqrt((x[2] - x[1])**2 + (y[2] - y[1])**2)
    s = (l1+l2+l3)/2
    Ae = np.sqrt(s*(s-l1)*(s-l2)*(s-l3))
    dNdx = [dN1dx, dN2dx, dN3dx]
    dNdy = [dN1dy, dN2dy, dN3dy]
       
    return dNdx, dNdy, Ae 