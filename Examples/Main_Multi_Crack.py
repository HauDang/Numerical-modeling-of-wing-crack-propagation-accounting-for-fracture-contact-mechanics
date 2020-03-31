from IPython import get_ipython
get_ipython().magic('reset -sf') 

import numpy as np
import numpy.matlib
import meshing
import Grid
import auxiliary
import FEM
import contact
import plot
import matplotlib.pyplot as plt

aux = auxiliary.AuxiFu()
discre = FEM.Discretization()
boucon = FEM.SetupModel()
solution = FEM.PostProcessing()
thick = 1
young = 15*10**9
poisson = 0.21
material = young/(1-poisson**2)*np.array(([[1,poisson,0],
                                    [poisson,1,0],
                                    [0,0,(1-poisson)/2]]))*thick
k = (3 - poisson)/(1 + poisson)
KIC = 1.5*1E6
tor = 2E6
tor0 = 0.1*5E6
p, t = meshing.gmshpy(open('Mesh_multi_cracks.msh'))

p, t, nodaro, tipcra, moucra, roscra, nodcra, iniang, dc, aremin, lmin, slapoi, maspoi = meshing.reprocessing(p, t, 5, 'multi crack')
lmin = lmin*1.2
aremin = lmin**2*np.sqrt(3)/4
    
fig, grid = plt.subplots()
plot.trisurf2d(fig, grid, p, t, eleind = 0, nodind = 0, line = nodcra, point = tipcra, parame = [])
plt.show()
plt.pause(0.0001)

numtip = len(tipcra)
gloang = np.zeros((numtip))
crapro = np.zeros((numtip)) == 0
tipava = np.zeros((numtip)) == 0
error = np.zeros((t.shape[0]))
proste = 0
remesh = 1
dsi = np.zeros((numtip)) + lmin*4

rm = Grid.ReMesh( p, t, tipcra, moucra, roscra, nodcra, nodaro, dc )
data = []
while proste < 35:
    print('# Pro.Tip =',crapro, '# Pro.Step =',proste )
    print('  Tip.Ava =',tipava )
    if remesh == 1:
        p, t = rm.refinement( tipava, crapro, error, aremin, lmin, 'based on error' )
        p, t, moucra, roscra = rm.quaterelement( tipava, crapro, lmin )
        print('done remesh')
        p6, t6, qpe = meshing.t3tot6(tipava, p, t, tipcra)
        lefhs = discre.stiffness( p6, t6, material, qpe) 
    dirdof, dirval, neudof, neuval = boucon.shearslip (p6, t6, nodaro, tor)
    righs = discre.loadsegment( p6, neudof, neuval)
    gap = np.zeros(shape = p6.shape[0])
    disp, consla, conmas, traction = contact.contactsolver(p6, t6, material, qpe, lefhs, righs, dirdof, dirval, slapoi, maspoi, gap, dc)

    Gi, ki, keq, craang = solution.SIF(tipava, p6, t6, numtip, disp, young, poisson, k, qpe )
    if max(abs(keq)) >= KIC:
        p, t, tipcra, nodcra, moucra, gloang, tipava, crapro = rm.adjustmesh( tipava, KIC, Gi, keq, lmin, iniang, gloang, craang )
        proste = proste + 1; remesh = 1    
        plot.trisurf2d(fig, grid, p, t, eleind = 0, nodind = 0, line = nodcra, point = p6[consla,:], parame = [tor])
        plt.show()
        plt.pause(0.0001)
        connod = np.array(([consla, conmas]))
        data.append(np.array(([np.copy(p)],[np.copy(t)],[np.copy(nodcra)],[np.copy(tipcra)],[np.copy(proste)],[np.copy(tor)],connod,[np.copy(p6)])))
        Sigxxi,Sigyyi,Sigxyi,error = solution.stresses(p6, t6, disp, material, qpe)
    else:
        remesh = 0
        tor = tor + tor0



    
    
    
    
    
    
    
    




