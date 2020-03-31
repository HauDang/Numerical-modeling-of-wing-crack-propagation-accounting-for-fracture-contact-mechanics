from IPython import get_ipython
get_ipython().magic('reset -sf') 

import numpy as np
import numpy.matlib
import meshing
import Grid
import auxiliary
import FEM
import plot
import matplotlib.pyplot as plt

aux = auxiliary.AuxiFu()
discre = FEM.Discretization()
boucon = FEM.SetupModel()
solution = FEM.PostProcessing()
thick = 25E-3
young = 15*10**9
poisson = 0.21
material = young/(1-poisson**2)*np.array(([[1,poisson,0],
                                    [poisson,1,0],
                                    [0,0,(1-poisson)/2]]))*thick
k = (3 - poisson)/(1 + poisson)
KIC = 2*1E6
fy = 1E6
fy0 = 0.01*fy
p, t = meshing.gmshpy(open('Mesh_circle_45.msh'))
p, t, nodaro, tipcra, moucra, roscra, nodcra, iniang, dc, aremin, lmin, R = meshing.reprocessing(p, t, 1,'circle crack')
lmin = lmin*1.5
aremin = lmin**2*np.sqrt(3)/4

numtip = len(tipcra)
gloang = np.zeros((numtip))
crapro = np.zeros((numtip)) == 0
tipava = np.zeros((numtip)) == 0
error = np.zeros((t.shape[0]))
proste = 0
remesh = 1

rm = Grid.ReMesh( p, t, tipcra, moucra, roscra, nodcra, nodaro, dc )
data = []
fig, grid = plt.subplots() 
while proste < 35:
    print('# Pro.Tip =',crapro, '# Pro.Step =',proste )
    print('  Tip.Ava =',tipava )
    if remesh == 1:
        p, t = rm.refinement( tipava, crapro, error, aremin, lmin, 'based on error' )
        p, t, moucra, roscra = rm.quaterelement( tipava, crapro, lmin )
        print('done remesh')
        p6, t6, qpe = meshing.t3tot6(tipava, p, t, tipcra)
        lefhs = discre.stiffness( p6, t6, material, qpe) 
    dirdof, dirval, neudof, neuval = boucon.circle( p6, t6, fy )
    righs = discre.loadpoint( p6, t6, thick, neudof, neuval )
    disp = solution.solver(lefhs,righs,dirdof,dirval, p6.shape[0]*2)
    Gi, ki, keq, craang = solution.SIF(tipava, p6, t6, numtip, disp, young, poisson, k, qpe )
    if max(abs(keq)) >= KIC:
        p, t, tipcra, nodcra, moucra, gloang, tipava, crapro = rm.adjustmesh( tipava, KIC, Gi, keq, lmin, iniang, gloang, craang )
        proste = proste + 1; remesh = 1    
        plot.trisurf2d(fig, grid, p, t, eleind = 0, nodind = 0, line = nodcra, point = tipcra, parame = [fy])
        plt.show()
        plt.pause(0.0001)
        data.append(np.array(([np.copy(p)],[np.copy(t)],[np.copy(nodcra)],[np.copy(tipcra)],[np.copy(proste)],[np.copy(fy)])))
        Sigxxi,Sigyyi,Sigxyi,error = solution.stresses(p6, t6, disp, material, qpe)
    else:
        remesh = 0
        fy = fy + fy0
   
fig, grid = plt.subplots()
plot.trisurf2d(fig, grid, p, t, eleind = 1, nodind = 1, line = nodcra, point = tipcra, parame = [fy])
plt.show()

