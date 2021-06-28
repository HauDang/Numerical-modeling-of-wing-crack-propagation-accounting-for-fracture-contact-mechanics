from IPython import get_ipython
get_ipython().magic('reset -sf') 
import time
import numpy as np
from typing import Dict
import scipy.sparse as sps
import porepy as pp
import mixedmode_fracture_analysis as analysis
class ModelSetup():
    def __init__(self):
        self.length_scale = 1
        nn = 4
        self.mesh_size = 0.005*self.length_scale*nn
        self.mesh_args = { "mesh_size_frac": self.mesh_size, "mesh_size_min": 1 * self.mesh_size, "mesh_size_bound": 1/(-2.5*nn + 15)*self.length_scale} 

        self.box = {"xmin": 0, "ymin": 0, "xmax": 1*self.length_scale, "ymax": 1*self.length_scale}  
        xcen =  (self.box['xmin'] + self.box['xmax'])/2
        ycen =  (self.box['ymin'] + self.box['ymax'])/2
        lenfra = 0.1*self.length_scale
        self.length_initial_fracture = lenfra
        
        phi = np.pi/4
        fracture1 = np.array([[xcen - lenfra/2*np.cos(phi), ycen - lenfra/2*np.sin(phi)],
                              [xcen + lenfra/2*np.cos(phi), ycen + lenfra/2*np.sin(phi)]]) 


        self.fracture = np.array([fracture1])

        self.initial_fracture = self.fracture.copy()
        self.GAP = 1.0e-3*self.length_scale
 
        self.tips, self.frac_pts, self.frac_edges = analysis.fracture_infor(self.fracture)
    def set_rock(self):
        self.YOUNG = 40e9
        self.POISSON = 0.2
        self.SH = 15E6
        self.Sh = -7E6*0
        self.material = dict([('YOUNG', self.YOUNG), ('POISSON', self.POISSON), ('KIC', 0.4e6) ]) 
    def create_grid(self):
        """ Define a fracture network and domain and create a GridBucket.
        """
        # Domain definition
        network = pp.FractureNetwork2d(self.frac_pts.T, self.frac_edges.T, domain=self.box)
        gb = network.mesh(self.mesh_args)       
        pp.contact_conditions.set_projections(gb)

        self.gb = gb
        self.Nd = self.gb.dim_max()
        self._Nd = self.gb.dim_max()
        g2d = self.gb.grids_of_dimension(2)[0]
        self.min_face = np.copy(self.mesh_size) #np.min(g2d.face_areas)
        self.min_cell = np.min(g2d.cell_volumes)
        self.p, self.t = analysis.adjustmesh(g2d, self.tips, self.GAP)
        self.fa_no =  g2d.face_nodes.indices.reshape((2, g2d.num_faces), order='f').T 
        return gb
    def boundary_condition(self, p):
        tol = np.copy(self.GAP)
        cbl = np.intersect1d(np.where(p[:,0] < np.min(p[:,0]) + tol)[0], 
                             np.where(p[:,1] < np.min(p[:,1]) + tol)[0])[0]
        cbr = np.intersect1d(np.where(p[:,0] > np.max(p[:,0]) - tol)[0], 
                             np.where(p[:,1] < np.min(p[:,1]) + tol)[0])[0]
        ctl = np.intersect1d(np.where(p[:,0] < np.min(p[:,0]) + tol)[0], 
                             np.where(p[:,1] > np.max(p[:,1]) - tol)[0])[0]
        ctr = np.intersect1d(np.where(p[:,0] > np.max(p[:,0]) - tol)[0], 
                             np.where(p[:,1] > np.max(p[:,1]) - tol)[0])[0]
        nodaro = p[[cbl, cbr, ctr, ctl, cbl],:]
        
        topind = analysis.p2index(p,nodaro[[3,2],:]) 
        botind = analysis.p2index(p,nodaro[[0,1],:])  
        lefind = analysis.p2index(p,nodaro[[0,3],:]) 
        rigind = analysis.p2index(p,nodaro[[1,2],:])  
        
        # dof_dir = np.concatenate((botind*2, botind*2 + 1,
        #                           lefind*2, lefind*2 + 1,
        #                           topind*2+1, rigind*2), axis = 0)
        # val_dir = np.concatenate((botind*0, botind*0,
        #                           lefind*0, lefind*0,
        #                           topind*0 - 0.001, rigind*0 - 0.002), axis = 0)
        
        dof_dir = np.concatenate((botind*2 + 1,
                                  lefind*2), axis = 0)
        val_dir = np.concatenate((botind*0,
                                  lefind*0), axis = 0)
        
        # dof_neu = []
        # val_neu = []
        
        
        dof_neu = np.empty(shape = (len(topind) + len(rigind), 2), dtype = np.int32)
        val_neu = np.zeros(shape = (len(topind) + len(rigind), 2))
        
        dof_neu[0:len(topind),0] = topind*2; dof_neu[0:len(topind),1] = topind*2 + 1
        dof_neu[len(topind)::,0] = rigind*2; dof_neu[len(topind)::,1] = rigind*2 + 1
        
        val_neu[0:len(topind),1] = self.Sh
        val_neu[len(topind)::,0] = self.SH
            
        self.bc_type = dict([('dir', dof_dir), ('neu', dof_neu) ]) 
        self.bc_value = dict([('dir', val_dir), ('neu', val_neu) ]) 
  
        
    def prepare_simulation(self):        
        self.create_grid()
        self.set_rock()
    def evaluate_propagation(self):
        tips, frac_pts, frac_edges = analysis.fracture_infor(self.fracture)
        
            
        if len(self.fracture) == 2:
            pt1 = self.fracture[0][0]; pt2 = self.fracture[0][-1]
            segments = self.fracture[-1]
            intpoi = analysis.intersectLines( pt1, pt2, segments )
            if len(intpoi) > 0:
                dis = np.sqrt((tips[:,0] - intpoi[0,0])**2 + (tips[:,1] - intpoi[0,1])**2)
                index = np.where(dis <= np.finfo(float).eps*1E5)[0]
                tips_actualy = np.delete(tips, index, axis = 0)
            else:
                tips_actualy = tips
        else:
            tips_actualy = tips
            
        pref, tref = analysis.refinement( self.p, self.t, self.p, self.t, self.fracture, tips_actualy, self.min_cell, self.min_face, self.GAP)

        pref, tref, fn, cf, iniang = analysis.do_remesh(pref, tref, self.min_face, self.fracture, self.GAP)
        
        p6, t6, qpe = analysis.t3tot6(pref, tref, tips)
        self.boundary_condition(p6)
        
        disp = analysis.FEM_solution(self.material, p6, t6, qpe, self.bc_type, self.bc_value)
        
        disp2 = disp.reshape((p6.shape[0],2))

        
        Gi, ki, keq, craang = analysis.SIF(p6, t6, disp, self.material['YOUNG'],  self.material['POISSON'], qpe )
        print(keq)
        ladv = np.zeros(tips.shape[0])
        # ladv[:] =  max_pro
        pos_pro = []
        if np.max(np.abs(keq)) >= self.material['KIC']:
            pos_pro = np.where(np.abs(keq)*1.1 >= self.material['KIC'])[0]
        
        if len(pos_pro) > 0:
            ladv[pos_pro] = self.min_face*( Gi[pos_pro]/np.max(Gi[pos_pro]) )**0.35
        newfrac = []
        for i in range(tips.shape[0]):
            if ladv[i] > self.min_face*0.6:
                tipi = tips[i]
                tipnew = tipi + ladv[i]*np.array([np.cos(craang[i] + iniang[i]), np.sin(craang[i] + iniang[i])])
                newfrac.append( np.array([tipi, tipnew]) )
        
    
        tips0 = np.copy(tips)
        if len(newfrac) > 0:
            for i in range(len(newfrac)):
                tipold = newfrac[i][0,:]
                index = np.where(np.sqrt((tips0[:,0] - tipold[0])**2 + (tips0[:,1] - tipold[1])**2) < TOL)[0]
                tips0[index,:] = newfrac[i][1,:]
        return keq, ki, newfrac, tips0, p6, t6, disp
    def split_face(self, newfrac = None):
        if newfrac is not None:
            dis = []
            for i in range(len(newfrac)):
                tipinew = newfrac[i][1,:]
                disi = np.min(np.array([np.abs(tipinew[0] - self.box['xmin']),
                                        np.abs(tipinew[0] - self.box['xmax']),
                                        np.abs(tipinew[1] - self.box['ymin']),
                                        np.abs(tipinew[1] - self.box['ymax'])]))
                dis.append(disi)
            if np.min(dis) > self.min_face*5:  
                p, t, fn, cf, iniang = analysis.do_remesh(self.p, self.t, self.min_face, self.fracture, self.GAP, newfrac = newfrac)   
                tip_prop = np.empty(shape=(len(newfrac),2))
                new_tip = np.empty(shape=(len(newfrac),2))
                for i in range(len(newfrac)):
                    # dis = np.sqrt( (p[:,0] - newfrac[i][0,0])**2 + (p[:,1] - newfrac[i][0,1])**2 )
                    # ind0 = np.argmin(dis)
                    dis = np.sqrt( (p[:,0] - newfrac[i][1,0])**2 + (p[:,1] - newfrac[i][1,1])**2 )
                    ind1 = np.argmin(dis)
                    p[ind1,:] = newfrac[i][1,:]
                    tip_prop[i,:] = newfrac[i][0,:]
                    new_tip[i,:] = newfrac[i][1,:]
                self.p, self.t = analysis.splitelement( p, t, self.fracture, newfrac, self.GAP )    
                frac_aft = []            
                for j, fracturej in enumerate(self.fracture):
                    for k, tip_propi in enumerate(tip_prop):
                        if np.sum((fracturej[0,:] - tip_propi)**2) < TOL:
                            fracturej = np.concatenate(( new_tip[k,:].reshape(1,2), fracturej ), axis = 0)
                        if np.sum((fracturej[-1,:] - tip_propi)**2) < TOL:
                            fracturej = np.concatenate(( fracturej, new_tip[k,:].reshape(1,2) ), axis = 0)
            
                    frac_aft.append(fracturej)   
                self.fracture = frac_aft
    def propagation_process(self):
        keq, ki, newfrac, tips0, p6, t6, disp = self.evaluate_propagation()
        self.split_face(newfrac)
                
TOL = np.finfo(float).eps*1E8
setup = ModelSetup() 
setup.prepare_simulation()  
for ii in range(20):
    setup.propagation_process()
# keq, ki, newfrac, tips0, p6, t6, disp = setup.evaluate_propagation()
# disp2 = disp.reshape((p6.shape[0],2))
# analysis.trisurf( p6 + disp2*5e0, t6, point = tips0)




analysis.trisurf( setup.p, setup.t, point = setup.fracture[0], infor = None)





