import numpy as np 
import numpy.matlib
import auxiliary
aux = auxiliary.AuxiFu()
def NNinterpolation( points, values, p, t):
    ''' Natural neighbor interpolation '''
    snode = np.max(t) + 1   
    Sigid = np.zeros((t.shape[0],snode))
    Sigxx = np.zeros((t.shape[0],snode))
    ne = t.shape[0]         
        
    for e in range(ne):
        Sigxx[e,t[e,:]] = values[e]
        Sigid[e,t[e,:]] = 1
               
    X = p[t,0]
    Y = p[t,1]
    Ae = aux.polygon_area(X,Y)
    Ae = Ae.reshape(len(Ae),1)
    valint = np.dot(np.transpose(Sigxx),Ae)/np.dot(np.transpose(Sigid),Ae) # values at interpolation points
    return valint