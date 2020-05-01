import numpy as np 
import math
class AuxiFu():
    def area(self, X, Y):
        if len(X.shape) == 2:
            l1 = np.sqrt((X[:,1] - X[:,0])**2 + (Y[:,1] - Y[:,0])**2)
            l2 = np.sqrt((X[:,2] - X[:,0])**2 + (Y[:,2] - Y[:,0])**2)
            l3 = np.sqrt((X[:,1] - X[:,2])**2 + (Y[:,1] - Y[:,2])**2)
        if len(X.shape) == 1:
            l1 = np.sqrt((X[1] - X[0])**2 + (Y[1] - Y[0])**2)
            l2 = np.sqrt((X[2] - X[0])**2 + (Y[2] - Y[0])**2)
            l3 = np.sqrt((X[1] - X[2])**2 + (Y[1] - Y[2])**2)
        s = (l1+l2+l3)/2
        areatri = np.sqrt(s*(s-l1)*(s-l2)*(s-l3))     
        return areatri
    def polygon_area(self, x, y): 
        '''https://stackoverrun.com/vi/q/6706068'''
        correction = x[:,-1] * y[:,0] - y[:,-1]* x[:,0]
        main_area = np.sum(x[:,:-1] * y[:,1:], axis = 1) - np.sum(y[:,:-1] * x[:,1:], axis = 1)
        return 0.5*np.abs(main_area + correction)
    def ismember(self, A, B):
        return [ np.sum(a == B) for a in A ]
    def norm(self, n):
        if len(n.shape) == 1:
            nor = np.sqrt(n[0]**2 + n[1]**2)
        elif len(n.shape) == 2 and n.shape[0] == 1:
            nor = np.sqrt(n[0,0]**2 + n[0,1]**2)
        else:
            nor = np.sqrt(n[:,0]**2 + n[:,1]**2)
        return nor
    def p2segment(self, p, pv):
        # To find the distance of our point p  to the line segment between points A and B,
        # we need the closest point on the line segment. We can represent such a point q on the segment by:
        # q = A + t(B - A)
        # => t = (Ap.AB)/AB^2
        # if t > 1 => q = B
        # if t < 0 => q = A
        # else => q locates between A and B
        # distance = pq
        if len(p.shape) == 1:
            p = p.reshape(1,2)
        d = np.empty((p.shape[0], pv.shape[0]-1))
        ds = np.empty((p.shape[0], 1))
        for i in range(pv.shape[0]-1):
            A0 = pv[i,0] * np.ones(p.shape[0])
            A1 = pv[i,1] * np.ones(p.shape[0])
            B0 = pv[i+1,0] * np.ones(p.shape[0])
            B1 = pv[i+1,1] * np.ones(p.shape[0])
            q = np.empty((p.shape[0], 2))
            VecAB = pv[i+1,:] - pv[i,:]
            DotAB = VecAB[0]**2 + VecAB[1]**2
            if DotAB == 0:
                q[:,0] = A0
                q[:,1] = A1
            else:
                Ap = np.empty((p.shape[0], 2))
                Ap[:,0] = p[:,0] - A0
                Ap[:,1] = p[:,1] - A1
                t = (Ap[:,0]*VecAB[0] + Ap[:,1]*VecAB[1])/DotAB
                id1 = t < 0 
                id2 = t > 1 
                id3 = np.logical_and(t <= 1.0, t >= 0.0) 
                q[id1,0] = A0[id1]
                q[id1,1] = A1[id1]
                q[id2,0] = B0[id2]
                q[id2,1] = B1[id2]
                q[id3,0] = A0[id3] + t[id3] * VecAB[0]
                q[id3,1] = A1[id3] + t[id3] * VecAB[1]
            d[:,i] = np.sqrt((p[:,0] - q[:,0])**2 + (p[:,1]- q[:,1])**2)
        ds[:,0] = d.min(1)
        return ds
    def p2index(self, p, pv, sort):
        if len(pv) == 1:
            pv = pv[0]
        if len(pv.shape) == 1:
            id1 = np.where(np.isclose(p[:,0],pv[0]))[0]
            id2 = np.where(np.isclose(p[:,1],pv[1]))[0]
            bcs = np.intersect1d(id2,id1)
        elif len(pv.shape) == 2 and pv.shape[0] == 1:
            id1 = np.where(np.isclose(p[:,0],pv[0,0]))[0]
            id2 = np.where(np.isclose(p[:,1],pv[0,1]))[0]
             
            bcs = np.intersect1d(id2,id1)
        else:
            ds = self.p2segment(p, pv)
            bc = ds <= np.finfo(float).eps*1E3
            bcInd = np.where(bc)[0]
            if sort == 1:
                dis = np.sqrt((p[bcInd,0] -  pv[0,0])**2 + (p[bcInd,1] -  pv[0,1])**2)
                ind = sorted(range(len(dis)), key=lambda k: dis[k])
                bcs = bcInd[ind]
            else:
                bcs = bcInd
        return bcs
    def angle(self, p1,p0,p2):
        p20 = p2 - p0
        p10 = p1 - p0
        l20 = np.sqrt(sum(p20**2))
        l10 = np.sqrt(sum(p10**2))
       
        if l10*l20 == 0:
            angle = 0
        else:
            n1 = p20/l20
            n2 = p10/l10
            angle = np.arctan2(abs(n1[0]*n2[1] - n1[1]*n2[0]),n1[0]*n2[0] + n1[1]*n2[1])
        return angle
    def inpolygon(self, p, pv):
        ds = self.p2segment(p, pv)
        onboun = np.where(ds == 0)[0]
        linex = p[:,0]
        liney = p[:,1]
        polyx = pv[:,0]
        polyy = pv[:,1]
            
        """Simple method to detect points on the interior or exterior of a closed 
        polygon.  Returns a boolean for single points, or an array of booleans for a 
        line masking the segment(s) of the line within the polygon.
        For each point, operates via a ray-casting approach -- the function projects 
        a semi-infinite ray parallel to the positive horizontal axis, and counts how 
        many edges of the polygon this ray intersects.  For a simply-connected 
        polygon, this determines whether the point is inside (even number of crossings) 
        or outside (odd number of crossings) the polygon, by the Jordan Curve Theorem.
        """
        """Calculate whether given points are within a 2D simply-connected polygon.
        Returns a boolean 
        ARGS:
            polyx: array-like.
                Array of x-coordinates of the vertices of a polygon.
            polyy: array-like.
                Array of y-coordinates of the vertices of a polygon.  Must match 
                dimension of polyx.
            linex: array-like or float.
                x-coordinate(s) of test point(s).
            liney: array-like or float.
                y-coordiante(s) of test point(s).  Must match dimension of linex.
        RETURNS:
            mask: boolean or array of booleans.
                For each (linex,liney) point, True if point is in the polygon, 
                else False.
        """
        # check type, dimensions of polyx,polyy
        # try:
        #     # check that polyx, polyy are iterable
        #     iter(polyx)
        #     iter(polyy)
        # except TypeError:
        #     raise TypeError("polyx, polyy must be iterable")
        # if len != len(polyy):
        #     raise ValueError("polyx, poly must be of same size")
        # if len(polyx) < 3:
        #     raise ValueError("polygon must consist of at least three points")
    
        # handler for single-value vs. array versions for linex, liney
        single_val = True
        try:
            iter(linex)
        except TypeError:
            linex = np.asarray([linex],dtype=float)
        else:
            linex = np.asarray(linex,dtype=float)
            single_val = False
    
        try:
            iter(liney)
        except TypeError:
            liney = np.asarray([liney],dtype=float)
        else:
            liney = np.asarray(liney,dtype=float)
            single_val = False
    
        if linex.shape != liney.shape:
            raise ValueError("linex, liney must be of same shape")
        
        # generator for points in polygon
        def lines():
            p0x = polyx[-1]
            p0y = polyy[-1]
            p0 = (p0x,p0y)
            for i,x in enumerate(polyx):
                y = polyy[i]
                p1 = (x,y)
                yield p0,p1
                p0 = p1
    
        mask = np.array([False for i in range(len(linex))])
        for i,x in enumerate(linex):
            y = liney[i]
            result = False
    
            for p0,p1 in lines():
                if ((p0[1] > y) != (p1[1] > y)) and (x < ((p1[0]-p0[0])*(y-p0[1])/(p1[1]-p0[1]) + p0[0])):
                    result = not result 
            mask[i] = result
    
        # recast mask -- single Boolean if single_val inputs, else return array of booleans
        if single_val:
            mask = mask[0]
        mask[onboun] = True
    
        return mask
    # checks if line segment p1p2 and p3p4 intersect
    def intersect(self,p1, p2, p3, p4):
        d1 = self.direction(p3, p4, p1)
        d2 = self.direction(p3, p4, p2)
        d3 = self.direction(p1, p2, p3)
        d4 = self.direction(p1, p2, p4)
    
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
            ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
    
        elif d1 == 0 and self.on_segment(p3, p4, p1):
            return True
        elif d2 == 0 and self.on_segment(p3, p4, p2):
            return True
        elif d3 == 0 and self.on_segment(p1, p2, p3):
            return True
        elif d4 == 0 and self.on_segment(p1, p2, p4):
            return True
        else:
            return False
    # checks if p lies on the segment p1p2
    def on_segment(self, p1, p2, p):
        return min(p1.x, p2.x) <= p.x <= max(p1.x, p2.x) and min(p1.y, p2.y) <= p.y <= max(p1.y, p2.y)
    def projection(self,A,B,M):
        dis1 = np.sqrt(sum((M - A)**2))
        dis2 = np.sqrt(sum((M - B)**2))
        if dis1 < np.finfo(float).eps*1E3:
            N = A; flag = 1; dis = 0
        elif dis2 < np.finfo(float).eps*1E3:
            N = B; flag = 1; dis = 0
        else:
            AB = np.sqrt(sum((B - A)**2))
            tanvec = (B - A)/AB
            a1 = -tanvec[1]
            b1 = tanvec[0]
            c1 = -a1*A[0] - b1*A[1]
            
            a2 = tanvec[0]
            b2 = tanvec[1]
            c2 = -a2*M[0] - b2*M[1]
            if (a1 == 0 and b1 == 0) or (a2 == 0 and b2 == 0):
                print('something wrong in intersection. please check')
                xn = A[0]
                yn = A[1]
            elif a1 == 0 and b2 == 0:
                xn = -c2/a2
                yn = -c1/b1
            elif b1 == 0 and a2 == 0:
                xn = -c1/a1
                yn = -c2/b2
            elif a1 == 0:
                xn = (-c2 + b2*c1/b1)/a2
                yn = -c1/b1
            elif b1 == 0:
                xn = -c1/a1
                yn = (-c2 + a2*c1/a1)/b2
            elif a2 == 0:
                xn = (-c1 + b1*c2/b2)/a1
                yn = -c2/b2
            elif b2 == 0:
                xn = -c2/a2
                yn = (-c1 + a1*c2/a2)/b1
            else:
                yn = -(c1/a1 - c2/a2)/(b1/a1 - b2/a2)
                xn = -(c1/b1 - c2/b2)/(a1/b1 - a2/b2)
            N = np.array(([xn, yn]))
            dis = np.sqrt(sum((M - N)**2))
            if dis < np.finfo(float).eps*1E3:
                if abs(np.sqrt(sum((A - N)**2)) + np.sqrt(sum((B - N)**2)) - AB) < np.finfo(float).eps*1E3:
                    flag = 2 # N belong to AB
                else:
                    flag = 0 # N dose not belong to AB
            else:
                dir1 = np.sign(N - A)
                dir2 = np.sign(N - B)
                if (dir1[0] == 0 and dir1[1] == 0) or (dir2[0] == 0 and dir2[1] == 0):
                    flag = 1 # N == A or N == B
                elif dir1[0] == -dir2[0] and dir1[1] == -dir2[1]:
                    flag = 2 # N belong to AB
                else:
                    flag = 0
        return N, flag, dis
            
    def intersection(self,A,B,C,D):
        # 1 = in point; 2 = between
        dis1 = np.sqrt(sum((C - A)**2))
        dis2 = np.sqrt(sum((C - B)**2))
        dis3 = np.sqrt(sum((D - A)**2))
        dis4 = np.sqrt(sum((D - B)**2))
        if dis1 < np.finfo(float).eps*1E3 or dis2 < np.finfo(float).eps*1E3:
            N = C; flag = 1
        elif dis3 < np.finfo(float).eps*1E3 or dis4 < np.finfo(float).eps*1E3:
            N = D; flag = 1
        else:
            AB = np.sqrt(sum((B - A)**2))
            tanvec = (B - A)/AB
            a1 = -tanvec[1]
            b1 = tanvec[0]
            c1 = -a1*A[0] - b1*A[1]
            
            DC = np.sqrt(sum((D - C)**2))
            tanvec = (D - C)/DC
            a2 = -tanvec[1]
            b2 = tanvec[0]
            c2 = -a2*C[0] - b2*C[1]
            
            if (a1 == 0 and b1 == 0) or (a2 == 0 and b2 == 0):
                print('something wrong in intersection. please check')
                xn = A[0]
                yn = A[1]
            elif a1 == 0 and b2 == 0:
                xn = -c2/a2
                yn = -c1/b1
            elif b1 == 0 and a2 == 0:
                xn = -c1/a1
                yn = -c2/b2
            elif a1 == 0:
                xn = (-c2 + b2*c1/b1)/a2
                yn = -c1/b1
            elif b1 == 0:
                xn = -c1/a1
                yn = (-c2 + a2*c1/a1)/b2
            elif a2 == 0:
                xn = (-c1 + b1*c2/b2)/a1
                yn = -c2/b2
            elif b2 == 0:
                xn = -c2/a2
                yn = (-c1 + a1*c2/a2)/b1
            else:
                yn = -(c1/a1 - c2/a2)/(b1/a1 - b2/a2)
                xn = -(c1/b1 - c2/b2)/(a1/b1 - a2/b2)
            N = np.array(([xn, yn]))
            
            dis1 = np.sqrt(sum((C - N)**2))
            dis2 = np.sqrt(sum((D - N)**2))
            if dis1 < np.finfo(float).eps*1E3 or dis2 < np.finfo(float).eps*1E3:
                if abs(np.sqrt(sum((A - N)**2)) + np.sqrt(sum((B - N)**2)) - AB) < np.finfo(float).eps*1E3:
                    flag = 2 # N belong to AB
                else:
                    flag = 0 # N dose not belong to AB
            else:
                dir1 = np.sign(N - A)
                dir2 = np.sign(N - B)
                if (dir1[0] == 0 and dir1[1] == 0) or (dir2[0] == 0 and dir2[1] == 0):
                    flag = 1 # N == A or N == B
                elif dir1[0] == -dir2[0] and dir1[1] == -dir2[1]:
                    flag = 2 # N belong to AB
                else:
                    flag = 0
   
        return N, flag
                
                