def trisurf2d(fig, grid, p, t, eleind, nodind, line, point, parame):
    grid.clear()
    if len(parame) == 3:
        grid.set_title('Energy = {}, SIF = {}, MAX.Error = {}'.format(round(parame[0],4), round(parame[1],4), round(parame[2],2)))
    elif len(parame) == 1:
        grid.set_title('Load(MPa) = {}'.format(round(parame[0]/1E6,4)))
    nnod = t.shape[1]
    if nnod == 3:
        X = [p[t[:,0],0], p[t[:,1],0], p[t[:,2],0], p[t[:,0],0]]
        Y = [p[t[:,0],1], p[t[:,1],1], p[t[:,2],1], p[t[:,0],1]]
        grid.plot(X, Y, 'b-')
    elif nnod == 6:
        X = [p[t[:,0],0], p[t[:,1],0], p[t[:,2],0], p[t[:,3],0], p[t[:,4],0], p[t[:,5],0], p[t[:,0],0]]
        Y = [p[t[:,0],1], p[t[:,1],1], p[t[:,2],1], p[t[:,3],1], p[t[:,4],1], p[t[:,5],1], p[t[:,0],1]]
        grid.plot(X, Y, 'b-')
    if eleind == 1:
        if nnod == 3:
            cenx = (p[t[:,0],0] + p[t[:,1],0] + p[t[:,2],0])/3
            ceny = (p[t[:,0],1] + p[t[:,1],1] + p[t[:,2],1])/3
        elif nnod == 6:
            cenx = (p[t[:,0],0] + p[t[:,2],0] + p[t[:,4],0])/3
            ceny = (p[t[:,0],1] + p[t[:,2],1] + p[t[:,4],1])/3
        for i in range(t.shape[0]):
            grid.annotate(str(i), (cenx[i], ceny[i]), (cenx[i], ceny[i]), color='blue')
    if nodind == 1:
        for j in range(p.shape[0]):
            grid.annotate(str(j), (p[j,0], p[j,1]), (p[j,0], p[j,1]), color='red')
        
    if len(line) != 0:
        for i in range(len(line)):
            linei = line[i]
            if len(linei) == 1:
                linei = linei[0]
            grid.plot(linei[:,0], linei[:,1],'r-') 
    if len(point) != 0:
        for i in range(len(point)):
            pi = point[i]
            if len(pi) > 0:
                grid.plot(pi[0], pi[1],'ko')
                
def trisurf3d(fig, grid, p, value, t):
    import matplotlib.tri as mtri
    name_color_map = 'seismic';
    ax = fig.gca(projection='3d');
    x = p[:,0]
    y = p[:,1]
    z = value[:,0]
    triang = mtri.Triangulation(x, y, t);
    ax.plot_trisurf(triang, z, cmap = name_color_map, shade=True, linewidth=0.2)