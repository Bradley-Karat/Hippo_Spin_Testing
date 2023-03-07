import numpy as np
import copy
#From the hippunfold_toolbox by Jordan DeKraker (https://github.com/jordandekraker/hippunfold_toolbox)


def fillnanvertices(F,V):
    '''Fills NaNs by iterative nearest neighbors.'''
    Vnew = copy.deepcopy(V)
    while np.isnan(np.sum(Vnew)):
        # index of vertices containing nan
        vrows = np.unique(np.where(np.isnan(Vnew))[0])
        # replace with the nanmean of neighbouring vertices
        for n in vrows:
            frows = np.where(F == n)[0]
            neighbours = np.unique(F[frows,:])
            Vnew[n] = np.nanmean(Vnew[neighbours], 0)
    return Vnew