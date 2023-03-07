import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import NearestNDInterpolator
import nibabel as nib
from hippo_spin_testing import fill_nans

#From the hippunfold_toolbox by Jordan DeKraker (https://github.com/jordandekraker/hippunfold_toolbox)


def density_interp(indensity, outdensity, cdata, method='nearest'):
    '''interpolates data from one surface density onto another via unfolded space
    Inputs:
      indensity: one of '0p5mm', '1mm', '2mm', or 'unfoldiso
      outdensity: one of '0p5mm', '1mm', '2mm', or 'unfoldiso
      cdata: data to be interpolated (same number of vertices, N, as indensity)
      method: 'nearest', 'linear', or 'cubic'. 
    Outputs: 
      interp: interpolated data'''
    
    VALID_STATUS = {'0p5mm', '1mm', '2mm', 'unfoldiso'}
    if indensity not in VALID_STATUS:
        raise ValueError("results: indensity must be one of %r." % VALID_STATUS)
    if outdensity not in VALID_STATUS:
        raise ValueError("results: outdensity must be one of %r." % VALID_STATUS)
    
    # load unfolded surfaces for topological matching
    startsurf = nib.load(f'resources/tpl-avg_space-unfold_den-{indensity}_midthickness.surf.gii')
    vertices_start = startsurf.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
    targetsurf = nib.load(f'resources/tpl-avg_space-unfold_den-{outdensity}_midthickness.surf.gii')
    vertices_target = targetsurf.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
    faces = targetsurf.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data

    # interpolate
    interp = griddata(vertices_start[:,:2], values=cdata, xi=vertices_target[:,:2], method=method)
    # fill any NaNs
    interp = fill_nans.fillnanvertices(faces,interp)
    return interp,vertices_target