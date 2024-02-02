from hippo_spin_testing import interpolate_densities
import numpy as np
import nibabel as nib
import scipy.io as spio
from scipy.ndimage import rotate
from scipy.ndimage import shift
import warnings

def spin_test(imgfix,imgperm,nperm,metric='pearson',space='orig'):

    '''Permutation testing of unfolded hippocampus maps
    Inputs:
      imgfix: Fixed map (path or loaded in data)
      imgperm: Map which will be permuted (path or loaded in data)
      nperm: Number of permutations to perform
      metric: Metric for comparing maps (one of pearson, spearman, adjusted rand, or adjusted mutual info)
      space: Space the correlation will be performed in. If 'orig' will perform 
             the correlation at the original density. If 'unfoldiso' will perform
             the correlation at the isotropic density which is the density used 
             for permutations.
    Outputs: 
      metricnull: null distribution of metric
      permutedimg: All permuted spatial maps at 'unfoldiso' density
      r_obs: The observed association between the two aligned maps
      pval: p-value based on metricnull r_obs'''
    if type(imgfix) == str:
        fixedimg = nib.load(imgfix)
        fixedimgdata = fixedimg.agg_data()
    else:
        fixedimgdata = imgfix
    if type(imgperm) == str:
        permimg = nib.load(imgperm)
        permimgdata = permimg.agg_data()
    else:
        permimgdata = imgperm
    fixedimgvertnum = np.max(fixedimgdata.shape) #number of vertices
    permimgvertnum = np.max(permimgdata.shape)

    vertexnumber = [32004,7262,2004,419] #corresponds to unfoldiso, 0p5mm, 1mm, and 2mm respectively
    surfacespacing = ['unfoldiso','0p5mm', '1mm', '2mm']

    if fixedimgvertnum not in vertexnumber or permimgvertnum not in vertexnumber:
        raise ValueError(f"Surface number of vertices must be one of {vertexnumber}.")
    else:
        permind = vertexnumber.index(permimgvertnum)
        fixind = vertexnumber.index(fixedimgvertnum) #find the surface spacing which corresponds to the vertex number of that map
        if surfacespacing[permind] == 'unfoldiso': #already at unfold iso, dont interpolate
            imgperminterp = permimgdata
            imgperminterp = np.reshape(imgperminterp,(126,254))#get maps to 126x254
        else:
            imgperminterp = interpolate_densities.density_interp(surfacespacing[permind], 'unfoldiso', permimgdata, method='nearest')[0]
            imgperminterp = np.reshape(imgperminterp,(126,254))#get maps to 126x254
        if space=='unfoldiso': #if unfoldiso, then need to interpolate fixed image to unfoldiso for correlation
            imgfixobs = interpolate_densities.density_interp(surfacespacing[fixind], 'unfoldiso', fixedimgdata, method='nearest')[0] #interpolate to unfoldiso density
            imgpermobs = imgperminterp.flatten()
            permutedimg = np.empty((126*254,nperm))
        elif space=='orig': #if orig, then correlations performed at original density, need to check that 
            if fixedimgvertnum != permimgvertnum: #if two maps not the same size, need to interpolate for comparison
                imgfixobs = interpolate_densities.density_interp(surfacespacing[fixind], surfacespacing[permind], fixedimgdata, method='nearest')[0] #interpolate to unfoldiso density
            else:
                imgfixobs = fixedimgdata
            imgpermobs = permimgdata
            permutedimg = np.empty((permimgvertnum,nperm))

    rotation = np.random.randint(1,360,nperm) #generate random rotations
    translate1 = np.random.randint(-63,64,nperm) #generate random translations
    translate2 = np.random.randint(-127,128,nperm)

    imgsize = imgperminterp.shape
    permutedimgiso = np.empty((imgsize[0],imgsize[1],nperm))
    metricnull = np.empty((nperm))

    for ii in range(nperm):
        rotimg = rotate(imgperminterp, rotation[ii], axes=(1, 0), reshape=False, output=None, order=3, mode='wrap', cval=0.0, prefilter=True) #rotate image
        transrotimg = shift(rotimg, [translate1[ii],translate2[ii]], output=None, order=3, mode='wrap', cval=0.0, prefilter=True) #translate image
        permutedimgiso[:,:,ii] = transrotimg #this is our permuted image at unfoldiso density
        if space=='orig': #resample permuted maps back to original density for correlation
            permutedimg[:,ii] = interpolate_densities.density_interp('unfoldiso',surfacespacing[permind], permutedimgiso[:,:,ii].flatten(), method='nearest')[0]
        elif space=='unfoldiso': #permuted map can remain in unfoldiso for correlation
            permutedimg[:,ii] = permutedimgiso[:,:,ii].flatten()

    if metric=='pearson':
        from scipy.stats import pearsonr
        r_obs = pearsonr(imgfixobs,imgpermobs)[0] #observed correspondance when maps are anatomically aligned
        for ii in range(nperm):
            imgpermflat = permutedimg[:,ii]
            metricnull[ii] = pearsonr(imgfixobs,imgpermflat)[0] #null distribution of correspondance between permuted and fixed map
    elif metric=='spearman':
            from scipy.stats import spearmanr
            r_obs = spearmanr(imgfixobs,imgpermobs)[0]
            for ii in range(nperm):
                imgpermflat = permutedimg[:,ii]
                metricnull[ii] = spearmanr(imgfixobs,imgpermflat)[0]
    elif metric=='adjusted rand':
        from sklearn.metrics import adjusted_rand_score
        r_obs = adjusted_rand_score(imgfixobs,imgpermobs)[0]
        for ii in range(nperm):
            imgpermflat = permutedimg[:,ii]
            metricnull[ii] = adjusted_rand_score(imgfixobs,imgpermflat)[0]
    elif metric=='adjusted mutual info':
        from sklearn.metrics import adjusted_mutual_info_score
        r_obs = adjusted_mutual_info(imgfixobs,imgpermobs)[0]
        for ii in range(nperm):
            imgpermflat = permutedimg[:,ii]
            metricnull[ii] = (imgfixobs,imgpermflat)           

    pval = np.mean(np.abs(metricnull) >= np.abs(r_obs)) #p-value is the sum of all instances where null correspondance is >= observed correspondance / nperm
                
    return metricnull,permutedimgiso,pval,r_obs
