from hippo_spin_testing import interpolate_densities
import numpy as np
import nibabel as nib
import scipy.io as spio
from scipy.ndimage import rotate
from scipy.ndimage import shift
import warnings

def spin_test(imgfix,imgperm,nperm,metric='pearson'):

    '''Permutation testing of unfolded hippocampus maps
    Inputs:
      imgfix: path to the fixed map
      imgperm: path to the map which wll be permuted
      nperm: Number of permutations to perform
      metric: Metric for comparing maps (one of pearson, spearman, adjusted rand, adjusted mutual info)
    Outputs: 
      metricnull: null distribution of metric
      permutedimg: All permuted spatial maps
      r_obs: The observed association between the two aligned maps
      pval: p-value based on metricnull r_obs'''


    fixedimg = nib.load(imgfix)
    fixedimgdata = fixedimg.agg_data()
    permimg = nib.load(imgperm)
    permimgdata = permimg.agg_data()

    fixedimgvertnum = np.max(fixedimgdata.shape)
    permimgvertnum = np.max(permimgdata.shape)

    if fixedimgvertnum != permimgvertnum:
        warnings.warn("Warning fixed and permuted map not the same size. Program will continue to interpolation")

    vertexnumber = [7262,2004,419] #corresponds to 0p5mm, 1mm, and 2mm respectively
    surfacespacing = ['0p5mm', '1mm', '2mm']

    if fixedimgvertnum not in vertexnumber or permimgvertnum not in vertexnumber:
        raise ValueError(f"Surface number of vertices must be one of {vertexnumber}.")
    else:
        fixind = vertexnumber.index(fixedimgvertnum)
        permind = vertexnumber.index(permimgvertnum)
        imgfixinterp = interpolate_densities.density_interp(surfacespacing[fixind], 'unfoldiso', fixedimgdata, method='nearest')[0]
        imgfixinterp = np.reshape(imgfixinterp,(126,254))#get maps to 124x256
        imgperminterp = interpolate_densities.density_interp(surfacespacing[permind], 'unfoldiso', permimgdata, method='nearest')[0]
        imgperminterp = np.reshape(imgperminterp,(126,254))#get maps to 124x256


    rotation = np.random.randint(1,360,nperm) #generate random rotations
    translate1 = np.random.randint(-63,64,nperm) #generate random translations
    translate2 = np.random.randint(-127,128,nperm)
    
    imgsize = imgfixinterp.shape
    permutedimg = np.empty((imgsize[0],imgsize[1],nperm))
    metricnull = np.empty((nperm))

    for ii in range(nperm):
        rotimg = rotate(imgperminterp, rotation[ii], axes=(1, 0), reshape=False, output=None, order=3, mode='wrap', cval=0.0, prefilter=True)
        transrotimg = shift(rotimg, [translate1[ii],translate2[ii]], output=None, order=3, mode='wrap', cval=0.0, prefilter=True)
        permutedimg[:,:,ii] = transrotimg
        
    if metric=='pearson':
        from scipy.stats import pearsonr
        r_obs = pearsonr(imgfixinterp.flatten(),imgperminterp.flatten())[0]
        for ii in range(nperm):
            imgpermflat = permutedimg[:,:,ii].flatten()
            metricnull[ii] = pearsonr(imgfixinterp.flatten(),imgpermflat)[0]
    elif metric=='spearman':
            from scipy.stats import spearmanr
            r_obs = spearmanr(imgfixinterp.flatten(),imgperminterp.flatten())[0]
            for ii in range(nperm):
                imgpermflat = permutedimg[:,:,ii].flatten()
                metricnull[ii] = spearmanr(imgfixinterp.flatten(),imgpermflat)[0]
    elif metric=='adjusted rand':
        from sklearn.metrics import adjusted_rand_score
        r_obs = adjusted_rand_score(imgfixinterp.flatten(),imgperminterp.flatten())[0]
        for ii in range(nperm):
            imgpermflat = permutedimg[:,:,ii].flatten()
            metricnull[ii] = adjusted_rand_score(imgfixinterp.flatten(),imgpermflat)[0]
    elif metric=='adjusted mutual info':
        from sklearn.metrics import adjusted_mutual_info_score
        r_obs = adjusted_mutual_info(imgfixinterp.flatten(),imgperminterp.flatten())[0]
        for ii in range(nperm):
            imgpermflat = permutedimg[:,:,ii].flatten()
            metricnull[ii] = (imgfixinterp.flatten(),imgpermflat)           

    pval = np.mean(np.abs(metricnull) >= np.abs(r_obs))
                
    return metricnull,permutedimg,pval,r_obs
