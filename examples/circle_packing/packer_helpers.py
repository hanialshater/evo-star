import numpy as np
import math
from typing import Tuple, Dict, Any
# from .packer_constants import FLOAT_PRECISION_TOLERANCE # If you move it there

# For Colab, it's easier if constants are passed or globally available from the main script
# So, these helpers will take FLOAT_PRECISION_TOLERANCE as an arg if needed, or assume it's global from main.

def helper_validate_packing(centers: np.ndarray, radii: np.ndarray, n_circles_arg: int, float_tol: float) -> Tuple[bool, Dict[str, Any]]:
    validation_details = {'out_of_bounds_count': 0, 'overlap_count': 0, 'error_message': None, 'shape_ok': True}
    if not isinstance(centers,np.ndarray) or centers.ndim!=2 or centers.shape[0]!=n_circles_arg or centers.shape[1]!=2:
        validation_details.update({'error_message':f"Centers shape invalid", 'shape_ok':False}); return False, validation_details
    if not isinstance(radii,np.ndarray) or radii.ndim!=1 or radii.shape[0]!=n_circles_arg:
        validation_details.update({'error_message':f"Radii shape invalid", 'shape_ok':False}); return False, validation_details
    for i in range(n_circles_arg):
        x,y,r = centers[i,0], centers[i,1], radii[i]
        if r < -float_tol: validation_details['out_of_bounds_count']+=1; validation_details['error_message']=(validation_details.get('error_message','')or"")+f"C{i} neg R."
        if not (x-r>=-float_tol and x+r<=1+float_tol and y-r>=-float_tol and y+r<=1+float_tol): validation_details['out_of_bounds_count']+=1
    if validation_details.get('error_message') and validation_details['out_of_bounds_count']>0: return False, validation_details
    for i in range(n_circles_arg):
        for j in range(i+1, n_circles_arg):
            dist_sq = np.sum((centers[i]-centers[j])**2)
            sum_r_val = radii[i]+radii[j]
            if dist_sq < (sum_r_val**2) - (float_tol * abs(sum_r_val) + float_tol**2):
                if math.sqrt(max(0,dist_sq)) < sum_r_val - float_tol: validation_details['overlap_count']+=1
    is_valid = (validation_details['out_of_bounds_count']==0 and validation_details['overlap_count']==0 and validation_details['shape_ok'])
    return is_valid, validation_details

def helper_compute_max_radii(centers: np.ndarray, n_circles_arg: int, float_tol: float) -> np.ndarray:
    if not isinstance(centers,np.ndarray) or centers.shape!=(n_circles_arg,2): return np.zeros(n_circles_arg)
    radii = np.array([min(c[0],1-c[0],c[1],1-c[1]) for c in centers])
    radii = np.maximum(radii,0)
    for iteration in range(50):
        changed = False
        for i in range(n_circles_arg):
            for j in range(i+1, n_circles_arg):
                if radii[i]<float_tol and radii[j]<float_tol: continue
                dist = math.sqrt(max(0,np.sum((centers[i]-centers[j])**2)))
                sum_r = radii[i]+radii[j]
                if sum_r > dist + float_tol:
                    if dist < float_tol:
                        if radii[i]>0: radii[i]=0; changed=True
                        if radii[j]>0: radii[j]=0; changed=True; continue
                    scale = dist/sum_r if sum_r > float_tol else 0
                    ni,nj = radii[i]*scale, radii[j]*scale
                    if ni<radii[i]-float_tol: radii[i]=ni; changed=True
                    if nj<radii[j]-float_tol: radii[j]=nj; changed=True
        if not changed and iteration>1: break
    return np.maximum(radii,0)

print("examples/circle_packing/packer_helpers.py written")
