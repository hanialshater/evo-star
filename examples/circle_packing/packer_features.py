import numpy as np
from typing import Dict, Any, Optional, List, Tuple

# EXAMPLE_FEATURE_DEFINITIONS can also be defined here or imported from packer_constants
# from .packer_constants import PACKER_FEATURE_DEFINITIONS (if constants are in their own file)

def extract_circle_packing_features(
    eval_details: Dict[str, Any],
    problem_specific_configs: Dict[str, Any], # Expects {'n_circles': value}
    feature_definitions: List[Dict[str, Any]]
) -> Optional[Tuple[float, ...]]:
    prog_id = eval_details.get('program_id', 'N/A')
    n_circles = problem_specific_configs.get('n_circles')
    if n_circles is None:
        print(f"Feature Extractor ({prog_id}): n_circles missing in problem_specific_configs.")
        return None
    if not eval_details.get('shape_ok', False):
        # print(f"Feature Extractor ({prog_id}): Skipping, 'shape_ok' is False.")
        return None
    radii, centers = eval_details.get('radii_for_features'), eval_details.get('centers_for_features')
    if radii is None or centers is None:
        # print(f"Feature Extractor ({prog_id}): Skipping, radii/centers missing.")
        return None
    if not isinstance(radii,np.ndarray): radii=np.array(radii)
    if not isinstance(centers,np.ndarray): centers=np.array(centers)
    if radii.ndim!=1 or radii.shape[0]!=n_circles or centers.ndim!=2 or centers.shape[0]!=n_circles or centers.shape[1]!=2:
        # print(f"Feature Extractor ({prog_id}): Skipping, shape mismatch. R:{radii.shape}, C:{centers.shape}, N:{n_circles}.")
        return None
    if n_circles == 0:
        # print(f"Feature Extractor ({prog_id}): Skipping, n_circles is 0.")
        return None # Or default min features

    computed_features = []
    square_center = np.array([0.5, 0.5])
    for f_def in feature_definitions:
        try:
            if f_def['name'] == 'std_dev_radii':
                val = np.std(radii) if n_circles > 1 else 0.0
                computed_features.append(np.clip(val, f_def['min_val'], f_def['max_val']))
            elif f_def['name'] == 'avg_dist_from_center':
                val = np.mean(np.linalg.norm(centers - square_center, axis=1)) if n_circles > 0 else 0.0
                computed_features.append(np.clip(val, f_def['min_val'], f_def['max_val']))
            else: return None # Unknown feature
        except Exception as e: print(f"Feature Extractor ({prog_id}): Error for '{f_def['name']}': {e}"); return None
    if len(computed_features) != len(feature_definitions): return None
    return tuple(computed_features)

print("examples/circle_packing/packer_features.py written")
