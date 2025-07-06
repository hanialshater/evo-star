import numpy as np
from typing import Dict, Any, Tuple, List, Optional

# These helpers are now self-contained and don't need a separate file
def _helper_compute_max_radii(centers: np.ndarray, n_circles: int, iterations: int = 50) -> np.ndarray:
    """Computes the maximum radii for a given set of centers."""
    radii = np.array([min(c[0], 1-c[0], c[1], 1-c[1]) for c in centers])
    radii = np.maximum(radii, 0)
    for _ in range(iterations):
        changed = False
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.linalg.norm(centers[i] - centers[j])
                sum_r = radii[i] + radii[j]
                if sum_r > dist and sum_r > 1e-9:
                    scale = dist / sum_r
                    radii[i], radii[j] = radii[i] * scale, radii[j] * scale
                    changed = True
        if not changed:
            break
    return np.maximum(radii, 0)

def _helper_validate_packing(centers: np.ndarray, radii: np.ndarray, n_circles: int) -> Dict[str, Any]:
    """Validates the packing for overlaps and boundary violations."""
    details = {'out_of_bounds_count': 0, 'overlap_count': 0}
    for i in range(n_circles):
        if not (radii[i] <= centers[i, 0] <= 1 - radii[i] and radii[i] <= centers[i, 1] <= 1 - radii[i]):
            details['out_of_bounds_count'] += 1
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            if np.linalg.norm(centers[i] - centers[j]) < radii[i] + radii[j] - 1e-9:
                details['overlap_count'] += 1
    return details

# --- Problem Function 1: Initial Code ---
def get_packer_initial_code() -> str:
    """Provides the initial Python code string for the circle packing problem."""
    return """
import numpy as np

# EVOLVE-BLOCK-START packing_core_logic
def evolved_construct_packing_core(N_CIRCLES_CONST_ARG):
    # A simple initial strategy: place circles on a grid
    num_per_side = int(np.ceil(np.sqrt(N_CIRCLES_CONST_ARG)))
    centers = np.zeros((N_CIRCLES_CONST_ARG, 2))
    xs = np.linspace(0.1, 0.9, num_per_side)
    ys = np.linspace(0.1, 0.9, num_per_side)
    count = 0
    for y in ys:
        for x in xs:
            if count < N_CIRCLES_CONST_ARG:
                centers[count] = [x, y]
                count += 1
    return centers
# EVOLVE-BLOCK-END

# Fixed evaluation harness
def run_evaluation_harness(N_CIRCLES_ARG_HARNESS):
    centers_internal = evolved_construct_packing_core(N_CIRCLES_CONST_ARG=N_CIRCLES_ARG_HARNESS)
    if not isinstance(centers_internal, np.ndarray) or centers_internal.shape != (N_CIRCLES_ARG_HARNESS, 2):
        return None, f"Centers shape error: Expected ({N_CIRCLES_ARG_HARNESS}, 2), got {getattr(centers_internal, 'shape', type(centers_internal))}"
    return centers_internal, None
"""

# --- Problem Function 2: Evaluator ---
def circle_packer_evaluator(full_code_to_evaluate: str, config: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Evaluates a circle packing solution.

    Args:
        full_code_to_evaluate: The complete Python script as a string.
        config: A dictionary with necessary parameters like 'n_circles' and 'evaluation_stage'.
    """
    n_circles = config['n_circles']
    stage = config.get('evaluation_stage', 0)
    scores, details = {'main_score': -1000.0}, {'error_message': 'Evaluation failed to start'}

    try:
        exec_globals = {}
        exec(full_code_to_evaluate, exec_globals)
        harness_fn = exec_globals.get("run_evaluation_harness")
        if not callable(harness_fn):
            raise ValueError("`run_evaluation_harness` function not found in code.")

        centers, error_msg = harness_fn(n_circles)
        if error_msg:
            raise ValueError(error_msg)

        iterations = 10 if stage == 0 else 50 # Cheap vs. expensive evaluation
        radii = _helper_compute_max_radii(centers, n_circles, iterations)
        validation_details = _helper_validate_packing(centers, radii, n_circles)

        sum_radii = np.sum(radii)
        penalty = validation_details['out_of_bounds_count'] * 2.0 + validation_details['overlap_count'] * 3.0

        scores = {'main_score': sum_radii - penalty, 'sum_radii': sum_radii}
        details = {'is_valid': penalty == 0, **validation_details, 'radii_for_features': radii, 'centers_for_features': centers, 'shape_ok': True}

    except Exception as e:
        scores, details = {'main_score': -1000.0}, {'error_message': str(e), 'shape_ok': False}

    return scores, details

# --- Optional Problem Function 3: Feature Extractor ---
def extract_packer_features(eval_details: Dict[str, Any], config: Dict[str, Any]) -> Optional[Tuple[float, ...]]:
    """Extracts features for MAP-Elites if evaluation was successful."""
    if not eval_details.get('shape_ok', False):
        return None
    radii = eval_details['radii_for_features']
    centers = eval_details['centers_for_features']
    std_dev_radii = np.std(radii)
    avg_dist_from_center = np.mean(np.linalg.norm(centers - 0.5, axis=1))
    return (std_dev_radii, avg_dist_from_center)

def get_packer_feature_definitions() -> List[Dict[str, Any]]:
    """Returns feature definitions for MAP-Elites."""
    return [
        {'name': 'std_dev_radii', 'min_val': 0.0, 'max_val': 0.15, 'bins': 10},
        {'name': 'avg_dist_from_center', 'min_val': 0.0, 'max_val': 0.4, 'bins': 10}
    ]

print("examples/circle_packing/problem_functions.py written.")
