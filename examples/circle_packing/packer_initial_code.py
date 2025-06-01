# Using f-string here means N_CIRCLES must be defined when this module is imported and get_initial_code called.
# Or, make N_CIRCLES an argument to the function.

def get_packer_initial_code_n26_str() -> str:
    # This string is for N_CIRCLES = 26
    # It will be used by Codebase("...")
    # Ensure it uses N_CIRCLES_CONST_ARG as the argument name for the evolved function.
    initial_code = """
import numpy as np # LLM might forget this in the block

# EVOLVE-BLOCK-START packing_core_logic
def evolved_construct_packing_core(N_CIRCLES_CONST_ARG):
    # Initial structured strategy for N_CIRCLES_CONST_ARG (intended for 26)
    centers = np.zeros((N_CIRCLES_CONST_ARG, 2))
    if N_CIRCLES_CONST_ARG == 0:
        return centers # Empty case

    centers[0] = [0.5, 0.5]
    count = 1

    if count < N_CIRCLES_CONST_ARG: # Ring 1
        num_in_ring1 = min(8, N_CIRCLES_CONST_ARG - count)
        ring1_radius = 0.25 # Adjusted for N=26
        for i in range(num_in_ring1):
            angle = 2 * np.pi * i / 8
            centers[count + i] = [0.5 + ring1_radius * np.cos(angle), 0.5 + ring1_radius * np.sin(angle)]
        count += num_in_ring1

    if count < N_CIRCLES_CONST_ARG: # Ring 2
        num_in_ring2 = min(16, N_CIRCLES_CONST_ARG - count)
        ring2_radius = 0.40 # Adjusted for N=26
        for i in range(num_in_ring2):
            angle = 2 * np.pi * i / 16
            centers[count + i] = [0.5 + ring2_radius * np.cos(angle), 0.5 + ring2_radius * np.sin(angle)]
        count += num_in_ring2

    if count < N_CIRCLES_CONST_ARG: # Remaining circles
        # Simple placement for any leftovers (e.g., the 26th circle)
        # For N=26, one circle (index 25) will be placed here.
        print(f"Placing {{N_CIRCLES_CONST_ARG - count}} remaining circles at default positions.")
        for i in range(N_CIRCLES_CONST_ARG - count):
             centers[count + i] = [0.1 + i*0.01, 0.1 + i*0.01] # Stagger slightly to avoid all at 0.1,0.1

    return np.clip(centers, 0.01, 0.99) # Final safety clip
# EVOLVE-BLOCK-END

# Evaluation Harness (fixed part of the code, called by evaluator)
def run_evaluation_harness(N_CIRCLES_ARG_HARNESS):
    # This function will be called by the evaluator after exec-ing the whole script
    centers_internal = evolved_construct_packing_core(N_CIRCLES_CONST_ARG=N_CIRCLES_ARG_HARNESS)

    # Basic shape validation within harness before passing to external helpers
    if not isinstance(centers_internal, np.ndarray) or centers_internal.ndim != 2 or \
       centers_internal.shape[0] != N_CIRCLES_ARG_HARNESS or centers_internal.shape[1] != 2:
        # Ensure np is available (it is, from top of script)
        return np.array([[]]), np.array([]), 0.0, f"Centers shape error in harness: Expected ({{N_CIRCLES_ARG_HARNESS}}, 2), got {{getattr(centers_internal, 'shape', type(centers_internal))}}"

    # These helpers are expected to be injected into the exec_globals by the evaluator
    # For testing this module standalone, they'd need to be defined or imported here.
    # In the framework, the CirclePackingEvaluator ensures they are in scope.
    radii_internal = helper_compute_max_radii(centers_internal, N_CIRCLES_ARG_HARNESS, FLOAT_PRECISION_TOLERANCE_FOR_HELPERS)
    sum_r_internal = np.sum(radii_internal)
    return centers_internal, radii_internal, sum_r_internal, None
"""
    return initial_code

print("examples/circle_packing/packer_initial_code.py written")
