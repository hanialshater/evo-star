# Constants specific to the circle packing problem instance
# Example for N=26
N_CIRCLES = 26
TARGET_SUM_RADII_REF = 2.630
FLOAT_PRECISION_TOLERANCE = 1e-9

# Feature definitions for MAP-Elites for this N
# Feature 1: Standard deviation of radii. Range: [0, 0.15], Bins: 10
# Feature 2: Average distance of centers from the square's center [0.5, 0.5]. Range: [0, 0.4], Bins: 10
PACKER_FEATURE_DEFINITIONS = [
    {'name': 'std_dev_radii', 'min_val': 0.0, 'max_val': 0.15, 'bins': 10},
    {'name': 'avg_dist_from_center', 'min_val': 0.0, 'max_val': 0.4, 'bins': 10}
]
print("examples/circle_packing/packer_constants.py written")
