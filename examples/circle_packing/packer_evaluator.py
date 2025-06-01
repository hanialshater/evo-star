import numpy as np # evaluator might use np directly
from typing import Tuple, Dict, Any, Optional
# Assuming BaseEvaluator is in alpha_evolve_framework.evaluator_abc
# Adjust import path if your framework directory is named differently or not in sys.path
from alpha_evolve_framework.evaluator_abc import BaseEvaluator
from .packer_helpers import helper_validate_packing, helper_compute_max_radii
# from .packer_constants import N_CIRCLES, TARGET_SUM_RADII_REF, FLOAT_PRECISION_TOLERANCE (or pass them in)

class CirclePackingEvaluator(BaseEvaluator):
    def __init__(self, n_circles: int, target_sum_radii_ref: float, float_precision_tolerance: float,
                 # global_dependencies are less needed if helpers are imported
                 problem_name: str = "CirclePacking"
                ):
        super().__init__(problem_name=f"{problem_name}_N{n_circles}")
        self.n_circles = n_circles
        self.target_sum_radii_ref = target_sum_radii_ref
        self.float_tol = float_precision_tolerance
        # Ensure numpy is available if helpers don't import it themselves
        self.exec_globals_template = {'np': np, 'helper_validate_packing': helper_validate_packing, 'helper_compute_max_radii': helper_compute_max_radii, 'FLOAT_PRECISION_TOLERANCE_FOR_HELPERS': self.float_tol} # Make tol available to helpers if needed
        print(f"CirclePackingEvaluator: N={self.n_circles}, Target SumR={self.target_sum_radii_ref}")

    def evaluate(self, program_id: str, full_code_to_evaluate: str,
                 program_generation: int, stage: int = 0 # Added stage
                ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        scores: Dict[str,float]={'main_score':-float('inf')}
        details: Dict[str,Any]={'program_id':program_id,'generation':program_generation, 'error_message':None,
                                'runtime_error':False,'shape_ok':False,'sum_radii_achieved':0.0,
                                'is_fully_valid_packing':False,'radii_for_features':None,'centers_for_features':None}
        exec_globals = {**self.exec_globals_template} # Fresh copy for each exec
        harness_name = "run_evaluation_harness"
        try:
            exec(full_code_to_evaluate, exec_globals)
            if harness_name not in exec_globals or not callable(exec_globals[harness_name]):
                details.update({'error_message':f"`{harness_name}` not found.",'runtime_error':True}); scores['main_score']=-200.0; return scores,details

            centers_raw, radii_raw, sum_r_raw, exec_err = exec_globals[harness_name](self.n_circles)
            details['sum_radii_achieved'] = float(sum_r_raw) if isinstance(sum_r_raw,(int,float,np.number)) else 0.0
            if exec_err: details.update({'error_message':str(exec_err),'runtime_error':True,'shape_ok':False}); scores['main_score']=-200.0; return scores,details

            centers,radii = (np.array(centers_raw) if not isinstance(centers_raw,np.ndarray) else centers_raw,
                             np.array(radii_raw) if not isinstance(radii_raw,np.ndarray) else radii_raw)

            details['shape_ok']=(centers.ndim==2 and centers.shape==(self.n_circles,2) and radii.ndim==1 and radii.shape==(self.n_circles,))
            if not details['shape_ok']:
                details['error_message']=details.get('error_message', f"Shape error. C:{getattr(centers,'shape','N/A')}, R:{getattr(radii,'shape','N/A')}"); scores['main_score']=-150.0; return scores,details

            details.update({'radii_for_features':radii, 'centers_for_features':centers})
            is_geom_valid, val_details = helper_validate_packing(centers,radii,self.n_circles,self.float_tol)
            details.update(val_details)
            details['is_fully_valid_packing'] = is_geom_valid and details['shape_ok']

            current_sum_r = details['sum_radii_achieved']
            score_val = 0.0
            if details['is_fully_valid_packing']:
                score_val = current_sum_r
                if current_sum_r >= self.target_sum_radii_ref: score_val += (current_sum_r - self.target_sum_radii_ref) * 1.5
            else:
                score_val = current_sum_r if details['shape_ok'] else -10.0
                score_val -= (details.get('out_of_bounds_count',0)*2.0 + details.get('overlap_count',0)*3.0)
                if not details['shape_ok']: score_val -= 20.0
            scores.update({'main_score':max(score_val,-100.0), 'sum_radii':current_sum_r, 'validity_score':1.0 if details['is_fully_valid_packing'] else 0.0,
                           'constraint_violations':float(details.get('out_of_bounds_count',0)+details.get('overlap_count',0))})
        except Exception as e:
            import traceback; details.update({'error_message':f"Evaluator Exception: {e}\n{traceback.format_exc()}",'runtime_error':True}); scores['main_score']=-200.0
        return scores,details

print("examples/circle_packing/packer_evaluator.py written")
