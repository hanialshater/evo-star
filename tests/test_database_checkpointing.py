import unittest
import tempfile
import os
import numpy as np
import pickle # Not strictly needed for tests unless we manually pickle/unpickle outside methods

# Make sure PYTHONPATH is set up correctly if running from root, or adjust path
# For example, if 'src' is in the parent directory of 'tests'
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core_types import Program
from src.map_elites_database import MAPElitesDatabase
from src.simple_program_database import SimpleProgramDatabase

class TestMAPElitesDatabaseCheckpointing(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        self.checkpoint_file_path = self.temp_file.name

        self.feature_definitions = [
            {'name': 'feature1', 'min_val': 0.0, 'max_val': 1.0, 'bins': 10},
            {'name': 'feature2', 'min_val': 0.0, 'max_val': 1.0, 'bins': 10}
        ]
        self.context_program_capacity = 5
        self.db = MAPElitesDatabase(
            feature_definitions=self.feature_definitions,
            context_program_capacity=self.context_program_capacity
        )

        # Create and add some programs
        self.p1 = Program(id="p1", code_str="code1", scores={'main_score': 0.9}, features=(0.15, 0.25), generation=1) # Bin (1,2) approx
        self.p2 = Program(id="p2", code_str="code2", scores={'main_score': 0.8}, features=(0.85, 0.75), generation=1) # Bin (8,7) approx
        self.p3 = Program(id="p3", code_str="code3", scores={'main_score': 0.95}, features=(0.15, 0.28), generation=2) # Bin (1,2) - should replace p1 if score is higher
        self.p4_no_features = Program(id="p4", code_str="code4_nofeat", scores={'main_score': 0.7}, features=None, generation=1)


        self.db.add_program(self.p1)
        self.db.add_program(self.p2)
        self.db.add_program(self.p3) # p3 should be in map_elites for its bin, p1 in recent_good_programs
        self.db.add_program(self.p4_no_features) # Should go to recent_good_programs

    def tearDown(self):
        self.temp_file.close()
        if os.path.exists(self.checkpoint_file_path):
            os.remove(self.checkpoint_file_path)

    def test_save_load_checkpoint(self):
        self.db.save_checkpoint(self.checkpoint_file_path)

        loaded_db = MAPElitesDatabase.load_checkpoint(self.checkpoint_file_path)

        self.assertIsNotNone(loaded_db)
        self.assertIsInstance(loaded_db, MAPElitesDatabase)

        self.assertEqual(loaded_db.feature_definitions, self.db.feature_definitions)
        self.assertEqual(loaded_db.context_program_capacity, self.db.context_program_capacity)

        # Compare map_elites
        self.assertEqual(len(loaded_db.map_elites), len(self.db.map_elites))
        for bin_idx, original_prog in self.db.map_elites.items():
            self.assertIn(bin_idx, loaded_db.map_elites)
            loaded_prog = loaded_db.map_elites[bin_idx]
            self.assertEqual(original_prog.id, loaded_prog.id)
            self.assertEqual(original_prog.code_str, loaded_prog.code_str)
            self.assertEqual(original_prog.scores, loaded_prog.scores)
            self.assertEqual(original_prog.generation, loaded_prog.generation)
            if original_prog.features is not None and loaded_prog.features is not None:
                np.testing.assert_array_almost_equal(np.array(original_prog.features), np.array(loaded_prog.features), decimal=5)
            else:
                self.assertEqual(original_prog.features, loaded_prog.features)

        # Compare recent_good_programs (order should be preserved due to sorting on score)
        self.assertEqual(len(loaded_db.recent_good_programs), len(self.db.recent_good_programs))
        for i in range(len(self.db.recent_good_programs)):
            original_prog = self.db.recent_good_programs[i]
            loaded_prog = loaded_db.recent_good_programs[i]
            self.assertEqual(original_prog.id, loaded_prog.id)
            self.assertEqual(original_prog.code_str, loaded_prog.code_str)
            self.assertEqual(original_prog.scores, loaded_prog.scores)
            self.assertEqual(original_prog.generation, loaded_prog.generation)
            if original_prog.features is not None and loaded_prog.features is not None:
                 np.testing.assert_array_almost_equal(np.array(original_prog.features), np.array(loaded_prog.features), decimal=5)
            else:
                self.assertEqual(original_prog.features, loaded_prog.features)

        # Compare bin_edges
        self.assertEqual(len(loaded_db.bin_edges), len(self.db.bin_edges))
        for i in range(len(self.db.bin_edges)):
            np.testing.assert_array_equal(loaded_db.bin_edges[i], self.db.bin_edges[i])

        # Compare num_bins_per_dim
        self.assertEqual(loaded_db.num_bins_per_dim, self.db.num_bins_per_dim)


class TestSimpleProgramDatabaseCheckpointing(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        self.checkpoint_file_path = self.temp_file.name

        self.population_size = 3 # Smaller than num programs added to test pruning
        self.db = SimpleProgramDatabase(population_size=self.population_size)

        self.p1 = Program(id="s1", code_str="simple_code1", scores={'main_score': 0.95}, generation=1)
        self.p2 = Program(id="s2", code_str="simple_code2", scores={'main_score': 0.80}, generation=1)
        self.p3 = Program(id="s3", code_str="simple_code3", scores={'main_score': 0.90}, generation=2)
        self.p4 = Program(id="s4", code_str="simple_code4", scores={'main_score': 0.70}, generation=2) # This one should be pruned

        self.db.add_program(self.p1)
        self.db.add_program(self.p2)
        self.db.add_program(self.p3)
        self.db.add_program(self.p4) # After this, pruning should occur

    def tearDown(self):
        self.temp_file.close()
        if os.path.exists(self.checkpoint_file_path):
            os.remove(self.checkpoint_file_path)

    def test_save_load_checkpoint(self):
        # Ensure pruning has happened as expected before saving
        self.assertEqual(len(self.db.programs), self.population_size)
        self.assertNotIn("s4", self.db.programs) # p4 should have been pruned

        self.db.save_checkpoint(self.checkpoint_file_path)

        loaded_db = SimpleProgramDatabase.load_checkpoint(self.checkpoint_file_path)

        self.assertIsNotNone(loaded_db)
        self.assertIsInstance(loaded_db, SimpleProgramDatabase)

        self.assertEqual(loaded_db.population_size, self.db.population_size)

        # Compare programs dictionary
        self.assertEqual(len(loaded_db.programs), len(self.db.programs))
        for prog_id, original_prog in self.db.programs.items():
            self.assertIn(prog_id, loaded_db.programs)
            loaded_prog = loaded_db.programs[prog_id]
            self.assertEqual(original_prog.id, loaded_prog.id)
            self.assertEqual(original_prog.code_str, loaded_prog.code_str)
            self.assertEqual(original_prog.scores, loaded_prog.scores)
            self.assertEqual(original_prog.generation, loaded_prog.generation)
            # Features might not be primary for SimpleDB, but check if they exist
            if original_prog.features is not None and loaded_prog.features is not None:
                np.testing.assert_array_almost_equal(np.array(original_prog.features), np.array(loaded_prog.features), decimal=5)
            else:
                self.assertEqual(original_prog.features, loaded_prog.features)


if __name__ == '__main__':
    # This allows running the tests directly from this file
    # However, it's often better to run with `python -m unittest discover tests` or similar
    # For the sys.path hack to work correctly when running this file directly,
    # this file must be in a subdirectory (e.g. 'tests') of the project root.
    unittest.main()

# Final print statement to confirm file write (as per convention in this project)
print("tests/test_database_checkpointing.py written")
