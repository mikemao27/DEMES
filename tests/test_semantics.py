"""
Unit tests for semantic compilation, lambda-reduction, and quantifier evaluation.
"""

import unittest
import os
import shutil
from core.types import LogicalForm
from core.lexicon import LexiconManager
from core.semantics import SemanticCompiler
from core.world_model import WorldModel

class TestSemanticCompiler(unittest.TestCase):

    def setUp(self):
        self.test_dir = "tests/temp_data_semantics"
        os.makedirs(self.test_dir, exist_ok = True)
        self.store_path = os.path.join(self.test_dir, "lexicon.json")

        self.lexicon = LexiconManager(store_path = self.store_path)
        self.world_model = WorldModel()
        self.compiler = SemanticCompiler(self.world_model, self.lexicon)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_standard_predicate_evaluation(self):
        form = LogicalForm(predicate = "PORTABLE", arguments = ["suitcase"], is_negated = False, tense = "present")
        payload = self.compiler.compile_and_evaluate(form)
        self.assertEqual(payload["status"], "success")
        self.assertTrue(payload["truth_value"])

    def test_negated_predicate_evaluation(self):
        # "suitcase" has the PORTABLE primitive in the seeded lexicon, so an explicit
        # negation of that predicate should flip the evaluated result to false.
        form = LogicalForm(predicate = "PORTABLE", arguments = ["suitcase"], is_negated = True, tense = "present")
        payload = self.compiler.compile_and_evaluate(form)
        self.assertEqual(payload["status"], "success")
        self.assertFalse(payload["truth_value"])

    def test_quantified_forall_evaluation(self):
        # Test universal quantification ("Every suitcase is portable").
        form = LogicalForm(predicate = "PORTABLE", arguments = ["suitcase"], is_negated = False, tense = "present")
        form.quantifier_meta = {"operator": "FORALL", "variable": "x", "restrictor": "SUITCASE"}

        payload = self.compiler.compile_and_evaluate(form)
        self.assertEqual(payload["status"], "success")
        self.assertTrue(payload["truth_value"])

    def test_quantified_not_exists_evaluation(self):
        # Test negative existential quantification ("No suitcase is fast").
        form = LogicalForm(predicate = "FAST", arguments = ["suitcase"], is_negated = False, tense = "present")
        form.quantifier_meta = {"operator": "NOT_EXISTS", "variable": "x", "restrictor": "SUITCASE"}

        payload = self.compiler.compile_and_evaluate(form)
        self.assertEqual(payload["status"], "success")
        self.assertTrue(payload["truth_value"])

if __name__ == "__main__":
    unittest.main()
