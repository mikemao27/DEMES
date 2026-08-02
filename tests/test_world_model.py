"""
Unit tests for WorldModel fact validation, entity registration, and pronoun resolution.
"""

import unittest
from core.world_model import WorldModel

class TestWorldModel(unittest.TestCase):

    def setUp(self):
        self.world_model = WorldModel()

    def test_assertion_validation(self):
        isValid = self.world_model.validate_assertion("PORTABLE", ["suitcase"])
        self.assertTrue(isValid)

    def test_referent_registration_and_pronoun_resolution(self):
        ref_id = self.world_model.register_referent("suitcase", "noun", ["portable"])
        self.assertIn(ref_id, self.world_model.active_referents)

        resolved = self.world_model.resolve_pronoun("it")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.name, "suitcase")

    def test_discourse_clearance(self):
        self.world_model.register_referent("trophy", "noun", ["portable"])
        self.assertEqual(len(self.world_model.active_referents), 1)
        
        self.world_model.clear_discourse()
        self.assertEqual(len(self.world_model.active_referents), 0)

if __name__ == "__main__":
    unittest.main()