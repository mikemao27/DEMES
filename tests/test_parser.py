"""
Unit tests for the DEMES symbolic parser, lexicon manager, and semantic compiler.

How it solves the problem:
    1. Validates that the engine correctly parses both standard statements and quantified noun phrases (e.g., "Every suitcase is portable").
    2. Tests truth-conditional evaluation against the WorldModel database.
    3. Verifies runtime vocabulary growth through dynamic lexicon induction.
"""

import unittest
import os
import shutil
from core.types import LogicalForm
from core.lexicon import LexiconManager
from core.parser import ChartParser
from core.semantics import SemanticCompiler
from core.world_model import WorldModel

class TestDEMESCore(unittest.TestCase):
    
    def setUp(self):
        """
        Sets up a temporary test environment with a clean lexicon store.
        """
        self.test_dir = "tests/temp_data"
        os.makedirs(self.test_dir, exist_ok = True)
        self.store_path = os.path.join(self.test_dir, "lexicon.json")
        
        self.lexicon = LexiconManager(store_path = self.store_path)
        self.world_model = WorldModel()
        self.parser = ChartParser(self.lexicon, self.world_model)
        self.compiler = SemanticCompiler(self.world_model, self.lexicon)

    def tearDown(self):
        """
        Cleans up temporary test files.
        """
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_lexicon_loading_and_seeding(self):
        """
        Verify that the lexicon seeds default entries properly.
        """
        def_data = self.lexicon.get_word_definition("suitcase")
        self.assertIsNotNone(def_data)
        self.assertEqual(def_data["category"], "noun")

    def test_chart_parser_valid_sentence(self):
        """
        Verify parsing of a standard declarative sentence.
        """
        form = self.parser.parse("The suitcase is portable.")
        self.assertIsInstance(form, LogicalForm)
        self.assertEqual(form.predicate, "PORTABLE")
        self.assertIn("suitcase", form.arguments)

    def test_chart_parser_quantified_sentence(self):
        """
        Verify parsing of noun phrase quantification (e.g., "Every suitcase is portable").
        """
        form = self.parser.parse("Every suitcase is portable.")
        self.assertIsInstance(form, LogicalForm)
        self.assertEqual(form.predicate, "PORTABLE")
        self.assertTrue(hasattr(form, "quantifier_meta"))
        self.assertEqual(form.quantifier_meta["operator"], "FORALL")
        self.assertEqual(form.quantifier_meta["restrictor"], "SUITCASE")

    def test_chart_parser_unknown_word(self):
        """
        Verify graceful failure on un-bootstrapped unknown words.
        """
        form = self.parser.parse("Quantum mechanics is weird")
        self.assertIsNone(form)

    def test_semantic_compiler_evaluation(self):
        """Verify semantic evaluation payload generation."""
        form = self.parser.parse("The suitcase is portable.")
        payload = self.compiler.compile_and_evaluate(form)
        self.assertEqual(payload["status"], "success")
        self.assertTrue(payload["truth_value"])

    def test_runtime_lexicon_induction(self):
        """
        Verify dynamic word induction via structural bootstrapping.
        """
        self.lexicon.induce_word("blick", "verb", [{"name": "ACTION", "category": "action"}])
        def_data = self.lexicon.get_word_definition("blick")
        self.assertIsNotNone(def_data)
        self.assertEqual(def_data["category"], "verb")

if __name__ == "__main__":
    unittest.main()