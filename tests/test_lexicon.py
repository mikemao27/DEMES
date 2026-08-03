"""
Unit tests for core/lexicon.py: loading/validation, morphology, word-sense disambiguation, proper-noun minting, and the provisional-word induction lifecycle.
"""

import unittest
import os
import json
import shutil

from core.lexicon import LexiconManager, _generate_lemma_candidates
from core.primitives import InvalidPrimitiveError

class TestLexiconManagerBase(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_data_lexicon"
        os.makedirs(self.test_dir, exist_ok = True)
        self.store_path = os.path.join(self.test_dir, "lexicon.json")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

class TestSeedLexiconValidity(TestLexiconManagerBase):
    def test_seed_lexicon_has_no_validation_warnings(self):
        manager = LexiconManager(store_path = self.store_path)
        self.assertEqual(manager.load_warnings, [])

    def test_seed_words_are_present(self):
        manager = LexiconManager(store_path = self.store_path)
        self.assertIsNotNone(manager.get_word_definition("walk"))
        self.assertIsNotNone(manager.get_word_definition("suitcase"))
        self.assertIsNotNone(manager.get_word_definition("portable"))

    def test_seed_gets_persisted_to_disk(self):
        LexiconManager(store_path = self.store_path)
        self.assertTrue(os.path.exists(self.store_path))

class TestLoadTimeValidationExcludesBadEntries(TestLexiconManagerBase):
    def test_entry_with_invalid_primitive_is_excluded_and_warned(self):
        raw = {
            "walk": {
                "category": "verb",
                "semantic_type": "<e, t>",
                "primitives": [{"name": "MOVE", "category": "action"}],
                "valency": "intransitive",
            },
            "portable": {
                "category": "adjective",
                "semantic_type": "<e, t>",
                "primitives": [{"name": "MUSHROOM_OR_PORTABLE", "category": "property"}],
                "valency": "none",
            },
        }
        with open(self.store_path, "w", encoding = "utf-8") as f:
            json.dump(raw, f)

        manager = LexiconManager(store_path = self.store_path)

        self.assertIsNotNone(manager.get_word_definition("walk"))
        self.assertIsNone(manager.get_word_definition("portable"))
        self.assertEqual(len(manager.load_warnings), 1)
        self.assertIn("portable", manager.load_warnings[0])
        self.assertIn("MUSHROOM_OR_PORTABLE", manager.load_warnings[0])

class TestMorphologyCandidateGeneration(unittest.TestCase):
    def test_regular_plural(self):
        candidates = [candidate for candidate, _label in _generate_lemma_candidates("suitcases")]
        self.assertIn("suitcase", candidates)

    def test_regular_past_tense(self):
        candidates = [candidate for candidate, _label in _generate_lemma_candidates("walked")]
        self.assertIn("walk", candidates)

    def test_regular_progressive(self):
        candidates = [candidate for candidate, _label in _generate_lemma_candidates("walking")]
        self.assertIn("walk", candidates)

    def test_progressive_with_consonant_doubling(self):
        candidates = [candidate for candidate, _label in _generate_lemma_candidates("running")]
        self.assertIn("run", candidates)

    def test_progressive_with_dropped_e(self):
        candidates = [candidate for candidate, _label in _generate_lemma_candidates("making")]
        self.assertIn("make", candidates)

    def test_superlative_with_consonant_doubling(self):
        candidates = [candidate for candidate, _label in _generate_lemma_candidates("biggest")]
        self.assertIn("big", candidates)

    def test_comparative_with_consonant_doubling(self):
        candidates = [candidate for candidate, _label in _generate_lemma_candidates("bigger")]
        self.assertIn("big", candidates)

    def test_irregular_forms(self):
        self.assertIn(("give", "irregular"), _generate_lemma_candidates("gave"))
        self.assertIn(("run", "irregular"), _generate_lemma_candidates("ran"))
        self.assertIn(("good", "irregular"), _generate_lemma_candidates("better"))
        self.assertIn(("person", "irregular"), _generate_lemma_candidates("people"))

class TestLexiconLookupWithInflection(TestLexiconManagerBase):
    def setUp(self):
        super().setUp()
        self.manager = LexiconManager(store_path = self.store_path)

    def test_get_word_definition_handles_plural(self):
        definition = self.manager.get_word_definition("suitcases")
        self.assertIsNotNone(definition)
        self.assertEqual(definition["category"], "noun")

    def test_get_word_definition_handles_past_tense(self):
        self.assertIsNotNone(self.manager.get_word_definition("walked"))

    def test_get_word_definition_handles_progressive(self):
        self.assertIsNotNone(self.manager.get_word_definition("walking"))

    def test_get_word_definition_returns_none_for_truly_unknown_word(self):
        self.assertIsNone(self.manager.get_word_definition("quantum"))

    def test_lemmatize_base_form_returns_itself(self):
        self.assertEqual(self.manager.lemmatize("walk"), "walk")

    def test_lemmatize_inflected_form(self):
        self.assertEqual(self.manager.lemmatize("walked"), "walk")

    def test_lemmatize_unknown_word_returns_none(self):
        self.assertIsNone(self.manager.lemmatize("quantum"))

    def test_detect_inflection_past(self):
        self.assertEqual(self.manager.detect_inflection("walked"), "past")

    def test_detect_inflection_progressive(self):
        self.assertEqual(self.manager.detect_inflection("walking"), "progressive")

    def test_detect_inflection_base_form_is_none(self):
        self.assertIsNone(self.manager.detect_inflection("walk"))

    def test_detect_inflection_irregular_past_participle_is_distinguished(self):
        # "taken" is unambiguously a past participle (unlike the generic "irregular" label most irregular forms get): the distinction Phase 2.5's
        # passive-voice detection in core/parser.py needs, since a regular verb like "kicked" genuinely can't make this distinction on its own.
        self.manager.lexicon["take"] = {"category": "verb", "semantic_type": "<e,<e,t>>", "primitives": [], "valency": "transitive"}
        self.assertEqual(self.manager.detect_inflection("taken"), "irregular_past_participle")

    def test_detect_inflection_irregular_simple_past_stays_generic(self):
        self.manager.lexicon["take"] = {"category": "verb", "semantic_type": "<e,<e,t>>", "primitives": [], "valency": "transitive"}
        self.assertEqual(self.manager.detect_inflection("took"), "irregular")

class TestWordSenseDisambiguation(TestLexiconManagerBase):
    def setUp(self):
        super().setUp()
        self.manager = LexiconManager(store_path = self.store_path)

        # Inject a polysemous entry directly, mirroring the shape used in data/lexicon.json.
        self.manager.lexicon["bank"] = {
            "category": "noun",
            "semantic_type": "e",
            "primitives": [],
            "valency": "none",
            "senses": [
                {"sense_key": "bank.n.financial_institution", "selectional_constraint": "INSTITUTION"},
                {"sense_key": "bank.n.river_side", "selectional_constraint": "GEOGRAPHICAL_FEATURE"},
            ],
        }

    def test_picks_sense_matching_context(self):
        sense = self.manager.disambiguate_sense("bank", ["GEOGRAPHICAL_FEATURE"])
        self.assertEqual(sense, "bank.n.river_side")

    def test_falls_back_to_first_sense_when_no_context_matches(self):
        sense = self.manager.disambiguate_sense("bank", [])
        self.assertEqual(sense, "bank.n.financial_institution")

    def test_monosemous_word_returns_itself(self):
        sense = self.manager.disambiguate_sense("walk", ["ANYTHING"])
        self.assertEqual(sense, "walk")

    def test_unknown_word_returns_itself(self):
        sense = self.manager.disambiguate_sense("quantum", [])
        self.assertEqual(sense, "quantum")

class TestProperNounMinting(TestLexiconManagerBase):
    def setUp(self):
        super().setUp()
        self.manager = LexiconManager(store_path = self.store_path)

    def test_default_type_is_entity(self):
        referent = self.manager.mint_proper_noun("Seattle")
        self.assertEqual(referent.type, "ENTITY")
        self.assertFalse(referent.animate)

    def test_subject_of_animate_verb_mints_person(self):
        referent = self.manager.mint_proper_noun("John", syntactic_role = "subject_of_animate_verb")
        self.assertEqual(referent.type, "PERSON")
        self.assertTrue(referent.animate)

    def test_object_of_locative_preposition_mints_place(self):
        referent = self.manager.mint_proper_noun("Seattle", syntactic_role = "object_of_locative_preposition")
        self.assertEqual(referent.type, "PLACE")

    def test_minting_does_not_write_to_lexicon(self):
        self.manager.mint_proper_noun("John", syntactic_role = "subject_of_animate_verb")
        self.assertNotIn("john", self.manager.lexicon)
        self.assertNotIn("john", self.manager.provisional_lexicon)

    def test_successive_mints_get_distinct_ids(self):
        first = self.manager.mint_proper_noun("John")
        second = self.manager.mint_proper_noun("Mary")
        self.assertNotEqual(first.id, second.id)

class TestProvisionalWordInduction(TestLexiconManagerBase):
    def setUp(self):
        super().setUp()
        self.manager = LexiconManager(store_path = self.store_path)

    def test_induced_word_is_findable(self):
        self.manager.induce_word("blick", "verb", [{"name": "MOVE", "category": "action"}])
        definition = self.manager.get_word_definition("blick")
        self.assertIsNotNone(definition)
        self.assertEqual(definition["provenance"], "induced_unverified")

    def test_induced_word_is_provisional_not_permanent(self):
        self.manager.induce_word("blick", "verb", [{"name": "MOVE", "category": "action"}])
        self.assertIn("blick", self.manager.provisional_lexicon)
        self.assertNotIn("blick", self.manager.lexicon)

    def test_induced_word_is_not_persisted_to_disk(self):
        self.manager.induce_word("blick", "verb", [{"name": "MOVE", "category": "action"}])
        with open(self.store_path, "r", encoding = "utf-8") as f:
            on_disk = json.load(f)
        self.assertNotIn("blick", on_disk)

    def test_induce_word_rejects_invalid_primitive(self):
        with self.assertRaises(InvalidPrimitiveError):
            self.manager.induce_word("blick", "verb", [{"name": "NOT_A_REAL_PRIMITIVE", "category": "action"}])

    def test_promote_to_permanent_moves_and_persists(self):
        self.manager.induce_word("blick", "verb", [{"name": "MOVE", "category": "action"}])
        self.manager.promote_to_permanent("blick")

        self.assertIn("blick", self.manager.lexicon)
        self.assertNotIn("blick", self.manager.provisional_lexicon)
        self.assertEqual(self.manager.lexicon["blick"]["provenance"], "authoritative")

        # Confirm it actually reached disk by loading a completely fresh manager from the same path.
        reloaded = LexiconManager(store_path = self.store_path)
        self.assertIsNotNone(reloaded.get_word_definition("blick"))

    def test_promote_unknown_word_raises(self):
        with self.assertRaises(KeyError):
            self.manager.promote_to_permanent("never_induced")

if __name__ == "__main__":
    unittest.main()