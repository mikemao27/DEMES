"""
Unit tests for core/parser.py: category algebra, combinators, supertagging, derivation-tree utilities (c-command), NPI licensing, and end-to-end chart parsing.
"""

import unittest
import os
import shutil

from core.lexicon import LexiconManager
from core.types import DerivationNode
from core.parser import (
    CCGCategory,
    S, NP, N,
    INTRANSITIVE_VERB, TRANSITIVE_VERB, DITRANSITIVE_VERB,
    PREDICATIVE_ADJECTIVE, ATTRIBUTIVE_ADJECTIVE,
    COMPARATIVE_PREDICATIVE, DETERMINER,
    try_forward_application, try_backward_application,
    try_forward_composition, try_backward_composition,
    type_raise,
    supertag_content_word, supertag_function_word,
    find_parent, c_commands, collect_leaves, check_npi_licensing,
    ChartParser,
)

class TestCCGCategoryAlgebra(unittest.TestCase):

    def test_atomic_repr(self):
        self.assertEqual(repr(S), "S")
        self.assertEqual(repr(NP), "NP")

    def test_complex_repr(self):
        self.assertEqual(repr(INTRANSITIVE_VERB), r"(S\NP)")
        self.assertEqual(repr(TRANSITIVE_VERB), r"((S\NP)/NP)")

    def test_equality_and_hash_for_equivalent_categories(self):
        rebuilt = CCGCategory(S, NP, "\\")
        self.assertEqual(INTRANSITIVE_VERB, rebuilt)
        self.assertEqual(hash(INTRANSITIVE_VERB), hash(rebuilt))

    def test_inequality_for_different_direction(self):
        self.assertNotEqual(CCGCategory(S, NP, "\\"), CCGCategory(S, NP, "/"))

    def test_is_atomic(self):
        self.assertTrue(S.is_atomic())
        self.assertFalse(INTRANSITIVE_VERB.is_atomic())

class TestCombinators(unittest.TestCase):

    def test_forward_application(self):
        # "the"(NP/N) + "suitcase"(N) -> NP.
        self.assertEqual(try_forward_application(DETERMINER, N), NP)

    def test_forward_application_fails_on_type_mismatch(self):
        self.assertIsNone(try_forward_application(DETERMINER, S))

    def test_backward_application(self):
        # "suitcase"(NP) + "walks"(S\NP) -> S.
        self.assertEqual(try_backward_application(NP, INTRANSITIVE_VERB), S)

    def test_backward_application_fails_on_atomic_right(self):
        self.assertIsNone(try_backward_application(NP, S))

    def test_forward_composition(self):
        # X/Y + Y/Z -> X/Z.
        x_over_y = CCGCategory(S, N, "/")
        y_over_z = CCGCategory(N, NP, "/")
        result = try_forward_composition(x_over_y, y_over_z)
        self.assertEqual(result, CCGCategory(S, NP, "/"))

    def test_backward_composition(self):
        # Y\Z + X\Y -> X\Z.
        y_back_z = CCGCategory(N, NP, "\\")
        x_back_y = CCGCategory(S, N, "\\")
        result = try_backward_composition(y_back_z, x_back_y)
        self.assertEqual(result, CCGCategory(S, NP, "\\"))

    def test_type_raise(self):
        raised = type_raise(NP, S)
        # T/(T\X) where T = S, X = NP.
        self.assertEqual(raised, CCGCategory(S, CCGCategory(S, NP, "\\"), "/"))

class TestSupertagging(unittest.TestCase):
    def test_common_noun_gets_bare_n(self):
        entry = {"category": "noun", "valency": "none"}
        self.assertEqual(supertag_content_word("suitcase", entry, None), [N])

    def test_proper_noun_gets_np_directly(self):
        entry = {"category": "proper_noun", "valency": "none"}
        self.assertEqual(supertag_content_word("john", entry, None), [NP])

    def test_pronoun_gets_np(self):
        entry = {"category": "pronoun", "valency": "none"}
        self.assertEqual(supertag_content_word("it", entry, None), [NP])

    def test_intransitive_verb(self):
        entry = {"category": "verb", "valency": "intransitive"}
        self.assertEqual(supertag_content_word("walk", entry, None), [INTRANSITIVE_VERB])

    def test_transitive_verb(self):
        entry = {"category": "verb", "valency": "transitive"}
        self.assertEqual(supertag_content_word("kick", entry, None), [TRANSITIVE_VERB])

    def test_ditransitive_verb(self):
        entry = {"category": "verb", "valency": "ditransitive"}
        self.assertEqual(supertag_content_word("give", entry, None), [DITRANSITIVE_VERB])

    def test_plain_adjective_offers_predicative_then_attributive(self):
        entry = {"category": "adjective", "valency": "none"}
        candidates = supertag_content_word("portable", entry, None)
        self.assertEqual(candidates[0], PREDICATIVE_ADJECTIVE)
        self.assertIn(ATTRIBUTIVE_ADJECTIVE, candidates)

    def test_comparative_adjective_offers_pp_taking_categories_first(self):
        entry = {"category": "adjective", "valency": "none"}
        candidates = supertag_content_word("greater", entry, "comparative")
        self.assertEqual(candidates[0], COMPARATIVE_PREDICATIVE)

    def test_determiner_is_recognized(self):
        self.assertEqual(supertag_function_word("the"), [DETERMINER])

    def test_unrecognized_function_word_returns_empty(self):
        self.assertEqual(supertag_function_word("suitcase"), [])

class TestDerivationTreeCCommand(unittest.TestCase):
    def setUp(self):
        # S -> [NP("he"), VP -> [V("entered"), NP("the room")]].
        self.he = DerivationNode(label = "NP", token = "he", span = (0, 1))
        self.entered = DerivationNode(label = "V", token = "entered", span = (1, 2))
        self.the_room = DerivationNode(label = "NP", token = "room", span = (2, 4))
        self.vp = DerivationNode(label = "VP", children = (self.entered, self.the_room), span = (1, 4))
        self.sentence = DerivationNode(label = "S", children = (self.he, self.vp), span = (0, 4))

    def test_find_parent(self):
        self.assertIs(find_parent(self.sentence, self.he), self.sentence)
        self.assertIs(find_parent(self.sentence, self.entered), self.vp)

    def test_find_parent_of_root_is_none(self):
        self.assertIsNone(find_parent(self.sentence, self.sentence))

    def test_subject_c_commands_everything_in_the_vp(self):
        self.assertTrue(c_commands(self.sentence, self.he, self.entered))
        self.assertTrue(c_commands(self.sentence, self.he, self.the_room))

    def test_verb_does_not_c_command_the_subject(self):
        self.assertFalse(c_commands(self.sentence, self.entered, self.he))

    def test_node_does_not_c_command_itself(self):
        self.assertFalse(c_commands(self.sentence, self.he, self.he))

    def test_node_does_not_c_command_its_own_dominated_descendant(self):
        self.assertFalse(c_commands(self.sentence, self.vp, self.entered))

    def test_collect_leaves_in_order(self):
        tokens = [leaf.token for leaf in collect_leaves(self.sentence)]
        self.assertEqual(tokens, ["he", "entered", "room"])

class TestNPILicensingOnTrees(unittest.TestCase):
    def test_licensed_when_negation_c_commands_npi(self):
        # Not -> [see, anyone] combined under a negation that c-commands both.
        anyone = DerivationNode(label = "NP", token = "anyone", span = (2, 3))
        see = DerivationNode(label = "V", token = "see", span = (1, 2))
        vp = DerivationNode(label = "VP", children = (see, anyone), span = (1, 3))
        not_node = DerivationNode(label = "NEG", token = "not", span = (0, 1))
        root = DerivationNode(label = "S", children = (not_node, vp), span = (0, 3))
        self.assertEqual(check_npi_licensing(root), [])

    def test_unlicensed_when_no_negation_present(self):
        anyone = DerivationNode(label = "NP", token = "anyone", span = (1, 2))
        see = DerivationNode(label = "V", token="see", span = (0, 1))
        root = DerivationNode(label = "VP", children = (see, anyone), span = (0, 2))
        self.assertEqual(check_npi_licensing(root), ["anyone"])

class TestChartParserEndToEnd(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_data_parser"
        os.makedirs(self.test_dir, exist_ok = True)
        self.store_path = os.path.join(self.test_dir, "lexicon.json")
        self.lexicon = LexiconManager(store_path = self.store_path)

        # Extra vocabulary beyond the minimal seed, for the sentence shapes under test.
        self.lexicon.lexicon["john"] = {"category": "proper_noun", "semantic_type": "e", "primitives": [], "valency": "none"}
        self.lexicon.lexicon["i"] = {"category": "pronoun", "semantic_type": "e", "primitives": [], "valency": "none"}
        self.lexicon.lexicon["ball"] = {"category": "noun", "semantic_type": "e", "primitives": [{"name": "SOMETHING", "category": "entity"}], "valency": "none"}
        self.lexicon.lexicon["bucket"] = {"category": "noun", "semantic_type": "e", "primitives": [{"name": "SOMETHING", "category": "entity"}], "valency": "none"}
        self.lexicon.lexicon["trophy"] = {"category": "noun", "semantic_type": "e", "primitives": [{"name": "SOMETHING", "category": "entity"}], "valency": "none"}
        self.lexicon.lexicon["kick"] = {"category": "verb", "semantic_type": "<e,<e,t>>", "primitives": [{"name": "DO", "category": "action"}], "valency": "transitive"}
        self.lexicon.lexicon["see"] = {"category": "verb", "semantic_type": "<e,<e,t>>", "primitives": [{"name": "SEE", "category": "action"}], "valency": "transitive"}
        self.lexicon.lexicon["heavy"] = {"category": "adjective", "semantic_type": "<e,t>", "primitives": [{"name": "BIG", "category": "property"}], "valency": "none"}
        self.lexicon.lexicon["great"] = {"category": "adjective", "semantic_type": "<e,t>", "primitives": [{"name": "GOOD", "category": "property"}], "valency": "none"}

        self.parser = ChartParser(self.lexicon)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_predicative_adjective_sentence(self):
        form = self.parser.parse("The suitcase is portable.")
        self.assertEqual(form.predicate, "PORTABLE")
        self.assertEqual(form.arguments, ["suitcase"])

    def test_transitive_svo_sentence_predicate_is_the_verb(self):
        # The original bug this whole engine replaced: the old linear parser picked the last noun as the predicate. Real CCG composition gets this right structurally.
        form = self.parser.parse("John kicked the ball.")
        self.assertEqual(form.predicate, "KICKED")
        self.assertIn("john", form.arguments)
        self.assertIn("ball", form.arguments)

    def test_attributive_adjective_is_not_mistaken_for_the_predicate(self):
        form = self.parser.parse("The heavy suitcase is portable.")
        self.assertEqual(form.predicate, "PORTABLE")

    def test_negation_is_detected_and_not_treated_as_unknown_word(self):
        form = self.parser.parse("The suitcase is not portable.")
        self.assertTrue(form.is_negated)
        self.assertEqual(form.predicate, "PORTABLE")

    def test_past_tense_copula_is_detected(self):
        form = self.parser.parse("The suitcase was portable.")
        self.assertEqual(form.tense, "past")

    def test_future_marker_is_detected(self):
        form = self.parser.parse("John will kick the ball.")
        self.assertEqual(form.tense, "future")

    def test_quantified_sentence(self):
        form = self.parser.parse("Every suitcase is portable.")
        self.assertEqual(form.quantifier_meta["operator"], "FORALL")
        self.assertEqual(form.quantifier_meta["restrictor"], "SUITCASE")

    def test_comparative_with_explicit_standard(self):
        form = self.parser.parse("The suitcase is greater than the trophy.")
        self.assertEqual(form.predicate, "GREATER")
        self.assertIn("suitcase", form.arguments)
        self.assertIn("trophy", form.arguments)
        self.assertNotIn("than", form.arguments)

    def test_idiom_is_tagged_distinctly_from_literal_use(self):
        idiomatic = self.parser.parse("John kicked the bucket.")
        literal = self.parser.parse("John kicked the ball.")
        self.assertTrue(idiomatic.predicate.startswith("IDIOM:"))
        self.assertFalse(literal.predicate.startswith("IDIOM:"))

    def test_npi_licensed_by_negation_parses(self):
        form = self.parser.parse("I did not see anyone.")
        self.assertIsNotNone(form)

    def test_npi_without_licensor_is_rejected(self):
        form = self.parser.parse("I saw anyone.")
        self.assertIsNone(form)

    def test_unknown_word_fails_gracefully(self):
        form = self.parser.parse("Quantum mechanics is weird")
        self.assertIsNone(form)

    def test_inflected_forms_resolve_via_lexicon_lemmatization(self):
        # "kicked" is not its own lexicon entry: the lexicon's own inflection handling (core/lexicon.py) resolves it to "kick", and the parser never needs to know that happened.
        form = self.parser.parse("John kicked the ball.")
        self.assertIsNotNone(form)

if __name__ == "__main__":
    unittest.main()