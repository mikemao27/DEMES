"""
Unit tests for core/parser.py: category algebra, combinators, supertagging, derivation-tree utilities (c-command), NPI licensing, and end-to-end chart parsing.
"""

import unittest
import os
import shutil

from core.lexicon import LexiconManager
from core.types import DerivationNode, Gender, GrammaticalNumber
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
    get_pronoun_features,
    find_parent, c_commands, collect_leaves, check_npi_licensing,
    ChartParser,
)

class TestPronounFeatures(unittest.TestCase):
    def test_it_is_singular_inanimate_no_gender(self):
        features = get_pronoun_features("it")
        self.assertIsNone(features["gender"])
        self.assertEqual(features["number"], GrammaticalNumber.SINGULAR)
        self.assertFalse(features["animate"])

    def test_he_is_masculine_animate(self):
        features = get_pronoun_features("he")
        self.assertEqual(features["gender"], Gender.MASCULINE)
        self.assertTrue(features["animate"])

    def test_they_is_plural_with_no_fixed_gender_or_animacy(self):
        features = get_pronoun_features("they")
        self.assertEqual(features["number"], GrammaticalNumber.PLURAL)
        self.assertIsNone(features["gender"])
        self.assertIsNone(features["animate"])

    def test_non_pronoun_returns_none(self):
        self.assertIsNone(get_pronoun_features("suitcase"))

    def test_pronoun_gets_np_category(self):
        self.assertEqual(supertag_function_word("it"), [NP])

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
        self.lexicon.lexicon["student"] = {"category": "noun", "semantic_type": "e", "primitives": [{"name": "PEOPLE", "category": "entity"}], "valency": "none"}
        self.lexicon.lexicon["book"] = {"category": "noun", "semantic_type": "e", "primitives": [{"name": "SOMETHING", "category": "entity"}], "valency": "none"}
        self.lexicon.lexicon["read"] = {"category": "verb", "semantic_type": "<e,<e,t>>", "primitives": [{"name": "SEE", "category": "action"}], "valency": "transitive"}
        self.lexicon.lexicon["enter"] = {"category": "verb", "semantic_type": "<e,<e,t>>", "primitives": [{"name": "MOVE", "category": "action"}], "valency": "transitive"}
        self.lexicon.lexicon["room"] = {"category": "noun", "semantic_type": "e", "primitives": [{"name": "SOMETHING", "category": "entity"}], "valency": "none"}
        self.lexicon.lexicon["walk"] = {"category": "verb", "semantic_type": "<e,t>", "primitives": [{"name": "MOVE", "category": "action"}], "valency": "intransitive"}
        self.lexicon.lexicon["mary"] = {"category": "proper_noun", "semantic_type": "e", "primitives": [], "valency": "none"}
        self.lexicon.lexicon["think"] = {"category": "verb", "semantic_type": "<s,<e,t>>", "primitives": [{"name": "THINK", "category": "mental"}], "valency": "clausal"}
        self.lexicon.lexicon["home"] = {"category": "adjective", "semantic_type": "<e,t>", "primitives": [{"name": "SOMEWHERE", "category": "entity"}], "valency": "none"}
        self.lexicon.lexicon["that"] = {"category": "function", "semantic_type": "None", "primitives": [], "valency": "none"}

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

    def test_single_quantifier_also_populates_the_store(self):
        form = self.parser.parse("Every suitcase is portable.")
        self.assertEqual(len(form.quantifier_store), 1)
        self.assertEqual(form.quantifier_store[0].operator, "FORALL")
        self.assertEqual(form.quantifier_store[0].restrictor, "SUITCASE")

    def test_two_quantifiers_are_both_captured_with_correct_restrictors(self):
        # The exact ambiguity Cooper storage exists for: "every student" and "a book" each need their own restrictor found correctly, regardless of subject/object position.
        form = self.parser.parse("Every student read a book.")
        self.assertEqual(form.predicate, "READ")
        self.assertEqual(set(form.arguments), {"student", "book"})
        self.assertIsNone(form.quantifier_meta)  # Not the single-quantifier shape.
        operators_by_restrictor = {q.restrictor: q.operator for q in form.quantifier_store}
        self.assertEqual(operators_by_restrictor, {"STUDENT": "FORALL", "BOOK": "EXISTS"})

    def test_plural_noun_is_tagged(self):
        form = self.parser.parse("The suitcases is portable.")
        self.assertIn("suitcases", form.plural_arguments)

    def test_singular_noun_is_not_tagged_plural(self):
        form = self.parser.parse("The suitcase is portable.")
        self.assertEqual(form.plural_arguments, [])

    def test_fronted_adjunct_clause_extracts_only_the_matrix_clause(self):
        form = self.parser.parse("Before he entered the room, John walked.")
        self.assertEqual(form.predicate, "WALKED")
        self.assertEqual(form.arguments, ["john"])

    def test_trailing_adjunct_clause_extracts_only_the_matrix_clause(self):
        form = self.parser.parse("John walked before he entered the room.")
        self.assertEqual(form.predicate, "WALKED")
        self.assertEqual(form.arguments, ["john"])

    def test_fronted_adjunct_full_tree_still_contains_the_adjunct_for_structural_checks(self):
        _form, tree = self.parser.parse_with_derivation("Before he entered the room, John walked.")
        all_tokens = [leaf.token for leaf in collect_leaves(tree)]
        self.assertIn("he", all_tokens)
        self.assertIn("entered", all_tokens)

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

    def test_pronoun_parses_as_an_unresolved_argument(self):
        # Resolving what "it" refers to is core/discourse.py's job: the parser's job stops at producing a syntactically valid sentence with the pronoun as a literal argument.
        form = self.parser.parse("It is portable.")
        self.assertEqual(form.predicate, "PORTABLE")
        self.assertEqual(form.arguments, ["it"])

class TestClausalComplements(unittest.TestCase):
    """
    Phase 2.3: mental-predicate verbs ("think") taking a full embedded sentence as their complement, rather than a flat NP argument, and the transparent "that" complementizer that 
    optionally introduces it. The whole point is that the complement gets its own real, independently-extracted LogicalForm (with its own negation and tense) rather than the matrix 
    and embedded clauses' content being flattened together or leaking into each other.
    """
    def setUp(self):
        self.test_dir = "tests/temp_data_parser_clausal"
        os.makedirs(self.test_dir, exist_ok = True)
        self.store_path = os.path.join(self.test_dir, "lexicon.json")
        self.lexicon = LexiconManager(store_path = self.store_path)

        self.lexicon.lexicon["john"] = {"category": "proper_noun", "semantic_type": "e", "primitives": [], "valency": "none"}
        self.lexicon.lexicon["mary"] = {"category": "proper_noun", "semantic_type": "e", "primitives": [], "valency": "none"}
        self.lexicon.lexicon["think"] = {"category": "verb", "semantic_type": "<s,<e,t>>", "primitives": [{"name": "THINK", "category": "mental"}], "valency": "clausal"}
        self.lexicon.lexicon["home"] = {"category": "adjective", "semantic_type": "<e,t>", "primitives": [{"name": "SOMEWHERE", "category": "entity"}], "valency": "none"}
        self.lexicon.lexicon["that"] = {"category": "function", "semantic_type": "None", "primitives": [], "valency": "none"}

        self.parser = ChartParser(self.lexicon)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_clausal_complement_with_complementizer_produces_a_nested_logical_form(self):
        form = self.parser.parse("John thinks that Mary is home.")
        self.assertEqual(form.predicate, "THINKS")
        self.assertEqual(form.arguments[0], "john")
        embedded = form.arguments[1]
        self.assertEqual(embedded.predicate, "HOME")
        self.assertEqual(embedded.arguments, ["mary"])

    def test_complementizer_is_optional(self):
        form = self.parser.parse("John thinks Mary is home.")
        self.assertEqual(form.predicate, "THINKS")
        embedded = form.arguments[1]
        self.assertEqual(embedded.predicate, "HOME")

    def test_embedded_negation_stays_scoped_to_the_embedded_clause_only(self):
        form = self.parser.parse("John thinks Mary is not home.")
        self.assertFalse(form.is_negated)
        embedded = form.arguments[1]
        self.assertTrue(embedded.is_negated)

    def test_matrix_negation_around_the_whole_complement_is_scoped_to_the_matrix_only(self):
        # "did not think" negates the matrix attitude itself, not John's belief about where Mary is - the embedded clause's own truth stays untouched.
        form = self.parser.parse("John did not think that Mary is home.")
        self.assertTrue(form.is_negated)
        self.assertEqual(form.tense, "past")
        embedded = form.arguments[1]
        self.assertFalse(embedded.is_negated)

    def test_matrix_predicate_is_the_clausal_verb_not_the_embedded_predicate(self):
        # The rightmost-predicate-shaped-leaf heuristic used for flat sentences would wrongly pick "home" (the embedded predicate) here, since it comes
        # later in the flat leaf sequence than "thinks" does: a clausal-verb leaf must be preferred whenever one is present.
        form = self.parser.parse("John thinks that Mary is home.")
        self.assertNotEqual(form.predicate, "HOME")
        self.assertEqual(form.predicate, "THINKS")

    def test_parse_with_derivation_exposes_the_tree(self):
        form, root = self.parser.parse_with_derivation("The suitcase is portable.")
        self.assertIsNotNone(form)
        self.assertIsNotNone(root)
        self.assertEqual(root.label, "S")

    def test_parse_with_derivation_returns_none_none_on_failure(self):
        form, root = self.parser.parse_with_derivation("Quantum mechanics is weird")
        self.assertIsNone(form)
        self.assertIsNone(root)

if __name__ == "__main__":
    unittest.main()