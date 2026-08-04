"""
Unit tests for core/semantics.py: the term language, beta-reduction, Cooper storage, aktionsart derivation, and the practical SemanticCompiler evaluator.
"""

import unittest
import os
import shutil

from core.lexicon import LexiconManager
from core.world_model import WorldModel
from core.types import LogicalForm, LambdaExpression, StoredQuantifier, Explication, FrameTemplate, Aktionsart
from core.semantics import (
    Variable, Constant, Application,
    substitute, beta_reduce, evaluate_term,
    store_quantifier, enumerate_scope_readings,
    derive_aktionsart,
    SemanticCompiler,
)

class TestTermEquality(unittest.TestCase):
    def test_variable_equality(self):
        self.assertEqual(Variable("x"), Variable("x"))
        self.assertNotEqual(Variable("x"), Variable("y"))

    def test_constant_equality(self):
        self.assertEqual(Constant("PORTABLE"), Constant("PORTABLE"))

    def test_application_equality(self):
        a = Application(Constant("PORTABLE"), Variable("x"))
        b = Application(Constant("PORTABLE"), Variable("x"))
        self.assertEqual(a, b)

    def test_terms_are_hashable(self):
        seen = {Variable("x"), Constant("PORTABLE"), Application(Constant("P"), Variable("x"))}
        self.assertEqual(len(seen), 3)

class TestSubstitution(unittest.TestCase):
    def test_substitutes_matching_variable(self):
        result = substitute(Variable("x"), "x", Constant("suitcase"))
        self.assertEqual(result, Constant("suitcase"))

    def test_leaves_non_matching_variable_alone(self):
        result = substitute(Variable("y"), "x", Constant("suitcase"))
        self.assertEqual(result, Variable("y"))

    def test_constants_are_unaffected(self):
        result = substitute(Constant("PORTABLE"), "x", Constant("suitcase"))
        self.assertEqual(result, Constant("PORTABLE"))

    def test_substitutes_through_application(self):
        term = Application(Constant("PORTABLE"), Variable("x"))
        result = substitute(term, "x", Constant("suitcase"))
        self.assertEqual(result, Application(Constant("PORTABLE"), Constant("suitcase")))

    def test_shadowed_variable_in_nested_lambda_is_untouched(self):
        # \x.(\x. x): substituting the outer x must not reach into the inner rebinding.
        inner = LambdaExpression(variable = "x", body = Variable("x"))
        result = substitute(inner, "x", Constant("suitcase"))
        self.assertEqual(result, inner)

    def test_substitutes_inside_non_shadowing_lambda_body(self):
        # \y. PORTABLE(x): substituting x should reach inside, since y != x.
        expr = LambdaExpression(variable = "y", body = Application(Constant("PORTABLE"), Variable("x")))
        result = substitute(expr, "x", Constant("suitcase"))
        self.assertEqual(result.body, Application(Constant("PORTABLE"), Constant("suitcase")))

class TestBetaReduction(unittest.TestCase):
    def test_simple_property_application(self):
        # \x. PORTABLE(x), applied to "suitcase" -> PORTABLE(suitcase).
        portable = LambdaExpression(variable = "x", body = Application(Constant("PORTABLE"), Variable("x")))
        result = beta_reduce(portable, Constant("suitcase"))
        self.assertEqual(result, Application(Constant("PORTABLE"), Constant("suitcase")))

    def test_curried_two_argument_application_fully_reduces(self):
        # \y.\x. KICK(x, y): John kicked the ball: apply to object first, then subject.
        kick = LambdaExpression(
            variable = "y",
            body = LambdaExpression(
                variable = "x",
                body = Application(Application(Constant("KICK"), Variable("x")), Variable("y")),
            ),
        )

        applied_to_object = beta_reduce(kick, Constant("ball"))
        fully_applied = evaluate_term(Application(applied_to_object, Constant("john")))
        expected = Application(Application(Constant("KICK"), Constant("john")), Constant("ball"))
        self.assertEqual(fully_applied, expected)

    def test_evaluate_term_on_already_reduced_constant_is_a_no_op(self):
        self.assertEqual(evaluate_term(Constant("suitcase")), Constant("suitcase"))

class TestCooperStorage(unittest.TestCase):
    def test_store_quantifier_appends_without_mutating_original(self):
        original = []
        q = StoredQuantifier(operator = "FORALL", restrictor = "STUDENT", bound_variable = "x", context_id = "global")
        updated = store_quantifier(original, q)
        self.assertEqual(original, [])
        self.assertEqual(updated, [q])

    def test_two_quantifiers_yield_two_distinct_scope_readings(self):
        every_student = StoredQuantifier(operator = "FORALL", restrictor = "STUDENT", bound_variable = "x", context_id = "global")
        a_book = StoredQuantifier(operator = "EXISTS", restrictor = "BOOK", bound_variable = "y", context_id = "global")
        readings = enumerate_scope_readings([every_student, a_book])
        self.assertEqual(len(readings), 2)
        self.assertIn([every_student, a_book], readings)
        self.assertIn([a_book, every_student], readings)

    def test_single_quantifier_has_one_reading(self):
        q = StoredQuantifier(operator = "EXISTS", restrictor = "BOOK", bound_variable = "y", context_id = "global")
        self.assertEqual(enumerate_scope_readings([q]), [[q]])

    def test_first_reading_is_always_the_surface_left_to_right_order(self):
        # SemanticCompiler._evaluate_scoped_quantifiers relies on this itertools.permutations property as its default reading: pinned explicitly
        # here so that reliance is never silently invalidated by a future change to enumerate_scope_readings' implementation.
        every_student = StoredQuantifier(operator = "FORALL", restrictor = "STUDENT", bound_variable = "x0", context_id = "global")
        a_book = StoredQuantifier(operator = "EXISTS", restrictor = "BOOK", bound_variable = "x1", context_id = "global")
        self.assertEqual(enumerate_scope_readings([every_student, a_book])[0], [every_student, a_book])

class TestAktionsartDerivation(unittest.TestCase):
    def test_has_property_is_a_state(self):
        explication = Explication(frame = FrameTemplate.HAS_PROPERTY, slots = {"x": "suitcase", "property": "PORTABLE"})
        self.assertEqual(derive_aktionsart(explication), Aktionsart.STATE)

    def test_bare_does_is_an_activity(self):
        explication = Explication(frame = FrameTemplate.DOES, slots = {"x": "john", "action": "RUN"})
        self.assertEqual(derive_aktionsart(explication), Aktionsart.ACTIVITY)

    def test_does_with_result_is_an_accomplishment(self):
        explication = Explication(frame = FrameTemplate.DOES, slots = {"x": "john", "action": "BUILD", "result": "HOUSE"})
        self.assertEqual(derive_aktionsart(explication), Aktionsart.ACCOMPLISHMENT)

    def test_happens_to_is_an_achievement(self):
        explication = Explication(frame = FrameTemplate.HAPPENS_TO, slots = {"event": "WIN", "x": "john"})
        self.assertEqual(derive_aktionsart(explication), Aktionsart.ACHIEVEMENT)

    def test_causes_is_an_accomplishment(self):
        explication = Explication(frame = FrameTemplate.CAUSES, slots = {"event1": "PUSH", "event2": "FALL"})
        self.assertEqual(derive_aktionsart(explication), Aktionsart.ACCOMPLISHMENT)

    def test_unrelated_frame_returns_none(self):
        explication = Explication(frame = FrameTemplate.PART_OF, slots = {"x": "wheel", "y": "car"})
        self.assertIsNone(derive_aktionsart(explication))

class TestSemanticCompilerEvaluation(unittest.TestCase):
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

    def test_failure_payload_for_unparsable_sentence(self):
        payload = self.compiler.compile_and_evaluate(None)
        self.assertEqual(payload["status"], "failure")

    def test_intensional_match_for_seeded_portable_suitcase(self):
        form = LogicalForm(predicate = "PORTABLE", arguments = ["suitcase"])
        payload = self.compiler.compile_and_evaluate(form)
        self.assertEqual(payload["status"], "success")
        self.assertTrue(payload["truth_value"])

    def test_negation_inverts_the_result(self):
        form = LogicalForm(predicate = "PORTABLE", arguments = ["suitcase"], is_negated = True)
        payload = self.compiler.compile_and_evaluate(form)
        self.assertFalse(payload["truth_value"])

    def test_quantified_forall_evaluation(self):
        form = LogicalForm(predicate = "PORTABLE", arguments = ["suitcase"])
        form.quantifier_meta = {"operator": "FORALL", "variable": "x", "restrictor": "SUITCASE"}
        payload = self.compiler.compile_and_evaluate(form)
        self.assertTrue(payload["truth_value"])

    def test_quantified_not_exists_evaluation(self):
        form = LogicalForm(predicate = "FAST", arguments = ["suitcase"])
        form.quantifier_meta = {"operator": "NOT_EXISTS", "variable": "x", "restrictor": "SUITCASE"}
        payload = self.compiler.compile_and_evaluate(form)
        self.assertTrue(payload["truth_value"]) # Nothing is recorded as "fast", so "no suitcase is fast" holds.

    def test_extensional_fallback_for_untracked_adjective(self):
        # "heavy" isn't in the seeded lexicon at all here, so intensional match can't apply: falls through to the world model's own knowledge_base, which does track "heavy".
        form = LogicalForm(predicate = "HEAVY", arguments = ["suitcase"])
        payload = self.compiler.compile_and_evaluate(form)
        self.assertTrue(payload["truth_value"])

class TestScopedQuantifierEvaluation(unittest.TestCase):
    """
    Phase 3, Sub-step A3: real Cooper-storage evaluation for a genuinely multi-quantifier sentence, using core/world_model.py's entities_of_kind
    and relational_facts (Sub-steps A1/A2) plus core/semantics.py's own beta-reduction machinery at the leaf.
    """
    def setUp(self):
        self.test_dir = "tests/temp_data_semantics_scoped"
        os.makedirs(self.test_dir, exist_ok = True)
        self.store_path = os.path.join(self.test_dir, "lexicon.json")

        self.lexicon = LexiconManager(store_path = self.store_path)
        self.world_model = WorldModel()
        self.world_model.knowledge_base["student"] = ["alice", "bob"]
        self.world_model.knowledge_base["book"] = ["book1"]
        self.compiler = SemanticCompiler(self.world_model, self.lexicon)

        self.every_student = StoredQuantifier(operator = "FORALL", restrictor = "STUDENT", bound_variable = "x0", context_id = "global")
        self.a_book = StoredQuantifier(operator = "EXISTS", restrictor = "BOOK", bound_variable = "x1", context_id = "global")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _form(self, is_negated: bool = False) -> LogicalForm:
        form = LogicalForm(predicate = "READ", arguments = ["student", "book"], is_negated = is_negated)
        form.quantifier_store = [self.every_student, self.a_book]
        return form

    def test_true_when_every_student_has_a_recorded_pair(self):
        self.world_model.relational_facts["read"] = [("alice", "book1"), ("bob", "book1")]
        payload = self.compiler.compile_and_evaluate(self._form())
        self.assertTrue(payload["truth_value"])

    def test_false_when_one_student_has_no_recorded_pair(self):
        # The first genuinely non-vacuous multi-quantifier falsification: bob was never recorded as having read anything.
        self.world_model.relational_facts["read"] = [("alice", "book1")]
        payload = self.compiler.compile_and_evaluate(self._form())
        self.assertFalse(payload["truth_value"])

    def test_untracked_relation_stays_permissive(self):
        # "read" has no relational facts recorded for it at all: holds_relationally's own permissive-when-untracked default applies.
        payload = self.compiler.compile_and_evaluate(self._form())
        self.assertTrue(payload["truth_value"])

    def test_negation_inverts_the_scoped_result(self):
        self.world_model.relational_facts["read"] = [("alice", "book1")] # False without negation (bob missing).
        payload = self.compiler.compile_and_evaluate(self._form(is_negated = True))
        self.assertTrue(payload["truth_value"])

    def test_quantifier_scope_payload_is_attached(self):
        self.world_model.relational_facts["read"] = [("alice", "book1"), ("bob", "book1")]
        payload = self.compiler.compile_and_evaluate(self._form())
        self.assertEqual(payload["quantifier_scope"]["reading_count"], 2)
        self.assertEqual(payload["quantifier_scope"]["surface_reading"], [self.every_student, self.a_book])
        self.assertEqual(payload["quantifier_scope"]["claim"], "READ(x0)(x1)")

    def test_mismatched_shape_falls_back_without_a_scope_payload(self):
        # Three quantifiers isn't a shape core/parser.py can currently produce: falls back to the ordinary standard-predicate evaluator
        # (permissive, since READ is untracked in knowledge_base) rather than guessing at a generalization nothing can test yet.
        form = LogicalForm(predicate = "READ", arguments = ["student", "book"])
        form.quantifier_store = [self.every_student, self.a_book, self.every_student]
        payload = self.compiler.compile_and_evaluate(form)
        self.assertTrue(payload["truth_value"])
        self.assertNotIn("quantifier_scope", payload)

    def test_exists_operator_only_needs_one_matching_entity(self):
        # "Some student read a book": EXISTS instead of FORALL - alice alone is enough.
        form = LogicalForm(predicate = "READ", arguments = ["student", "book"])
        form.quantifier_store = [
            StoredQuantifier(operator = "EXISTS", restrictor = "STUDENT", bound_variable = "x0", context_id = "global"),
            self.a_book,
        ]
        self.world_model.relational_facts["read"] = [("alice", "book1")]
        payload = self.compiler.compile_and_evaluate(form)
        self.assertTrue(payload["truth_value"])

if __name__ == "__main__":
    unittest.main()