"""
Unit tests for the shared data shapes in core/types.py.
"""

import unittest
from core.types import (
    Gender,
    GrammaticalNumber,
    ModalFlavor,
    Aktionsart,
    Primitive,
    LambdaExpression,
    StoredQuantifier,
    ContextIndex,
    Entity,
    EventRecord,
    DerivationNode,
    UnboundPronoun,
    DiscourseReferent,
    LogicalForm,
)

class TestLogicalForm(unittest.TestCase):
    def test_defaults(self):
        form = LogicalForm(predicate = "PORTABLE")
        self.assertEqual(form.arguments, [])
        self.assertFalse(form.is_negated)
        self.assertEqual(form.tense, "present")
        self.assertIsNone(form.quantifier_meta)

    def test_to_dict_includes_quantifier_meta(self):
        form = LogicalForm(predicate = "PORTABLE", arguments = ["suitcase"])
        form.quantifier_meta = {"operator": "FORALL", "variable": "x", "restrictor": "SUITCASE"}
        as_dict = form.to_dict()
        self.assertEqual(as_dict["quantifier_meta"]["operator"], "FORALL")

    def test_to_dict_nested_logical_form_argument(self):
        inner = LogicalForm(predicate = "EAT", arguments = ["dog", "food"])
        outer = LogicalForm(predicate = "KNOW", arguments = ["john", inner])
        as_dict = outer.to_dict()
        self.assertEqual(as_dict["arguments"][1]["predicate"], "EAT")

class TestEntityLattice(unittest.TestCase):
    def test_atomic_entity(self):
        suitcase = Entity(id = "suitcase_1", kind = "SUITCASE")
        self.assertTrue(suitcase.is_atomic())
        self.assertFalse(suitcase.is_sum)

    def test_sum_entity(self):
        students = Entity(id = "students_group", kind = "STUDENT", is_sum = True, members = ["s1", "s2", "s3"])
        self.assertFalse(students.is_atomic())
        self.assertEqual(len(students.members), 3)

class TestContextIndex(unittest.TestCase):
    def test_root_context_has_no_parent(self):
        root = ContextIndex(id = "global")
        self.assertTrue(root.is_root())

    def test_child_context_records_holder_and_flavor(self):
        belief = ContextIndex(id = "john_belief_1", holder = "john", modal_flavor = ModalFlavor.EPISTEMIC, parent_id = "global")
        self.assertFalse(belief.is_root())
        self.assertEqual(belief.holder, "john")
        self.assertEqual(belief.modal_flavor, ModalFlavor.EPISTEMIC)

class TestEventRecordReichenbachTime(unittest.TestCase):
    def test_pluperfect_event_before_reference_before_speech(self):
        # "Mary had left (event) by the time John arrived (reference)." Speech time is now, later than both.
        record = EventRecord(predicate = "LEAVE", event_time = 1, reference_time = 2, speech_time = 3)
        self.assertTrue(record.is_pluperfect())
        self.assertTrue(record.is_past())

    def test_progressive_overlap_event_surrounds_reference(self):
        # "Mary was leaving (event, ongoing) when John arrived (reference)."
        record = EventRecord(predicate = "LEAVE", event_time = 2, reference_time = 2, speech_time = 3)
        self.assertTrue(record.is_progressive_overlap())

    def test_simple_past_is_not_pluperfect(self):
        record = EventRecord(predicate = "WALK", event_time = 2, reference_time = 2, speech_time = 3)
        self.assertTrue(record.is_past())
        self.assertFalse(record.is_pluperfect())

    def test_aktionsart_defaults_to_none(self):
        record = EventRecord(predicate = "KNOW")
        self.assertIsNone(record.aktionsart)

class TestDerivationNodeDominance(unittest.TestCase):
    def setUp(self):
        # A tiny tree shaped like: S -> [NP("he"), VP -> [V("entered"), NP("the room")]].
        self.he = DerivationNode(label = "NP", token = "he", span = (0, 1))
        self.entered = DerivationNode(label = "V", token = "entered", span = (1, 2))
        self.the_room = DerivationNode(label = "NP", token = "room", span = (2, 4))
        self.vp = DerivationNode(label = "VP", children = (self.entered, self.the_room), span = (1, 4))
        self.sentence = DerivationNode(label = "S", children = (self.he, self.vp), span = (0, 4))

    def test_node_dominates_itself(self):
        self.assertTrue(self.he.dominates(self.he))

    def test_parent_dominates_child(self):
        self.assertTrue(self.sentence.dominates(self.he))
        self.assertTrue(self.vp.dominates(self.entered))

    def test_parent_dominates_grandchild(self):
        self.assertTrue(self.sentence.dominates(self.entered))
        self.assertTrue(self.sentence.dominates(self.the_room))

    def test_sibling_does_not_dominate_sibling(self):
        self.assertFalse(self.he.dominates(self.vp))
        self.assertFalse(self.entered.dominates(self.the_room))

    def test_leaf_detection(self):
        self.assertTrue(self.he.is_leaf())
        self.assertFalse(self.vp.is_leaf())

class TestUnboundPronoun(unittest.TestCase):
    def test_carries_structural_requirements(self):
        pronoun = UnboundPronoun(text = "he", required_gender = Gender.MASCULINE, required_animate = True)
        self.assertEqual(pronoun.required_gender, Gender.MASCULINE)
        self.assertTrue(pronoun.required_animate)
        self.assertIsNone(pronoun.origin_node)

class TestDiscourseReferentAgreementFields(unittest.TestCase):
    def test_agreement_fields_default_to_none(self):
        referent = DiscourseReferent(id = "ref_1", name = "suitcase", type = "noun")
        self.assertIsNone(referent.gender)
        self.assertIsNone(referent.number)
        self.assertIsNone(referent.animate)

    def test_agreement_fields_can_be_set(self):
        referent = DiscourseReferent(id = "ref_2", name = "John", type = "noun", gender = Gender.MASCULINE, animate = True)
        self.assertEqual(referent.gender, Gender.MASCULINE)
        self.assertTrue(referent.animate)

class TestStoredQuantifier(unittest.TestCase):
    def test_carries_context_id(self):
        stored = StoredQuantifier(operator = "EXISTS", restrictor = "UNICORN", bound_variable = "x", context_id = "john_want_1")
        self.assertEqual(stored.context_id, "john_want_1")

class TestLambdaExpressionIsPureData(unittest.TestCase):
    def test_holds_variable_and_body_without_evaluating(self):
        expr = LambdaExpression(variable = "x", body = "PORTABLE(x)")
        self.assertEqual(expr.variable, "x")
        self.assertEqual(expr.semantic_type.value, "<e, t>")

class TestPrimitiveRepr(unittest.TestCase):
    def test_repr_format(self):
        prim = Primitive(name = "MOVE", category = "action")
        self.assertEqual(repr(prim), "[MOVE:action]")

if __name__ == "__main__":
    unittest.main()