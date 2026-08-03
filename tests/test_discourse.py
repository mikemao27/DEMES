"""
Unit tests for core/discourse.py: cataphora/anaphora resolution, speech act classification, QUD/ellipsis and focus, presupposition accommodation, 
scalar implicature, and rhetorical relations.
"""

import unittest

from core.types import DerivationNode, Gender, GrammaticalNumber, DiscourseReferent, LogicalForm, Aktionsart, FrameTemplate
from core.world_model import WorldModel, FactProvenance
from core.discourse import (
    resolve_pronoun_with_agreement,
    attempt_cataphora_resolution,
    find_pronoun_and_name_leaves,
    SpeechActCategory, classify_speech_act,
    QUDEntry, QUDStack,
    detect_focus,
    accommodate_presupposition,
    generate_scalar_implicature,
    RhetoricalRelation, classify_rhetorical_relation,
)

class TestPronounAgreement(unittest.TestCase):
    def setUp(self):
        self.world_model = WorldModel()

    def test_no_referents_returns_none(self):
        self.assertIsNone(resolve_pronoun_with_agreement(self.world_model))

    def test_resolves_to_most_recent_when_unconstrained(self):
        self.world_model.register_referent("suitcase", "noun")
        self.world_model.register_referent("trophy", "noun")
        result = resolve_pronoun_with_agreement(self.world_model)
        self.assertEqual(result.name, "trophy")

    def test_gender_mismatch_skips_the_recent_candidate(self):
        ref_id = self.world_model.register_referent("Mary", "noun")
        self.world_model.active_referents[ref_id].gender = Gender.FEMININE
        result = resolve_pronoun_with_agreement(self.world_model, required_gender = Gender.MASCULINE)
        self.assertIsNone(result)

    def test_falls_back_to_an_earlier_matching_candidate(self):
        john_id = self.world_model.register_referent("John", "noun")
        self.world_model.active_referents[john_id].gender = Gender.MASCULINE
        mary_id = self.world_model.register_referent("Mary", "noun")
        self.world_model.active_referents[mary_id].gender = Gender.FEMININE

        result = resolve_pronoun_with_agreement(self.world_model, required_gender = Gender.MASCULINE)
        self.assertEqual(result.name, "John")

    def test_unset_referent_gender_is_never_treated_as_a_conflict(self):
        self.world_model.register_referent("suitcase", "noun") # Gender left unset.
        result = resolve_pronoun_with_agreement(self.world_model, required_gender = Gender.FEMININE)
        self.assertIsNotNone(result)

    def test_number_and_animacy_constraints_apply(self):
        ref_id = self.world_model.register_referent("students", "noun")
        self.world_model.active_referents[ref_id].number = GrammaticalNumber.PLURAL
        self.world_model.active_referents[ref_id].animate = True
        self.assertIsNone(resolve_pronoun_with_agreement(self.world_model, required_number = GrammaticalNumber.SINGULAR))
        self.assertIsNotNone(resolve_pronoun_with_agreement(self.world_model, required_animate = True))

class TestCataphoraResolution(unittest.TestCase):
    """
    "Before he entered the room, John took off his coat" is grammatical (he doesn't c-command John); "He entered the room before John did" with the same 
    intended coreference is not (he does c-command John). Both trees are hand-built: core/parser.py doesn't yet support adjunct/subordinate clauses, so 
    this tests the resolution logic against the correct target structure directly, the same way core/parser.py's own c-command tests do.
    """
    def test_licensed_when_pronoun_does_not_c_command_the_name(self):
        he = DerivationNode(label = "NP", token = "he", span = (0, 1))
        entered = DerivationNode(label = "V", token = "entered", span = (1, 2))
        adjunct = DerivationNode(label = "ADJUNCT", children = (he, entered), span = (0, 2))

        john = DerivationNode(label = "NP", token = "John", span = (2, 3))
        took_off = DerivationNode(label = "V", token = "took_off", span = (3, 4))
        matrix = DerivationNode(label = "MATRIX", children = (john, took_off), span = (2, 4))

        root = DerivationNode(label = "S", children = (adjunct, matrix), span = (0, 4))

        result = attempt_cataphora_resolution(root, he, [john])
        self.assertIs(result, john)

    def test_blocked_when_pronoun_c_commands_the_name(self):
        he = DerivationNode(label = "NP", token = "he", span = (0, 1))
        entered = DerivationNode(label = "V", token = "entered", span = (1, 2))
        john = DerivationNode(label = "NP", token = "John", span = (2, 3))
        did = DerivationNode(label = "V", token = "did", span = (3, 4))
        adjunct_with_john = DerivationNode(label = "ADJUNCT", children = (john, did), span = (2, 4))
        vp = DerivationNode(label = "VP", children = (entered, adjunct_with_john), span = (1, 4))
        root = DerivationNode(label = "S", children = (he, vp), span = (0, 4))

        result = attempt_cataphora_resolution(root, he, [john])
        self.assertIsNone(result)

    def test_candidate_before_the_pronoun_is_not_considered_cataphora(self):
        john = DerivationNode(label = "NP", token = "John", span = (0, 1))
        he = DerivationNode(label = "NP", token = "he", span = (1, 2))
        root = DerivationNode(label = "S", children = (john, he), span = (0, 2))

        result = attempt_cataphora_resolution(root, he, [john])
        self.assertIsNone(result) # John precedes he: ordinary backward anaphora's job, not this function's.

    def test_find_pronoun_and_name_leaves_splits_correctly(self):
        he = DerivationNode(label = "NP", token = "he", span = (0, 1))
        john = DerivationNode(label = "NP", token = "John", span = (1, 2))
        room = DerivationNode(label = "NP", token = "room", span = (2, 3))
        root = DerivationNode(label = "S", children = (he, DerivationNode(label = "X", children = (john, room), span = (1, 3))), span = (0, 3))

        pronouns, names = find_pronoun_and_name_leaves(root, {"he", "it", "they"})
        self.assertEqual([p.token for p in pronouns], ["he"])
        self.assertEqual([n.token for n in names], ["John"])

class TestSpeechActClassification(unittest.TestCase):
    def test_greeting_is_expressive(self):
        self.assertEqual(classify_speech_act(None, "Hello"), SpeechActCategory.EXPRESSIVE)

    def test_farewell_is_expressive(self):
        self.assertEqual(classify_speech_act(None, "Goodbye"), SpeechActCategory.EXPRESSIVE)

    def test_question_is_directive(self):
        self.assertEqual(classify_speech_act(None, "Is the suitcase portable?"), SpeechActCategory.DIRECTIVE)

    def test_wh_question_without_question_mark_is_still_directive(self):
        self.assertEqual(classify_speech_act(None, "What time is it"), SpeechActCategory.DIRECTIVE)

    def test_declaration_verb(self):
        form = LogicalForm(predicate = "RESIGN", arguments = ["john"])
        self.assertEqual(classify_speech_act(form, "I resign."), SpeechActCategory.DECLARATION)

    def test_commissive_verb(self):
        form = LogicalForm(predicate = "PROMISE", arguments = ["john"])
        self.assertEqual(classify_speech_act(form, "I promise."), SpeechActCategory.COMMISSIVE)

    def test_directive_verb(self):
        form = LogicalForm(predicate = "FEED", arguments = ["dog"])
        self.assertEqual(classify_speech_act(form, "Feed the dog."), SpeechActCategory.DIRECTIVE)

    def test_default_declarative_is_assertive(self):
        form = LogicalForm(predicate = "PORTABLE", arguments = ["suitcase"])
        self.assertEqual(classify_speech_act(form, "The suitcase is portable."), SpeechActCategory.ASSERTIVE)

    def test_no_logical_form_and_no_special_marking_is_assertive(self):
        self.assertEqual(classify_speech_act(None, "The suitcase is portable."), SpeechActCategory.ASSERTIVE)

class TestQUDStackAndFocus(unittest.TestCase):
    def test_empty_stack_resolves_nothing(self):
        stack = QUDStack()
        self.assertIsNone(stack.resolve_fragment(["friday"]))

    def test_fragment_fills_the_open_slot(self):
        stack = QUDStack()
        stack.push(QUDEntry(predicate = "FREE", arguments = ["you", "tuesday"], open_slot_index = 1))
        resolved = stack.resolve_fragment(["friday"])
        self.assertEqual(resolved.predicate, "FREE")
        self.assertEqual(resolved.arguments, ["you", "friday"])

    def test_entry_with_no_open_slot_resolves_nothing(self):
        stack = QUDStack()
        stack.push(QUDEntry(predicate = "FREE", arguments = ["you", "tuesday"], open_slot_index = None))
        self.assertIsNone(stack.resolve_fragment(["friday"]))

    def test_current_reflects_most_recently_pushed_entry(self):
        stack = QUDStack()
        stack.push(QUDEntry(predicate = "FREE", arguments = ["you", "tuesday"], open_slot_index = 1))
        stack.push(QUDEntry(predicate = "AVAILABLE", arguments = ["room"], open_slot_index = 0))
        self.assertEqual(stack.current().predicate, "AVAILABLE")

    def test_detect_focus_finds_the_word_after_the_particle(self):
        self.assertEqual(detect_focus(["only", "john", "left"]), "john")

    def test_detect_focus_returns_none_without_a_particle(self):
        self.assertIsNone(detect_focus(["john", "left"]))

    def test_detect_focus_ignores_a_trailing_particle_with_nothing_after_it(self):
        self.assertIsNone(detect_focus(["john", "left", "only"]))

class TestPresuppositionAccommodation(unittest.TestCase):
    def setUp(self):
        self.world_model = WorldModel()

    def test_accommodates_when_unknown(self):
        accommodated = accommodate_presupposition(self.world_model, "stop", "john")
        self.assertTrue(accommodated)
        self.assertTrue(self.world_model.check_presupposition("stop", "john"))

    def test_does_not_re_accommodate_when_already_known(self):
        self.world_model.assert_presupposition("stop", "john", holds = True)
        accommodated = accommodate_presupposition(self.world_model, "stop", "john")
        self.assertFalse(accommodated)

    def test_checks_episodic_fact_graph_before_blind_accommodation(self):
        self.world_model.assert_episodic_fact(FrameTemplate.HAS_PART, "john", "sister")
        accommodated = accommodate_presupposition(self.world_model, "stop", "john", related_relation = FrameTemplate.HAS_PART)
        self.assertTrue(accommodated)
        self.assertTrue(self.world_model.check_presupposition("stop", "john"))

class TestScalarImplicature(unittest.TestCase):
    def test_some_implicates_not_all(self):
        self.assertEqual(generate_scalar_implicature("some"), "NOT_ALL")

    def test_might_implicates_not_must(self):
        self.assertEqual(generate_scalar_implicature("might"), "NOT_MUST")

    def test_word_not_on_any_scale_returns_none(self):
        self.assertIsNone(generate_scalar_implicature("suitcase"))

    def test_the_strong_member_of_a_scale_generates_no_implicature(self):
        self.assertIsNone(generate_scalar_implicature("all"))

class TestRhetoricalRelationClassification(unittest.TestCase):
    def test_connective_but_is_contrast(self):
        self.assertEqual(classify_rhetorical_relation(None, None, connective = "but"), RhetoricalRelation.CONTRAST)

    def test_connective_because_is_explanation(self):
        self.assertEqual(classify_rhetorical_relation(None, None, connective = "because"), RhetoricalRelation.EXPLANATION)

    def test_connective_so_is_result(self):
        self.assertEqual(classify_rhetorical_relation(None, None, connective = "so"), RhetoricalRelation.RESULT)

    def test_two_achievements_in_sequence_without_a_connective_is_narration(self):
        result = classify_rhetorical_relation(Aktionsart.ACHIEVEMENT, Aktionsart.ACHIEVEMENT)
        self.assertEqual(result, RhetoricalRelation.NARRATION)

    def test_state_following_an_event_is_background(self):
        result = classify_rhetorical_relation(Aktionsart.ACHIEVEMENT, Aktionsart.STATE)
        self.assertEqual(result, RhetoricalRelation.BACKGROUND)

    def test_unclear_aktionsart_falls_back_to_elaboration(self):
        result = classify_rhetorical_relation(None, None)
        self.assertEqual(result, RhetoricalRelation.ELABORATION)

if __name__ == "__main__":
    unittest.main()