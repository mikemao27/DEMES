"""
Unit tests for interface/stylist.py: the symbolic sentence realizer and the (model-optional) LocalStylist wrapper around it.
"""

import unittest
import os
import shutil

from core.lexicon import LexiconManager
from core.types import LogicalForm
from interface.stylist import (
    _inflect_third_person_singular,
    _realize_subject_phrase,
    _realize_verb_phrase,
    _realize_coordinated,
    realize_logical_form,
    LocalStylist,
)

class TestThirdPersonInflection(unittest.TestCase):
    def test_regular_verb(self):
        self.assertEqual(_inflect_third_person_singular("walk"), "walks")

    def test_verb_ending_in_s_gets_es(self):
        self.assertEqual(_inflect_third_person_singular("pass"), "passes")

    def test_verb_ending_in_ch_gets_es(self):
        self.assertEqual(_inflect_third_person_singular("watch"), "watches")

    def test_verb_ending_in_consonant_y_becomes_ies(self):
        self.assertEqual(_inflect_third_person_singular("carry"), "carries")

    def test_verb_ending_in_vowel_y_just_adds_s(self):
        self.assertEqual(_inflect_third_person_singular("play"), "plays")

class _StylistLexiconFixture(unittest.TestCase):
    """
    Shared setUp/tearDown only: deliberately has no test_ methods of its own, so subclassing it for fixture reuse (the same pattern
    tests/test_pipeline.py's TestPipelineBase uses) never re-runs a base class's own tests once per subclass.
    """
    def setUp(self):
        self.test_dir = "tests/temp_data_stylist"
        os.makedirs(self.test_dir, exist_ok = True)
        self.store_path = os.path.join(self.test_dir, "lexicon.json")
        self.lexicon = LexiconManager(store_path = self.store_path)
        self.lexicon.lexicon["john"] = {"category": "proper_noun", "semantic_type": "e", "primitives": [], "valency": "none"}
        self.lexicon.lexicon["mary"] = {"category": "proper_noun", "semantic_type": "e", "primitives": [], "valency": "none"}
        self.lexicon.lexicon["kick"] = {"category": "verb", "semantic_type": "<e,<e,t>>", "primitives": [{"name": "DO", "category": "action"}], "valency": "transitive"}
        self.lexicon.lexicon["walk"] = {"category": "verb", "semantic_type": "<e, t>", "primitives": [{"name": "MOVE", "category": "action"}], "valency": "intransitive"}
        self.lexicon.lexicon["think"] = {"category": "verb", "semantic_type": "<s,<e,t>>", "primitives": [{"name": "THINK", "category": "mental"}], "valency": "clausal"}
        self.lexicon.lexicon["ball"] = {"category": "noun", "semantic_type": "e", "primitives": [{"name": "SOMETHING", "category": "entity"}], "valency": "none"}

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

class TestStylistWithLexicon(_StylistLexiconFixture):
    def test_proper_noun_subject_gets_no_determiner(self):
        self.assertEqual(_realize_subject_phrase("john", self.lexicon), "John")

    def test_common_noun_subject_gets_the_determiner(self):
        self.assertEqual(_realize_subject_phrase("suitcase", self.lexicon), "The suitcase")

    def test_present_tense_verb_phrase_is_inflected(self):
        self.assertEqual(_realize_verb_phrase("walk", "present", is_negated = False), "walks")

    def test_negated_present_tense_uses_does_not(self):
        self.assertEqual(_realize_verb_phrase("walk", "present", is_negated = True), "does not walk")

    def test_past_tense_uses_did_auxiliary(self):
        self.assertEqual(_realize_verb_phrase("walk", "past", is_negated = False), "did walk")

    def test_future_tense_uses_will_auxiliary(self):
        self.assertEqual(_realize_verb_phrase("walk", "future", is_negated = False), "will walk")

    def test_pluperfect_tense_uses_had_auxiliary(self):
        self.assertEqual(_realize_verb_phrase("walk", "pluperfect", is_negated = False), "had walk")

    def test_negated_pluperfect_uses_had_not(self):
        self.assertEqual(_realize_verb_phrase("walk", "pluperfect", is_negated = True), "had not walk")

    def test_realize_pluperfect_sentence(self):
        form = LogicalForm(predicate = "WALKED", arguments = ["john"], tense = "pluperfect")
        self.assertEqual(realize_logical_form(form, self.lexicon), "John had walk.")

    def test_realize_predicative_adjective_sentence(self):
        form = LogicalForm(predicate = "PORTABLE", arguments = ["suitcase"])
        self.assertEqual(realize_logical_form(form, self.lexicon), "The suitcase is portable.")

    def test_realize_negated_adjective_sentence(self):
        form = LogicalForm(predicate = "PORTABLE", arguments = ["suitcase"], is_negated = True)
        self.assertEqual(realize_logical_form(form, self.lexicon), "The suitcase is not portable.")

    def test_realize_past_tense_adjective_sentence(self):
        form = LogicalForm(predicate = "PORTABLE", arguments = ["suitcase"], tense = "past")
        self.assertEqual(realize_logical_form(form, self.lexicon), "The suitcase was portable.")

    def test_realize_intransitive_verb_sentence(self):
        form = LogicalForm(predicate = "WALK", arguments = ["john"])
        self.assertEqual(realize_logical_form(form, self.lexicon), "John walks.")

    def test_realize_transitive_verb_sentence(self):
        form = LogicalForm(predicate = "KICK", arguments = ["john", "ball"])
        self.assertEqual(realize_logical_form(form, self.lexicon), "John kicks the ball.")

    def test_realize_quantified_sentence(self):
        form = LogicalForm(predicate = "PORTABLE", arguments = ["suitcase"])
        form.quantifier_meta = {"operator": "FORALL", "variable": "x", "restrictor": "SUITCASE"}
        self.assertEqual(realize_logical_form(form, self.lexicon), "Every suitcase is portable.")

    def test_idiom_tagged_predicate_realizes_its_own_literal_surface_form(self):
        # Loose-ends cleanup, Sub-step L4: an idiom-tagged predicate used to return None here, falling through to the deterministic formatter,
        # which leaked the raw "IDIOM:kick_bucket" tag into user-facing text. It now recovers the literal trigger verb from the tag and realizes
        # its own correct, literal surface form: never a paraphrase into the idiom's NSM meaning, which stays the transparency view's job.
        form = LogicalForm(predicate = "IDIOM:kick_bucket", arguments = ["john", "bucket"])
        self.assertEqual(realize_logical_form(form, self.lexicon), "John kicks the bucket.")

    def test_verb_with_no_arguments_returns_none(self):
        form = LogicalForm(predicate = "KICK", arguments = [])
        self.assertIsNone(realize_logical_form(form, self.lexicon))

    def test_none_logical_form_returns_none(self):
        self.assertIsNone(realize_logical_form(None, self.lexicon))

    def test_list_logical_form_returns_none(self):
        # realize_logical_form deliberately rejects a coordinated List[LogicalForm] outright: _realize_coordinated is the only path meant to consume one.
        forms = [LogicalForm(predicate = "WALK", arguments = ["john"]), LogicalForm(predicate = "WALK", arguments = ["mary"])]
        self.assertIsNone(realize_logical_form(forms, self.lexicon))

class TestRealizePassiveVoice(_StylistLexiconFixture):
    def test_passive_with_agent(self):
        form = LogicalForm(predicate = "KICKED", arguments = ["suitcase", "john"], is_passive = True, tense = "past")
        self.assertEqual(realize_logical_form(form, self.lexicon), "The suitcase was kicked by John.")

    def test_passive_without_agent(self):
        form = LogicalForm(predicate = "KICKED", arguments = ["suitcase"], is_passive = True, tense = "past")
        self.assertEqual(realize_logical_form(form, self.lexicon), "The suitcase was kicked.")

    def test_passive_present_tense(self):
        form = LogicalForm(predicate = "KICKED", arguments = ["suitcase"], is_passive = True, tense = "present")
        self.assertEqual(realize_logical_form(form, self.lexicon), "The suitcase is kicked.")

    def test_passive_negation(self):
        form = LogicalForm(predicate = "KICKED", arguments = ["suitcase"], is_passive = True, tense = "past", is_negated = True)
        self.assertEqual(realize_logical_form(form, self.lexicon), "The suitcase was not kicked.")

class TestRealizeClausalComplement(_StylistLexiconFixture):
    def test_clausal_complement_sentence(self):
        embedded = LogicalForm(predicate = "HOME", arguments = ["mary"])
        form = LogicalForm(predicate = "THINKS", arguments = ["john", embedded])
        self.assertEqual(realize_logical_form(form, self.lexicon), "John thinks that Mary is home.")

    def test_embedded_negation_is_preserved(self):
        embedded = LogicalForm(predicate = "HOME", arguments = ["mary"], is_negated = True)
        form = LogicalForm(predicate = "THINKS", arguments = ["john", embedded])
        self.assertEqual(realize_logical_form(form, self.lexicon), "John thinks that Mary is not home.")

    def test_unrealizable_embedded_clause_propagates_none(self):
        # An idiom-tagged embedded clause no longer counts as "unrealizable" (Sub-step L4 above), so a verb with no arguments at all
        # (still genuinely uncovered: nothing to build a subject phrase from) is what this test now uses to exercise propagation.
        embedded = LogicalForm(predicate = "KICK", arguments = [])
        form = LogicalForm(predicate = "THINKS", arguments = ["john", embedded])
        self.assertIsNone(realize_logical_form(form, self.lexicon))

class TestRealizeWhQuestions(_StylistLexiconFixture):
    def test_subject_wh_question_ends_with_a_question_mark(self):
        form = LogicalForm(predicate = "WALK", arguments = ["who"])
        self.assertEqual(realize_logical_form(form, self.lexicon), "Who walks?")

    def test_wh_word_is_never_given_a_the_determiner(self):
        form = LogicalForm(predicate = "WALK", arguments = ["what"])
        result = realize_logical_form(form, self.lexicon)
        self.assertNotIn("The what", result)
        self.assertEqual(result, "What walks?")

class TestRealizeIdentityRelation(_StylistLexiconFixture):
    def test_inverted_copular_question_with_what(self):
        form = LogicalForm(predicate = "IS", arguments = ["what", "suitcase"])
        self.assertEqual(realize_logical_form(form, self.lexicon), "What is the suitcase?")

    def test_inverted_copular_question_with_who(self):
        form = LogicalForm(predicate = "IS", arguments = ["who", "john"])
        self.assertEqual(realize_logical_form(form, self.lexicon), "Who is John?")

    def test_right_hand_np_is_not_capitalized_mid_sentence(self):
        form = LogicalForm(predicate = "IS", arguments = ["what", "suitcase"])
        result = realize_logical_form(form, self.lexicon)
        self.assertNotIn("The suitcase?", result)

    def test_predicate_nominal_declarative(self):
        form = LogicalForm(predicate = "IS", arguments = ["john", "teacher"])
        self.assertEqual(realize_logical_form(form, self.lexicon), "John is the teacher.")

class TestRealizeCoordination(_StylistLexiconFixture):
    def test_two_conjuncts_joined_with_and(self):
        conjuncts = [LogicalForm(predicate = "PORTABLE", arguments = ["suitcase"]), LogicalForm(predicate = "WALK", arguments = ["john"])]
        self.assertEqual(_realize_coordinated(conjuncts, "AND", self.lexicon), "The suitcase is portable and John walks.")

    def test_two_conjuncts_joined_with_or(self):
        conjuncts = [LogicalForm(predicate = "PORTABLE", arguments = ["suitcase"]), LogicalForm(predicate = "WALK", arguments = ["john"])]
        self.assertEqual(_realize_coordinated(conjuncts, "OR", self.lexicon), "The suitcase is portable or John walks.")

    def test_proper_noun_leading_a_later_conjunct_stays_capitalized(self):
        # The bug this test locks in: a naive "lowercase every conjunct after the first" approach would wrongly produce "...and john walks."
        conjuncts = [LogicalForm(predicate = "PORTABLE", arguments = ["suitcase"]), LogicalForm(predicate = "WALK", arguments = ["john"])]
        result = _realize_coordinated(conjuncts, "AND", self.lexicon)
        self.assertIn("and John walks", result)
        self.assertNotIn("and john walks", result)

    def test_one_unrealizable_conjunct_fails_the_whole_coordination(self):
        # An idiom-tagged conjunct no longer counts as "unrealizable" (Sub-step L4 above), so a verb with no arguments at all
        # (still genuinely uncovered) is what this test now uses to exercise the whole-coordination-fails behavior.
        conjuncts = [LogicalForm(predicate = "PORTABLE", arguments = ["suitcase"]), LogicalForm(predicate = "KICK", arguments = [])]
        self.assertIsNone(_realize_coordinated(conjuncts, "AND", self.lexicon))

class TestLocalStylistWithoutModel(unittest.TestCase):
    """
    No GGUF weights present in this environment: exercises the fully symbolic path.
    """
    def setUp(self):
        self.test_dir = "tests/temp_data_stylist_render"
        os.makedirs(self.test_dir, exist_ok = True)
        self.store_path = os.path.join(self.test_dir, "lexicon.json")
        self.lexicon = LexiconManager(store_path = self.store_path)
        self.stylist = LocalStylist(self.lexicon, model_path = "definitely/does/not/exist.gguf")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_no_model_is_loaded(self):
        self.assertIsNone(self.stylist.llm)

    def test_render_uses_symbolic_realization_when_available(self):
        form = LogicalForm(predicate = "PORTABLE", arguments = ["suitcase"])
        payload = {"status": "success", "predicate": "PORTABLE", "arguments": ["suitcase"], "truth_value": True}
        self.assertEqual(self.stylist.render(payload, form), "The suitcase is portable.")

    def test_render_falls_back_to_deterministic_formatter_for_failure_payload(self):
        payload = {"status": "failure", "reason": "Unparsable syntax."}
        result = self.stylist.render(payload, None)
        self.assertIn("Unparsable syntax.", result)

    def test_render_falls_back_when_logical_form_shape_is_uncovered(self):
        # A verb predicate with no arguments at all: the symbolic realizer declines (nothing to build a subject phrase from), so the deterministic
        # formatter (which works from the payload dict, not the LogicalForm) takes over. An idiom-tagged predicate used to be the example here, but
        # is now realized correctly by the symbolic realizer (loose-ends cleanup, Sub-step L4), so it's no longer a valid "uncovered" example.
        form = LogicalForm(predicate = "KICK", arguments = [])
        payload = {"status": "success", "predicate": "KICK", "arguments": [], "truth_value": True}
        result = self.stylist.render(payload, form)
        self.assertIn("kick", result.lower())

    def test_render_dispatches_a_coordinated_list_to_the_coordination_realizer(self):
        # The bug this locks in: render() used to hand a List[LogicalForm] straight to realize_logical_form, which crashed with
        # AttributeError('list' object has no attribute 'predicate') the instant a coordinated sentence reached it.
        self.lexicon.lexicon["john"] = {"category": "proper_noun", "semantic_type": "e", "primitives": [], "valency": "none"}
        forms = [LogicalForm(predicate = "PORTABLE", arguments = ["suitcase"]), LogicalForm(predicate = "WALK", arguments = ["john"])]
        payload = {"status": "success", "connective": "AND", "truth_value": True, "conjuncts": []}
        result = self.stylist.render(payload, forms)
        self.assertEqual(result, "The suitcase is portable and John walks.")

    def test_fallback_render_handles_a_coordinated_payload_shape(self):
        # A coordinated semantic payload (core/pipeline.py's _combine_coordinated_payloads) has "conjuncts"/"connective" at the top level,
        # not "predicate"/"arguments": the fallback formatter must not silently produce a generic "STATEMENT ()" message for it.
        payload = {
            "status": "success",
            "connective": "AND",
            "truth_value": True,
            "conjuncts": [
                {"predicate": "PORTABLE", "arguments": ["suitcase"]},
                {"predicate": "WALK", "arguments": ["john"]},
            ],
        }
        result = self.stylist._fallback_render(payload)
        self.assertIn("portable (suitcase)", result)
        self.assertIn("walk (john)", result)
        self.assertIn(" and ", result)

class _FakeLLM:
    """
    A minimal stand-in for llama_cpp.Llama's call interface, for testing the polish path without real weights.
    """
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_prompt = None

    def __call__(self, prompt, max_tokens, stop, temperature):
        self.last_prompt = prompt
        return {"choices": [{"text": self.response_text}]}

class TestLocalStylistWithFakeModel(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_data_stylist_fake_model"
        os.makedirs(self.test_dir, exist_ok = True)
        self.store_path = os.path.join(self.test_dir, "lexicon.json")
        self.lexicon = LexiconManager(store_path = self.store_path)
        self.stylist = LocalStylist(self.lexicon, model_path = "definitely/does/not/exist.gguf")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_polish_receives_the_symbolic_sentence_as_its_base(self):
        fake_llm = _FakeLLM("The suitcase can be carried easily.")
        self.stylist.llm = fake_llm
        form = LogicalForm(predicate = "PORTABLE", arguments = ["suitcase"])
        payload = {"status": "success", "predicate": "PORTABLE", "arguments": ["suitcase"], "truth_value": True}

        result = self.stylist.render(payload, form)

        self.assertIn("The suitcase is portable.", fake_llm.last_prompt)
        self.assertEqual(result, "The suitcase can be carried easily.")

    def test_polish_failure_falls_back_to_the_symbolic_sentence(self):
        class _BrokenLLM:
            def __call__(self, *args, **kwargs):
                raise RuntimeError("model exploded")

        self.stylist.llm = _BrokenLLM()
        form = LogicalForm(predicate = "PORTABLE", arguments = ["suitcase"])
        payload = {"status": "success", "predicate": "PORTABLE", "arguments": ["suitcase"], "truth_value": True}

        result = self.stylist.render(payload, form)
        self.assertEqual(result, "The suitcase is portable.")

    def test_empty_polish_output_falls_back_to_the_symbolic_sentence(self):
        self.stylist.llm = _FakeLLM("   ")
        form = LogicalForm(predicate = "PORTABLE", arguments = ["suitcase"])
        payload = {"status": "success", "predicate": "PORTABLE", "arguments": ["suitcase"], "truth_value": True}

        result = self.stylist.render(payload, form)
        self.assertEqual(result, "The suitcase is portable.")

if __name__ == "__main__":
    unittest.main()