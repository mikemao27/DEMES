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

class TestStylistWithLexicon(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_data_stylist"
        os.makedirs(self.test_dir, exist_ok = True)
        self.store_path = os.path.join(self.test_dir, "lexicon.json")
        self.lexicon = LexiconManager(store_path = self.store_path)
        self.lexicon.lexicon["john"] = {"category": "proper_noun", "semantic_type": "e", "primitives": [], "valency": "none"}
        self.lexicon.lexicon["kick"] = {"category": "verb", "semantic_type": "<e,<e,t>>", "primitives": [{"name": "DO", "category": "action"}], "valency": "transitive"}
        self.lexicon.lexicon["walk"] = {"category": "verb", "semantic_type": "<e, t>", "primitives": [{"name": "MOVE", "category": "action"}], "valency": "intransitive"}
        self.lexicon.lexicon["ball"] = {"category": "noun", "semantic_type": "e", "primitives": [{"name": "SOMETHING", "category": "entity"}], "valency": "none"}

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

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

    def test_idiom_tagged_predicate_returns_none(self):
        form = LogicalForm(predicate = "IDIOM:kick_bucket", arguments = ["john"])
        self.assertIsNone(realize_logical_form(form, self.lexicon))

    def test_none_logical_form_returns_none(self):
        self.assertIsNone(realize_logical_form(None, self.lexicon))

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
        # An idiom-tagged predicate: the symbolic realizer declines, so the deterministic formatter (which works from the 
        # payload dict, not the LogicalForm) takes over.
        form = LogicalForm(predicate="IDIOM:kick_bucket", arguments=["john"])
        payload = {"status": "success", "predicate": "IDIOM:kick_bucket", "arguments": ["john"], "truth_value": True}
        result = self.stylist.render(payload, form)
        self.assertIn("idiom:kick_bucket", result.lower())

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
        self.stylist = LocalStylist(self.lexicon, model_path="definitely/does/not/exist.gguf")

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