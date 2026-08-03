"""
Unit tests for interface/neural_bridge.py: spelling correction, response parsing for all three model-assisted capabilities, and 
the NeuralBridge class's validation/rejection behavior.
"""

import unittest
import os
import shutil

from core.lexicon import LexiconManager
from core.world_model import WorldModel, FactProvenance
from core.types import FrameTemplate
from interface.neural_bridge import (
    _levenshtein_distance,
    suggest_spelling_correction,
    _parse_induction_response,
    _parse_fact_response,
    _parse_implicature_response,
    NeuralBridge,
)

class _FakeLLM:
    """
    Minimal stand-in for llama_cpp.Llama's call interface.
    """
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_prompt = None

    def __call__(self, prompt, max_tokens, stop, temperature):
        self.last_prompt = prompt
        return {"choices": [{"text": self.response_text}]}

class _BrokenLLM:
    def __call__(self, *args, **kwargs):
        raise RuntimeError("model exploded")

class TestLevenshteinDistance(unittest.TestCase):
    def test_identical_strings(self):
        self.assertEqual(_levenshtein_distance("portable", "portable"), 0)

    def test_single_substitution(self):
        self.assertEqual(_levenshtein_distance("portable", "portablr"), 1)

    def test_single_insertion(self):
        self.assertEqual(_levenshtein_distance("portable", "portables"), 1)

    def test_completely_different_words(self):
        self.assertGreater(_levenshtein_distance("portable", "xyz"), 3)

class TestSpellingCorrection(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_data_neural_bridge"
        os.makedirs(self.test_dir, exist_ok = True)
        self.store_path = os.path.join(self.test_dir, "lexicon.json")
        self.lexicon = LexiconManager(store_path = self.store_path)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_close_typo_is_corrected(self):
        self.assertEqual(suggest_spelling_correction("potrable", self.lexicon), "portable")

    def test_far_off_word_returns_none(self):
        self.assertIsNone(suggest_spelling_correction("xyzzyplugh", self.lexicon))

    def test_exact_match_returns_itself(self):
        self.assertEqual(suggest_spelling_correction("portable", self.lexicon), "portable")

    def test_checks_provisional_lexicon_too(self):
        self.lexicon.induce_word("blick", "verb", [{"name": "MOVE", "category": "action"}])
        self.assertEqual(suggest_spelling_correction("blicc", self.lexicon), "blick")

class TestInductionResponseParsing(unittest.TestCase):
    def test_valid_response(self):
        text = "CATEGORY: verb\nPRIMITIVES: MOVE, BODY"
        self.assertEqual(_parse_induction_response(text), ("verb", ["MOVE", "BODY"]))

    def test_missing_category_is_rejected(self):
        text = "PRIMITIVES: MOVE"
        self.assertIsNone(_parse_induction_response(text))

    def test_missing_primitives_is_rejected(self):
        text = "CATEGORY: verb"
        self.assertIsNone(_parse_induction_response(text))

    def test_invalid_category_word_is_rejected(self):
        text = "CATEGORY: spaceship\nPRIMITIVES: MOVE"
        self.assertIsNone(_parse_induction_response(text))

    def test_extraneous_text_around_the_two_lines_is_tolerated(self):
        text = "Sure, here you go:\nCATEGORY: adjective\nPRIMITIVES: CAN, MOVE\nHope that helps!"
        self.assertEqual(_parse_induction_response(text), ("adjective", ["CAN", "MOVE"]))

class TestFactResponseParsing(unittest.TestCase):
    def test_valid_answer(self):
        self.assertEqual(_parse_fact_response("ANSWER: washington"), "washington")

    def test_unknown_is_none(self):
        self.assertIsNone(_parse_fact_response("ANSWER: UNKNOWN"))

    def test_malformed_response_is_none(self):
        self.assertIsNone(_parse_fact_response("I'm not sure about that."))

    def test_empty_answer_is_none(self):
        self.assertIsNone(_parse_fact_response("ANSWER:"))

class TestImplicatureResponseParsing(unittest.TestCase):
    def test_valid_action_and_target(self):
        text = "ACTION: CLOSE\nTARGET: window"
        self.assertEqual(_parse_implicature_response(text), {"action": "CLOSE", "target": "window"})

    def test_none_action_is_rejected(self):
        text = "ACTION: NONE\nTARGET: NONE"
        self.assertIsNone(_parse_implicature_response(text))

    def test_action_outside_closed_vocabulary_is_rejected(self):
        text = "ACTION: LAUNCH_ROCKET\nTARGET: window"
        self.assertIsNone(_parse_implicature_response(text))

    def test_missing_target_is_rejected(self):
        text = "ACTION: CLOSE\nTARGET: NONE"
        self.assertIsNone(_parse_implicature_response(text))

class TestNeuralBridgeWithoutModel(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_data_neural_bridge_no_model"
        os.makedirs(self.test_dir, exist_ok = True)
        self.store_path = os.path.join(self.test_dir, "lexicon.json")
        self.lexicon = LexiconManager(store_path = self.store_path)
        self.world_model = WorldModel()
        self.bridge = NeuralBridge(llm = None)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_induce_word_returns_none_without_a_model(self):
        self.assertIsNone(self.bridge.induce_word("blick", "John did blick.", self.lexicon))

    def test_lookup_fact_returns_none_without_a_model(self):
        self.assertIsNone(self.bridge.lookup_fact(FrameTemplate.AT_PLACE, "seattle", self.world_model))

    def test_infer_indirect_speech_act_returns_none_without_a_model(self):
        self.assertIsNone(self.bridge.infer_indirect_speech_act("It's freezing in here."))

class TestNeuralBridgeWordInduction(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_data_neural_bridge_induce"
        os.makedirs(self.test_dir, exist_ok = True)
        self.store_path = os.path.join(self.test_dir, "lexicon.json")
        self.lexicon = LexiconManager(store_path = self.store_path)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_valid_proposal_gets_registered_provisionally(self):
        bridge = NeuralBridge(llm = _FakeLLM("CATEGORY: verb\nPRIMITIVES: MOVE, BODY"))
        result = bridge.induce_word("blick", "John did blick across the room.", self.lexicon)
        self.assertIsNotNone(result)
        self.assertIn("blick", self.lexicon.provisional_lexicon)
        self.assertNotIn("blick", self.lexicon.lexicon) # Never auto-promoted.

    def test_invalid_primitive_proposal_is_refused(self):
        bridge = NeuralBridge(llm = _FakeLLM("CATEGORY: verb\nPRIMITIVES: NOT_A_REAL_PRIMITIVE"))
        result = bridge.induce_word("blick", "John did blick.", self.lexicon)
        self.assertIsNone(result)
        self.assertNotIn("blick", self.lexicon.provisional_lexicon)

    def test_malformed_response_is_refused(self):
        bridge = NeuralBridge(llm = _FakeLLM("I have no idea what that word means."))
        result = bridge.induce_word("blick", "John did blick.", self.lexicon)
        self.assertIsNone(result)

    def test_broken_model_call_is_handled_gracefully(self):
        bridge = NeuralBridge(llm = _BrokenLLM())
        result = bridge.induce_word("blick", "John did blick.", self.lexicon)
        self.assertIsNone(result)

class TestNeuralBridgeFactLookup(unittest.TestCase):
    def setUp(self):
        self.world_model = WorldModel()

    def test_valid_answer_is_recorded_as_provisional(self):
        bridge = NeuralBridge(llm = _FakeLLM("ANSWER: washington"))
        result = bridge.lookup_fact(FrameTemplate.AT_PLACE, "seattle", self.world_model)
        self.assertIsNotNone(result)
        self.assertEqual(result["provenance"], FactProvenance.PROVISIONAL)

    def test_unknown_answer_records_nothing(self):
        bridge = NeuralBridge(llm = _FakeLLM("ANSWER: UNKNOWN"))
        result = bridge.lookup_fact(FrameTemplate.AT_PLACE, "seattle", self.world_model)
        self.assertIsNone(result)
        self.assertIsNone(self.world_model.query_episodic_fact(FrameTemplate.AT_PLACE, "seattle"))

class TestNeuralBridgeIndirectSpeechAct(unittest.TestCase):
    def test_detected_implicature(self):
        bridge = NeuralBridge(llm = _FakeLLM("ACTION: CLOSE\nTARGET: window"))
        result = bridge.infer_indirect_speech_act("It's freezing in here.")
        self.assertEqual(result, {"action": "CLOSE", "target": "window"})

    def test_no_implicature_detected(self):
        bridge = NeuralBridge(llm = _FakeLLM("ACTION: NONE\nTARGET: NONE"))
        result = bridge.infer_indirect_speech_act("The suitcase is portable.")
        self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()