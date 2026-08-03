"""
Integration tests for core/pipeline.py: real end-to-end turns exercising how the standalone, individually-tested mechanisms across the rest of the codebase 
actually chain together.
"""

import unittest
import os
import shutil

from core.pipeline import DEMESPipeline

class TestPipelineBase(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_data_pipeline"
        os.makedirs(self.test_dir, exist_ok = True)
        self.store_path = os.path.join(self.test_dir, "lexicon.json")
        self.pipeline = DEMESPipeline(lexicon_path = self.store_path)

        # Extra vocabulary for the sentence shapes under test.
        extra = {
            "john": {"category": "proper_noun", "semantic_type": "e", "primitives": [], "valency": "none"},
            "stop": {"category": "verb", "semantic_type": "<e, t>", "primitives": [{"name": "NOT", "category": "logical"}], "valency": "intransitive"},
            "kick": {"category": "verb", "semantic_type": "<e,<e,t>>", "primitives": [{"name": "DO", "category": "action"}], "valency": "transitive"},
            "bucket": {"category": "noun", "semantic_type": "e", "primitives": [{"name": "SOMETHING", "category": "entity"}], "valency": "none"},
            "heavy": {"category": "adjective", "semantic_type": "<e, t>", "primitives": [{"name": "BIG", "category": "property"}], "valency": "none"},
            "bank": {
                "category": "noun", "semantic_type": "e", "primitives": [], "valency": "none",
                "senses": [
                    {"sense_key": "bank.n.financial_institution", "selectional_constraint": "INSTITUTION"},
                    {"sense_key": "bank.n.river_side", "selectional_constraint": "GEOGRAPHICAL_FEATURE"},
                ],
            },
            "river": {
                "category": "noun", "semantic_type": "e", "primitives": [{"name": "SOMETHING", "category": "entity"}], "valency": "none",
                "senses": [{"sense_key": "river.n.waterway", "selectional_constraint": "GEOGRAPHICAL_FEATURE"}],
            },
        }
        self.pipeline.lexicon.lexicon.update(extra)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

class TestBasicTurn(TestPipelineBase):
    def test_payload_shape(self):
        result = self.pipeline.process_utterance("The suitcase is portable.")
        for key in ("raw_text", "logical_form", "semantics", "speech_act", "word_senses", "implicatures", "accommodated_presuppositions", "focus"):
            self.assertIn(key, result)

    def test_declarative_evaluates_true(self):
        result = self.pipeline.process_utterance("The suitcase is portable.")
        self.assertEqual(result["semantics"]["status"], "success")
        self.assertTrue(result["semantics"]["truth_value"])

    def test_turn_clock_and_event_log_advance(self):
        self.pipeline.process_utterance("The suitcase is portable.")
        self.assertEqual(self.pipeline.world_model.current_turn, 1)
        self.assertEqual(len(self.pipeline.world_model.event_log), 1)

    def test_unparsable_sentence_fails_gracefully(self):
        result = self.pipeline.process_utterance("Quantum mechanics is weird")
        self.assertIsNone(result["logical_form"])
        self.assertEqual(result["semantics"]["status"], "failure")

    def test_negation_flows_through_the_whole_pipeline(self):
        result = self.pipeline.process_utterance("The suitcase is not portable.")
        self.assertTrue(result["logical_form"].is_negated)
        self.assertFalse(result["semantics"]["truth_value"])

class TestNounRegistrationAndPronounResolution(TestPipelineBase):
    def test_noun_mention_is_registered_as_a_discourse_referent(self):
        self.pipeline.process_utterance("The suitcase is heavy.")
        names = [ref.name for ref in self.pipeline.world_model.active_referents.values()]
        self.assertIn("suitcase", names)

    def test_pronoun_in_a_later_turn_resolves_to_the_earlier_noun(self):
        self.pipeline.process_utterance("The suitcase is heavy.")
        result = self.pipeline.process_utterance("It is portable.")
        self.assertEqual(result["logical_form"].arguments, ["suitcase"])

    def test_pronoun_with_no_antecedent_stays_as_the_literal_word(self):
        result = self.pipeline.process_utterance("It is portable.")
        self.assertEqual(result["logical_form"].arguments, ["it"])

    def test_plural_noun_mention_is_registered_as_a_sum_entity(self):
        self.pipeline.process_utterance("The suitcases is portable.")
        entity = self.pipeline.world_model.entities.get("suitcases")
        self.assertIsNotNone(entity)
        self.assertTrue(entity.is_sum)

    def test_singular_noun_mention_is_not_registered_as_an_entity(self):
        self.pipeline.process_utterance("The suitcase is portable.")
        self.assertNotIn("suitcase", self.pipeline.world_model.entities)

class TestCataphoraIntegration(TestPipelineBase):
    """
    The whole point of Phase 2.2: cataphora resolution (core/discourse.py, built and tested against hand-built trees several files ago) actually reachable 
    from a real, live parse now that adjunct clauses exist. "Before he entered the room, John walked" is grammatical for the intended coreference 
    (the pronoun doesn't c-command the name); "John walked before he entered the room" is not (it does): the same licensed/blocked pair Principle C predicts.
    """
    def setUp(self):
        super().setUp()
        self.pipeline.lexicon.lexicon["enter"] = {"category": "verb", "semantic_type": "<e,<e,t>>", "primitives": [{"name": "MOVE", "category": "action"}], "valency": "transitive"}
        self.pipeline.lexicon.lexicon["room"] = {"category": "noun", "semantic_type": "e", "primitives": [{"name": "SOMETHING", "category": "entity"}], "valency": "none"}

    def test_fronted_adjunct_licenses_cataphora(self):
        result = self.pipeline.process_utterance("Before he entered the room, John walked.")
        self.assertEqual(result["logical_form"].arguments, ["john"])
        self.assertEqual(result["cataphora_resolutions"], [{"pronoun": "he", "resolved_to": "john"}])

    def test_trailing_adjunct_blocks_cataphora(self):
        # "He" is the matrix subject here and structurally c-commands "John" inside the trailing adjunct, so forward reference is blocked: unlike the fronted case above, 
        # where the pronoun sits inside the (non-c-commanding) adjunct instead.
        result = self.pipeline.process_utterance("He walked before John entered the room.")
        self.assertEqual(result["logical_form"].arguments, ["he"])
        self.assertEqual(result["cataphora_resolutions"], [])

class TestMultiQuantifierIntegration(TestPipelineBase):
    def setUp(self):
        super().setUp()
        self.pipeline.lexicon.lexicon["student"] = {"category": "noun", "semantic_type": "e", "primitives": [{"name": "PEOPLE", "category": "entity"}], "valency": "none"}
        self.pipeline.lexicon.lexicon["book"] = {"category": "noun", "semantic_type": "e", "primitives": [{"name": "SOMETHING", "category": "entity"}], "valency": "none"}
        self.pipeline.lexicon.lexicon["read"] = {"category": "verb", "semantic_type": "<e,<e,t>>", "primitives": [{"name": "SEE", "category": "action"}], "valency": "transitive"}

    def test_two_quantifiers_reach_the_pipeline_output_intact(self):
        result = self.pipeline.process_utterance("Every student read a book.")
        form = result["logical_form"]
        self.assertEqual(form.predicate, "READ")
        self.assertEqual(len(form.quantifier_store), 2)
        self.assertIsNone(form.quantifier_meta)

class TestWordSenseDisambiguationIntegration(TestPipelineBase):
    def test_bank_resolves_to_river_sense_near_a_geographical_sibling(self):
        result = self.pipeline.process_utterance("The bank is near the river.")
        self.assertEqual(result["word_senses"].get("bank"), "bank.n.river_side")

class TestSpeechActIntegration(TestPipelineBase):
    def test_greeting_is_expressive(self):
        result = self.pipeline.process_utterance("Hello")
        self.assertEqual(result["speech_act"], "expressive")

    def test_question_is_directive(self):
        result = self.pipeline.process_utterance("Is the suitcase portable?")
        self.assertEqual(result["speech_act"], "directive")

    def test_declarative_is_assertive(self):
        result = self.pipeline.process_utterance("The suitcase is portable.")
        self.assertEqual(result["speech_act"], "assertive")

class TestPresuppositionAccommodationIntegration(TestPipelineBase):
    def test_stop_trigger_is_accommodated_on_first_mention(self):
        result = self.pipeline.process_utterance("John stopped.")
        self.assertIn("stop", result["accommodated_presuppositions"])
        self.assertTrue(self.pipeline.world_model.check_presupposition("stop", "john"))

    def test_second_mention_is_not_re_accommodated(self):
        self.pipeline.process_utterance("John stopped.")
        result = self.pipeline.process_utterance("John stopped.")
        self.assertEqual(result["accommodated_presuppositions"], [])

class TestScalarImplicatureIntegration(TestPipelineBase):
    def test_some_generates_a_not_all_implicature(self):
        result = self.pipeline.process_utterance("Some students passed.")
        self.assertIn("NOT_ALL", result["implicatures"])

class TestFocusIntegration(TestPipelineBase):
    def test_focus_particle_identifies_the_focused_word(self):
        result = self.pipeline.process_utterance("Only John left.")
        self.assertEqual(result["focus"], "john")

class TestIdiomIntegration(TestPipelineBase):
    def test_idiom_meaning_is_attached_to_the_payload(self):
        result = self.pipeline.process_utterance("John kicked the bucket.")
        self.assertIn("idiom_meaning", result["semantics"])
        self.assertEqual(result["semantics"]["idiom_meaning"]["slots"]["event"], "DIE")

    def test_literal_use_has_no_idiom_meaning_attached(self):
        result = self.pipeline.process_utterance("John kicked the bucket.")
        self.assertIn("idiom_meaning", result["semantics"])

        # Compare against a literal object that wouldn't trigger the idiom's exact object trigger.
        self.pipeline.lexicon.lexicon["ball"] = {"category": "noun", "semantic_type": "e", "primitives": [{"name": "SOMETHING", "category": "entity"}], "valency": "none"}
        literal_result = self.pipeline.process_utterance("John kicked the ball.")
        self.assertNotIn("idiom_meaning", literal_result["semantics"])

if __name__ == "__main__":
    unittest.main()