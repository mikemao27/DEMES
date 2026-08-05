"""
Integration tests for core/pipeline.py: real end-to-end turns exercising how the standalone, individually-tested mechanisms across the rest of the codebase 
actually chain together.
"""

import unittest
import os
import shutil

from core.pipeline import DEMESPipeline
from core.types import FrameTemplate

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
        for key in ("raw_text", "logical_form", "semantics", "speech_act", "word_senses", "implicatures", "accommodated_presuppositions", "focus", "cataphora_resolutions", "modal_context", "qud_entry"):
            self.assertIn(key, result)

    def test_declarative_evaluates_true(self):
        result = self.pipeline.process_utterance("The suitcase is portable.")
        self.assertEqual(result["semantics"]["status"], "success")
        self.assertTrue(result["semantics"]["truth_value"])

    def test_turn_clock_and_event_log_advance(self):
        self.pipeline.process_utterance("The suitcase is portable.")
        self.assertEqual(self.pipeline.world_model.current_turn, 1)
        self.assertEqual(len(self.pipeline.world_model.event_log), 1)

    def test_pluperfect_sentence_records_a_real_pluperfect_event(self):
        # Phase 4, Sub-step C3: "had" + participle reaches core/world_model.py's EventRecord.is_pluperfect() from a real parse for the first time.
        result = self.pipeline.process_utterance("John had kicked the bucket.")
        self.assertEqual(result["logical_form"].tense, "pluperfect")
        self.assertTrue(self.pipeline.world_model.event_log[-1].is_pluperfect())

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

class TestLinksLatticeIntegration(TestPipelineBase):
    """
    Phase 4, Sub-step C2: an NP-coordinated subject ("John and Mary walked") gets joined into a real Link's-lattice sum entity with actual members,
    reachable from a real parse for the first time, core/world_model.py's join_entities/holds_collectively/holds_distributively were already correct
    and tested standalone, but always had to be constructed by hand since a bare plural mention alone never enumerates real individuals.
    """
    def setUp(self):
        super().setUp()
        self.pipeline.lexicon.lexicon["mary"] = {"category": "proper_noun", "semantic_type": "e", "primitives": [], "valency": "none"}

    def test_coordinated_subject_is_joined_into_a_real_sum_entity(self):
        self.pipeline.process_utterance("John and Mary walked.")
        entity = self.pipeline.world_model.entities.get("john + mary")
        self.assertIsNotNone(entity)
        self.assertTrue(entity.is_sum)
        self.assertEqual(entity.members, ["john", "mary"])

    def test_the_new_sum_entity_is_usable_by_holds_distributively(self):
        self.pipeline.process_utterance("John and Mary walked.")
        entity = self.pipeline.world_model.entities["john + mary"]
        self.pipeline.world_model.knowledge_base["walked"] = ["john", "mary"]
        self.assertTrue(self.pipeline.world_model.holds_distributively("walked", entity))

    def test_ordinary_non_coordinated_sentence_creates_no_sum_entity(self):
        self.pipeline.process_utterance("John walked.")
        self.assertEqual(self.pipeline.world_model.entities, {})

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

    def test_real_multi_turn_facts_make_the_quantified_sentence_true(self):
        # Fully natural, no manual world-model seeding: two ordinary declarative turns record real relational facts (Sub-step A1), which the
        # quantified sentence's scoped evaluation (Sub-step A3) then genuinely checks against.
        self.pipeline.lexicon.lexicon["alice"] = {"category": "proper_noun", "semantic_type": "e", "primitives": [], "valency": "none"}
        self.pipeline.lexicon.lexicon["bob"] = {"category": "proper_noun", "semantic_type": "e", "primitives": [], "valency": "none"}
        self.pipeline.world_model.knowledge_base["student"] = ["alice", "bob"]
        self.pipeline.world_model.knowledge_base["book"] = ["book"]

        self.pipeline.process_utterance("Alice read the book.")
        self.pipeline.process_utterance("Bob read the book.")

        result = self.pipeline.process_utterance("Every student read a book.")
        self.assertTrue(result["semantics"]["truth_value"])
        self.assertEqual(result["semantics"]["quantifier_scope"]["reading_count"], 2)

    def test_real_multi_turn_facts_make_the_quantified_sentence_false(self):
        self.pipeline.lexicon.lexicon["alice"] = {"category": "proper_noun", "semantic_type": "e", "primitives": [], "valency": "none"}
        self.pipeline.world_model.knowledge_base["student"] = ["alice", "bob"] # Bob never gets his own turn below.
        self.pipeline.world_model.knowledge_base["book"] = ["book"]

        self.pipeline.process_utterance("Alice read the book.")

        result = self.pipeline.process_utterance("Every student read a book.")
        self.assertFalse(result["semantics"]["truth_value"])

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
    """
    Phase 5, Sub-step D1: focus detection is now derivation-tree-based (core/parser.py's find_focused_constituent), reached via process_utterance's
    real derivation_tree rather than a raw-token scan disconnected from whether the sentence actually parsed.
    """
    def test_sentence_initial_focus_particle_identifies_the_focused_word(self):
        # "Only John walked." (not "left": "left" isn't in core/lexicon.py's irregular-forms table yet, a separate, already-flagged gap): this shape
        # genuinely didn't parse at all before this sub-step (focus particles had no sentence-initial candidate category).
        result = self.pipeline.process_utterance("Only John walked.")
        self.assertIsNotNone(result["logical_form"])
        self.assertEqual(result["focus"], "john")

    def test_mid_sentence_focus_particle_identifies_the_focused_word(self):
        result = self.pipeline.process_utterance("John only walked.")
        self.assertEqual(result["focus"], "walked")

    def test_unparsable_sentence_reports_no_focus_even_with_a_particle_present(self):
        # The bug this locks in: the old raw-token detect_focus reported a confident "john" here even though this exact sentence never parsed at all
        # (verified live during this sub-step's own design): the new tree-based version correctly reports nothing for a sentence DEMES didn't understand.
        result = self.pipeline.process_utterance("Only John left.")
        self.assertIsNone(result["logical_form"])
        self.assertIsNone(result["focus"])

class TestModalAttitudeIntegration(TestPipelineBase):
    """
    Phase 2.3's actual deliverable: a clausal-complement mental-predicate verb ("thinks") auto-opens a Modal & Attitude context (core/world_model.py, built and tested 
    standalone several files ago) for its subject and asserts the embedded clause into that context alone: never into global reality. This is the live-pipeline version 
    of the "John thinks Mary is at home, but she isn't" scenario already proven at the world_model.py unit level.
    """
    def setUp(self):
        super().setUp()
        self.pipeline.lexicon.lexicon["mary"] = {"category": "proper_noun", "semantic_type": "e", "primitives": [], "valency": "none"}
        self.pipeline.lexicon.lexicon["think"] = {"category": "verb", "semantic_type": "<s,<e,t>>", "primitives": [{"name": "THINK", "category": "mental"}], "valency": "clausal"}
        self.pipeline.lexicon.lexicon["home"] = {"category": "adjective", "semantic_type": "<e,t>", "primitives": [{"name": "SOMEWHERE", "category": "entity"}], "valency": "none"}
        self.pipeline.lexicon.lexicon["that"] = {"category": "function", "semantic_type": "None", "primitives": [], "valency": "none"}

    def test_clausal_complement_opens_an_epistemic_context_for_the_matrix_subject(self):
        result = self.pipeline.process_utterance("John thinks that Mary is home.")
        self.assertIsNotNone(result["modal_context"])
        self.assertEqual(result["modal_context"]["holder"], "john")
        self.assertEqual(result["modal_context"]["modal_flavor"], "epistemic")

    def test_embedded_belief_is_isolated_from_global_knowledge(self):
        self.pipeline.process_utterance("John thinks that Mary is home.")
        self.assertNotIn("home", self.pipeline.world_model.knowledge_base)

    def test_embedded_belief_is_recorded_in_its_own_context(self):
        result = self.pipeline.process_utterance("John thinks that Mary is home.")
        context_id = result["modal_context"]["context_id"]
        self.assertIn("mary", self.pipeline.world_model.context_facts[context_id]["home"])

    def test_a_contradicting_global_assertion_is_unaffected_by_the_belief_context(self):
        self.pipeline.process_utterance("John thinks that Mary is home.")
        result = self.pipeline.process_utterance("Mary is not home.")
        self.assertFalse(result["semantics"]["truth_value"])

    def test_ordinary_flat_sentences_open_no_modal_context(self):
        result = self.pipeline.process_utterance("The suitcase is portable.")
        self.assertIsNone(result["modal_context"])

class TestQUDIntegration(TestPipelineBase):
    """
    Phase 2.6's actual deliverable: core/discourse.py's QUDStack (correct and tested standalone since Phase 1) auto-populates from a real wh-question
    parse for the first time, rather than only working when a QUDEntry is constructed by hand.
    """
    def test_subject_wh_question_pushes_a_qud_entry(self):
        result = self.pipeline.process_utterance("Who walked?")
        self.assertEqual(result["qud_entry"], {"predicate": "WALKED", "arguments": ["who"], "open_slot_index": 0})
        self.assertEqual(self.pipeline.qud_stack.current().predicate, "WALKED")

    def test_inverted_copular_question_pushes_a_qud_entry(self):
        result = self.pipeline.process_utterance("What is the suitcase?")
        self.assertEqual(result["qud_entry"], {"predicate": "IS", "arguments": ["what", "suitcase"], "open_slot_index": 0})

    def test_ordinary_declarative_does_not_touch_the_qud_stack(self):
        result = self.pipeline.process_utterance("The suitcase is portable.")
        self.assertIsNone(result["qud_entry"])
        self.assertIsNone(self.pipeline.qud_stack.current())

    def test_qud_stack_persists_the_most_recent_question_across_turns(self):
        self.pipeline.process_utterance("Who walked?")
        self.pipeline.process_utterance("The suitcase is portable.")
        self.assertEqual(self.pipeline.qud_stack.current().predicate, "WALKED")

class TestFragmentAnswerIntegration(TestPipelineBase):
    """
    Phase 5, Sub-step D2: a bare fragment answer ("John.") to an open wh-question is resolved via QUDStack.resolve_fragment instead of being treated
    as its own free-standing (and unparseable) sentence, then runs through the entire rest of the ordinary pipeline unchanged.
    """
    def test_fragment_answer_resolves_and_evaluates(self):
        self.pipeline.process_utterance("Who walked?")
        result = self.pipeline.process_utterance("John.")
        self.assertTrue(result["resolved_from_fragment"])
        self.assertEqual(result["logical_form"].predicate, "WALKED")
        self.assertEqual(result["logical_form"].arguments, ["john"])
        self.assertEqual(result["semantics"]["status"], "success")

    def test_ordinary_sentence_is_not_marked_as_resolved_from_a_fragment(self):
        result = self.pipeline.process_utterance("The suitcase is portable.")
        self.assertFalse(result["resolved_from_fragment"])

    def test_a_genuinely_unparsable_utterance_with_no_open_question_still_fails(self):
        result = self.pipeline.process_utterance("Quantum mechanics is weird")
        self.assertFalse(result["resolved_from_fragment"])
        self.assertIsNone(result["logical_form"])

    def test_resolved_question_cannot_be_answered_twice(self):
        self.pipeline.process_utterance("Who walked?")
        self.pipeline.process_utterance("John.")
        result = self.pipeline.process_utterance("Mary.")
        self.assertFalse(result["resolved_from_fragment"])
        self.assertIsNone(result["logical_form"])

class TestCoordinationIntegration(TestPipelineBase):
    """
    Phase 2.4's actual deliverable: a coordinated sentence's conjuncts are each registered, resolved, and evaluated independently, then combined into
    one turn-level truth value via the coordinator's own connective: ordinary propositional AND/OR, not a new evaluation mechanism.
    """
    def setUp(self):
        super().setUp()
        self.pipeline.lexicon.lexicon["walk"] = {"category": "verb", "semantic_type": "<e,t>", "primitives": [{"name": "MOVE", "category": "action"}], "valency": "intransitive"}

    def test_and_coordination_is_true_only_when_both_conjuncts_are_true(self):
        result = self.pipeline.process_utterance("The suitcase is portable and John walked.")
        self.assertIsInstance(result["logical_form"], list)
        self.assertEqual(result["semantics"]["connective"], "AND")
        self.assertTrue(result["semantics"]["truth_value"])

    def test_and_coordination_is_false_when_one_conjunct_is_false(self):
        result = self.pipeline.process_utterance("The suitcase is not portable and John walked.")
        self.assertEqual(result["semantics"]["connective"], "AND")
        self.assertFalse(result["semantics"]["truth_value"])

    def test_or_coordination_is_true_when_only_one_conjunct_is_true(self):
        result = self.pipeline.process_utterance("The suitcase is not portable or John walked.")
        self.assertEqual(result["semantics"]["connective"], "OR")
        self.assertTrue(result["semantics"]["truth_value"])

    def test_each_conjunct_is_available_in_the_combined_payload(self):
        result = self.pipeline.process_utterance("The suitcase is portable and John walked.")
        self.assertEqual(len(result["semantics"]["conjuncts"]), 2)

    def test_pronoun_resolution_does_not_cross_between_conjuncts(self):
        # "It" must resolve against the world model's active referents (backward resolution, from the earlier turn below), never forward to "John" in
        # the other conjunct: two coordinated independent clauses aren't in the kind of subordinate relationship real cataphora requires.
        self.pipeline.lexicon.lexicon["heavy"] = {"category": "adjective", "semantic_type": "<e,t>", "primitives": [{"name": "BIG", "category": "property"}], "valency": "none"}
        self.pipeline.process_utterance("The suitcase is heavy.")
        result = self.pipeline.process_utterance("It is heavy and John walked.")
        self.assertEqual(result["logical_form"][0].arguments, ["suitcase"])
        self.assertEqual(result["cataphora_resolutions"], [])

    def test_ordinary_non_coordinated_sentences_are_unaffected(self):
        result = self.pipeline.process_utterance("The suitcase is portable.")
        self.assertNotIsInstance(result["logical_form"], list)
        self.assertEqual(result["semantics"]["truth_value"], True)

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

class TestFigurativeRepairIntegration(TestPipelineBase):
    """
    Phase 3, Sub-step B1: metonymic coercion and metaphor mapping (core/figurative.py, correct and tested standalone since the original build)
    reachable and inspectable from a real turn for the first time, via two new opt-in lexicon fields (selectional_requirements, literal_type).
    Present tense throughout: "leave"'s past-tense form "left" isn't in core/lexicon.py's irregular-forms table yet (a separate, pre-existing,
    already-flagged gap), so present tense is what actually exercises this without tripping over it.
    """
    def setUp(self):
        super().setUp()
        self.pipeline.lexicon.lexicon["john"]["literal_type"] = "PERSON"
        self.pipeline.lexicon.lexicon["leave"] = {
            "category": "verb", "semantic_type": "<e,t>", "primitives": [{"name": "MOVE", "category": "action"}],
            "valency": "intransitive", "selectional_requirements": {"subject": "PERSON"},
        }
        self.pipeline.lexicon.lexicon["sandwich"] = {
            "category": "noun", "semantic_type": "e", "primitives": [{"name": "SOMETHING", "category": "entity"}],
            "valency": "none", "metonymy_licenses": ["container_for_person"],
        }
        self.pipeline.lexicon.lexicon["rock"] = {"category": "noun", "semantic_type": "e", "primitives": [{"name": "SOMETHING", "category": "entity"}], "valency": "none"}
        self.pipeline.lexicon.lexicon["digest"] = {
            "category": "verb", "semantic_type": "<e,<e,t>>", "primitives": [{"name": "DO", "category": "action"}],
            "valency": "transitive", "selectional_requirements": {"object": "INFORMATION"},
        }
        self.pipeline.lexicon.lexicon["book"] = {
            "category": "noun", "semantic_type": "e", "primitives": [{"name": "SOMETHING", "category": "entity"}],
            "valency": "none", "conceptual_domain": "information",
        }

    def test_metonymy_is_reachable_from_a_real_parse(self):
        result = self.pipeline.process_utterance("The sandwich leaves.")
        self.assertEqual(result["semantics"]["figurative_repair"], {
            "argument_position": "subject", "argument": "sandwich", "required_type": "PERSON", "resolution": "COERCED:PERSON",
        })

    def test_literal_subject_that_already_satisfies_the_requirement_has_no_repair(self):
        result = self.pipeline.process_utterance("John leaves.")
        self.assertNotIn("figurative_repair", result["semantics"])

    def test_metaphor_is_reachable_from_a_real_parse(self):
        result = self.pipeline.process_utterance("John digested the book.")
        self.assertEqual(result["semantics"]["figurative_repair"]["resolution"], "COMPREHEND")

    def test_genuinely_ungrounded_argument_gets_no_repair(self):
        # "rock" licenses nothing: the closed-registry discipline holds through the live pipeline, not just figurative.py's own unit tests.
        result = self.pipeline.process_utterance("The rock leaves.")
        self.assertNotIn("figurative_repair", result["semantics"])

    def test_predicate_with_no_declared_requirement_is_untouched(self):
        result = self.pipeline.process_utterance("John kicked the bucket.")
        self.assertNotIn("figurative_repair", result["semantics"])

    def test_repair_never_changes_the_truth_value(self):
        # Purely an annotation, the same non-invasive precedent idiom attachment already set: does not redirect evaluation.
        result = self.pipeline.process_utterance("The sandwich leaves.")
        self.assertIn("figurative_repair", result["semantics"])
        self.assertTrue(result["semantics"]["truth_value"]) # "leave" is untracked in knowledge_base: permissive default, unaffected by the repair.

class TestEpisodicFactGraphIntegration(TestPipelineBase):
    """
    Phase 4, Sub-step C1: the copula-identity-relation shape ("Bob is an employee") is the only sentence shape core/parser.py can currently produce
    that naturally maps onto one of the Episodic Fact Graph's closed FrameTemplate relations: reachable and inspectable from a real turn for the
    first time, via a new IS_A write-then-query consultation in core/pipeline.py.
    """
    def setUp(self):
        super().setUp()
        self.pipeline.lexicon.lexicon["bob"] = {"category": "proper_noun", "semantic_type": "e", "primitives": [], "valency": "none"}
        self.pipeline.lexicon.lexicon["employee"] = {"category": "noun", "semantic_type": "e", "primitives": [{"name": "PEOPLE", "category": "entity"}], "valency": "none"}
        self.pipeline.lexicon.lexicon["manager"] = {"category": "noun", "semantic_type": "e", "primitives": [{"name": "PEOPLE", "category": "entity"}], "valency": "none"}

    def test_first_mention_is_written_and_flagged_new(self):
        result = self.pipeline.process_utterance("Bob is an employee.")
        self.assertEqual(result["semantics"]["episodic_fact"], {
            "relation": "IS-A", "subject": "bob", "object": "employee", "already_known": False,
        })

    def test_fact_is_actually_queryable_afterward(self):
        self.pipeline.process_utterance("Bob is an employee.")
        fact = self.pipeline.world_model.query_episodic_fact(FrameTemplate.IS_A, "bob", "employee")
        self.assertIsNotNone(fact)

    def test_repeated_mention_is_flagged_already_known(self):
        self.pipeline.process_utterance("Bob is an employee.")
        result = self.pipeline.process_utterance("Bob is an employee.")
        self.assertTrue(result["semantics"]["episodic_fact"]["already_known"])

    def test_a_different_object_for_the_same_subject_is_still_new(self):
        # IS_A isn't exclusive: "Bob is an employee" and "Bob is a manager" can both independently be true and both get recorded.
        self.pipeline.process_utterance("Bob is an employee.")
        result = self.pipeline.process_utterance("Bob is a manager.")
        self.assertFalse(result["semantics"]["episodic_fact"]["already_known"])
        self.assertIsNotNone(self.pipeline.world_model.query_episodic_fact(FrameTemplate.IS_A, "bob", "employee"))
        self.assertIsNotNone(self.pipeline.world_model.query_episodic_fact(FrameTemplate.IS_A, "bob", "manager"))

    def test_ordinary_verb_predicate_is_untouched(self):
        result = self.pipeline.process_utterance("John kicked the bucket.")
        self.assertNotIn("episodic_fact", result["semantics"])

    def test_truth_value_is_unaffected_by_the_consultation(self):
        # Purely an annotation, the same non-invasive precedent idiom/figurative-repair attachment already set.
        result = self.pipeline.process_utterance("Bob is an employee.")
        self.assertIn("episodic_fact", result["semantics"])
        self.assertTrue(result["semantics"]["truth_value"])

class TestRelationalFactIntegration(TestPipelineBase):
    """
    Phase 3, Sub-step A1: an ordinary true two-argument claim leaves behind a real binary fact (core/world_model.py's relational_facts), so a later
    multi-quantifier sentence has something real to check its relation against instead of staying permissively true forever.
    """
    def setUp(self):
        super().setUp()
        # "bucket" is idiom-tagged when kicked (see TestIdiomIntegration above): a plain, non-idiomatic object is needed here instead.
        self.pipeline.lexicon.lexicon["ball"] = {"category": "noun", "semantic_type": "e", "primitives": [{"name": "SOMETHING", "category": "entity"}], "valency": "none"}

    def test_true_transitive_claim_is_recorded(self):
        self.pipeline.process_utterance("John kicked the ball.")
        self.assertIn(("john", "ball"), self.pipeline.world_model.relational_facts.get("kicked", []))

    def test_negated_claim_is_not_recorded(self):
        result = self.pipeline.process_utterance("John did not kick the ball.")
        self.assertFalse(result["semantics"]["truth_value"])
        self.assertNotIn("kicked", self.pipeline.world_model.relational_facts)

    def test_single_argument_sentence_is_not_recorded(self):
        self.pipeline.process_utterance("The suitcase is portable.")
        self.assertEqual(self.pipeline.world_model.relational_facts, {})

class TestSDRTIntegration(TestPipelineBase):
    """
    Phase 5, Sub-step D4: a sentence-initial discourse connective ("However, Mary stayed.") is stripped before parsing, and (once this isn't the
    conversation's first turn) classified against the previous turn via core/discourse.py's connective-based half of classify_rhetorical_relation.
    """
    def setUp(self):
        super().setUp()
        self.pipeline.lexicon.lexicon["mary"] = {"category": "proper_noun", "semantic_type": "e", "primitives": [], "valency": "none"}
        self.pipeline.lexicon.lexicon["leave"] = {"category": "verb", "semantic_type": "<e, t>", "primitives": [{"name": "DO", "category": "action"}], "valency": "intransitive"}
        self.pipeline.lexicon.lexicon["stay"] = {"category": "verb", "semantic_type": "<e, t>", "primitives": [{"name": "DO", "category": "action"}], "valency": "intransitive"}

    def test_connective_after_a_prior_turn_is_classified(self):
        self.pipeline.process_utterance("John leaves.")
        result = self.pipeline.process_utterance("However, Mary stays.")
        self.assertEqual(result["rhetorical_relation"], "contrast")

    def test_connective_as_the_very_first_turn_yields_no_relation(self):
        result = self.pipeline.process_utterance("However, Mary stays.")
        self.assertIsNone(result["rhetorical_relation"])

    def test_connective_stripped_sentence_still_evaluates_its_own_content(self):
        self.pipeline.process_utterance("John leaves.")
        result = self.pipeline.process_utterance("However, Mary stays.")
        self.assertEqual(result["logical_form"].predicate, "STAYS")
        self.assertEqual(result["logical_form"].arguments, ["mary"])

    def test_raw_text_output_keeps_the_original_untouched_text(self):
        self.pipeline.process_utterance("John leaves.")
        result = self.pipeline.process_utterance("However, Mary stays.")
        self.assertEqual(result["raw_text"], "However, Mary stays.")

    def test_ordinary_sentence_without_a_connective_has_no_relation(self):
        self.pipeline.process_utterance("John leaves.")
        result = self.pipeline.process_utterance("Mary stays.")
        self.assertIsNone(result["rhetorical_relation"])

if __name__ == "__main__":
    unittest.main()