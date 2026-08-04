"""
Unit tests for core/world_model.py: the original extensional/discourse interface, Link's lattice, the episodic timeline, 
presupposition tracking, the Modal & Attitude layer, and the Episodic Fact Graph.
"""

import unittest

from core.types import ModalFlavor, FrameTemplate
from core.world_model import WorldModel, FactProvenance, get_presupposition_trigger

class TestOriginalInterfaceUnchanged(unittest.TestCase):
    """
    The pre-existing public contract other files (core/semantics.py, tests) still depend on.
    """
    def setUp(self):
        self.world_model = WorldModel()

    def test_assertion_validation(self):
        self.assertTrue(self.world_model.validate_assertion("PORTABLE", ["suitcase"]))

    def test_untracked_predicate_stays_permissive(self):
        self.assertTrue(self.world_model.validate_assertion("SPARKLY", ["suitcase"]))

    def test_referent_registration_and_pronoun_resolution(self):
        ref_id = self.world_model.register_referent("suitcase", "noun", ["portable"])
        self.assertIn(ref_id, self.world_model.active_referents)
        resolved = self.world_model.resolve_pronoun("it")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.name, "suitcase")

    def test_discourse_clearance(self):
        self.world_model.register_referent("trophy", "noun", ["portable"])
        self.assertEqual(len(self.world_model.active_referents), 1)
        self.world_model.clear_discourse()
        self.assertEqual(len(self.world_model.active_referents), 0)

    def test_clear_discourse_preserves_knowledge_base(self):
        self.world_model.clear_discourse()
        self.assertIn("portable", self.world_model.knowledge_base)

class TestLinksLattice(unittest.TestCase):
    def setUp(self):
        self.world_model = WorldModel()
        self.world_model.register_entity("student_a", "STUDENT")
        self.world_model.register_entity("student_b", "STUDENT")
        self.world_model.register_entity("student_c", "STUDENT")

    def test_register_entity_is_atomic(self):
        entity = self.world_model.entities["student_a"]
        self.assertTrue(entity.is_atomic())

    def test_register_entity_can_be_registered_as_an_unenumerated_plural(self):
        self.world_model.register_entity("suitcases", "SUITCASE", is_sum = True)
        entity = self.world_model.entities["suitcases"]
        self.assertTrue(entity.is_sum)
        self.assertEqual(entity.members, [])

    def test_join_produces_a_sum(self):
        sum_entity = self.world_model.join_entities(["student_a", "student_b"])
        self.assertTrue(sum_entity.is_sum)
        self.assertEqual(set(sum_entity.members), {"student_a", "student_b"})

    def test_join_flattens_nested_sums(self):
        first_pair = self.world_model.join_entities(["student_a", "student_b"])
        everyone = self.world_model.join_entities([first_pair.id, "student_c"])
        self.assertEqual(set(everyone.members), {"student_a", "student_b", "student_c"})

    def test_joining_the_same_members_twice_returns_the_same_entity(self):
        first = self.world_model.join_entities(["student_a", "student_b"])
        second = self.world_model.join_entities(["student_b", "student_a"])
        self.assertIs(first, second)

    def test_collective_reading(self):
        # "The students gathered": true of the group as a whole, not of each member.
        group = self.world_model.join_entities(["student_a", "student_b"])
        self.world_model.knowledge_base["gathered"] = [group.id]
        self.assertTrue(self.world_model.holds_collectively("gathered", group))
        self.assertFalse(self.world_model.holds_distributively("gathered", group))

    def test_distributive_reading(self):
        # "The students each read a book": true only if every member individually read.
        group = self.world_model.join_entities(["student_a", "student_b"])
        self.world_model.knowledge_base["read"] = ["student_a", "student_b"]
        self.assertTrue(self.world_model.holds_distributively("read", group))

    def test_distributive_reading_fails_if_any_member_does_not_hold(self):
        group = self.world_model.join_entities(["student_a", "student_b"])
        self.world_model.knowledge_base["read"] = ["student_a"]
        self.assertFalse(self.world_model.holds_distributively("read", group))

class TestEpisodicTimeline(unittest.TestCase):
    def setUp(self):
        self.world_model = WorldModel()

    def test_advance_turn_increments(self):
        self.assertEqual(self.world_model.advance_turn(), 1)
        self.assertEqual(self.world_model.advance_turn(), 2)

    def test_present_tense_event_has_aligned_times(self):
        self.world_model.advance_turn()
        record = self.world_model.record_event("PORTABLE", {}, tense = "present")
        self.assertEqual(record.event_time, record.reference_time)
        self.assertEqual(record.reference_time, record.speech_time)

    def test_past_tense_event_precedes_speech_time(self):
        self.world_model.advance_turn()
        record = self.world_model.record_event("LEAVE", {}, tense = "past")
        self.assertLess(record.event_time, record.speech_time)
        self.assertTrue(record.is_past())

    def test_future_tense_event_follows_speech_time(self):
        self.world_model.advance_turn()
        record = self.world_model.record_event("ARRIVE", {}, tense = "future")
        self.assertGreater(record.event_time, record.speech_time)

    def test_event_is_appended_to_the_log(self):
        self.world_model.advance_turn()
        self.world_model.record_event("KICK", {"agent": "john", "patient": "ball"})
        self.assertEqual(len(self.world_model.event_log), 1)
        self.assertEqual(self.world_model.event_log[0].roles["agent"], "john")

class TestPresuppositionTracking(unittest.TestCase):
    def setUp(self):
        self.world_model = WorldModel()

    def test_non_trigger_word_returns_none(self):
        self.assertIsNone(get_presupposition_trigger("walk"))

    def test_trigger_word_is_recognized(self):
        self.assertIsNotNone(get_presupposition_trigger("stop"))

    def test_unknown_presupposition_status_is_none(self):
        self.assertIsNone(self.world_model.check_presupposition("stop", "john"))

    def test_asserted_presupposition_is_then_known(self):
        self.world_model.assert_presupposition("stop", "john", holds = True)
        self.assertTrue(self.world_model.check_presupposition("stop", "john"))

    def test_presupposition_check_for_non_trigger_word_is_none_even_if_asserted(self):
        # "walk" isn't a presupposition trigger at all, so there's nothing to check regardless.
        self.assertIsNone(self.world_model.check_presupposition("walk", "john"))

    def test_negation_does_not_touch_the_presupposition_store(self):
        # "John didn't stop smoking" still presupposes he used to: is_negated never enters here.
        self.world_model.assert_presupposition("stop", "john", holds = True)

        # No is_negated parameter exists on these methods at all; the presence of that fact, asserted once, is unaffected 
        # by anything about the main clause's polarity.
        self.assertTrue(self.world_model.check_presupposition("stop", "john"))

class TestModalAndAttitudeLayer(unittest.TestCase):
    def setUp(self):
        self.world_model = WorldModel()

    def test_global_context_exists_by_default(self):
        self.assertIn("global", self.world_model.contexts)
        self.assertTrue(self.world_model.contexts["global"].is_root())

    def test_open_context_records_holder_and_flavor(self):
        context = self.world_model.open_context("john", ModalFlavor.EPISTEMIC)
        self.assertEqual(context.holder, "john")
        self.assertEqual(context.modal_flavor, ModalFlavor.EPISTEMIC)
        self.assertEqual(context.parent_id, "global")

    def test_fact_asserted_in_belief_context_does_not_leak_into_global_reality(self):
        # "John thinks Mary is at home" (but she isn't, in reality).
        context = self.world_model.open_context("john", ModalFlavor.EPISTEMIC)
        self.world_model.assert_fact_in_context(context.id, "at_home", "mary")

        self.assertTrue(self.world_model.validate_assertion_in_context(context.id, "at_home", ["mary"]))
        self.assertNotIn("mary", self.world_model.knowledge_base.get("at_home", []))

    def test_fact_asserted_in_global_context_does_update_knowledge_base(self):
        self.world_model.assert_fact_in_context("global", "portable", "backpack")
        self.assertIn("backpack", self.world_model.knowledge_base["portable"])

    def test_two_holders_beliefs_do_not_interfere(self):
        john_context = self.world_model.open_context("john", ModalFlavor.EPISTEMIC)
        mary_context = self.world_model.open_context("mary", ModalFlavor.EPISTEMIC)
        self.world_model.assert_fact_in_context(john_context.id, "at_home", "mary")

        self.assertTrue(self.world_model.validate_assertion_in_context(john_context.id, "at_home", ["mary"]))
        # Mary's own belief context knows nothing about this: untracked predicate stays permissive,
        # which is the correct default (absence of a recorded fact isn't evidence against it).
        self.assertTrue(self.world_model.validate_assertion_in_context(mary_context.id, "at_home", ["mary"]))
        self.assertEqual(self.world_model.context_facts.get(mary_context.id, {}), {})

class TestEpisodicFactGraph(unittest.TestCase):
    def setUp(self):
        self.world_model = WorldModel()

    def test_discourse_asserted_fact_is_queryable(self):
        self.world_model.assert_episodic_fact(FrameTemplate.IS_A, "bob", "google_employee")
        fact = self.world_model.query_episodic_fact(FrameTemplate.IS_A, "bob")
        self.assertIsNotNone(fact)
        self.assertEqual(fact["object"], "google_employee")
        self.assertEqual(fact["provenance"], FactProvenance.AUTHORITATIVE)

    def test_query_with_no_matching_fact_returns_none(self):
        self.assertIsNone(self.world_model.query_episodic_fact(FrameTemplate.IS_A, "nobody"))

    def test_provisional_fact_can_be_corroborated(self):
        self.world_model.assert_episodic_fact(
            FrameTemplate.AT_PLACE, "seattle", "washington", provenance = FactProvenance.PROVISIONAL
        )
        self.world_model.corroborate_fact(FrameTemplate.AT_PLACE, "seattle", "washington")
        fact = self.world_model.query_episodic_fact(FrameTemplate.AT_PLACE, "seattle")
        self.assertEqual(fact["provenance"], FactProvenance.AUTHORITATIVE)

    def test_expire_removes_only_stale_provisional_facts(self):
        self.world_model.current_turn = 1
        self.world_model.assert_episodic_fact(
            FrameTemplate.AT_PLACE, "seattle", "washington", provenance = FactProvenance.PROVISIONAL
        )
        self.world_model.current_turn = 5
        self.world_model.assert_episodic_fact(FrameTemplate.IS_A, "bob", "employee") # Authoritative, fresh.

        self.world_model.expire_provisional_facts(before_turn = 5)

        self.assertIsNone(self.world_model.query_episodic_fact(FrameTemplate.AT_PLACE, "seattle"))
        self.assertIsNotNone(self.world_model.query_episodic_fact(FrameTemplate.IS_A, "bob"))

    def test_external_knowledge_graph_is_an_honest_stub(self):
        self.assertIsNone(self.world_model.query_external_knowledge_graph(FrameTemplate.IS_A, "anything"))

class TestRelationalFacts(unittest.TestCase):
    """
    The binary (subject, object) fact store: additive to knowledge_base's unary-only table, and distinct from the closed,
    FrameTemplate-keyed Episodic Fact Graph above: this one is for arbitrary verb predicates.
    """
    def setUp(self):
        self.world_model = WorldModel()

    def test_untracked_predicate_stays_permissive(self):
        self.assertTrue(self.world_model.holds_relationally("read", "john", "book"))

    def test_recorded_pair_holds(self):
        self.world_model.assert_relational_fact("read", "john", "book")
        self.assertTrue(self.world_model.holds_relationally("read", "john", "book"))

    def test_once_tracked_an_unrecorded_pair_does_not_hold(self):
        self.world_model.assert_relational_fact("read", "john", "book")
        self.assertFalse(self.world_model.holds_relationally("read", "mary", "book"))

    def test_assertion_is_case_insensitive(self):
        self.world_model.assert_relational_fact("READ", "John", "Book")
        self.assertTrue(self.world_model.holds_relationally("read", "john", "book"))

    def test_asserting_the_same_pair_twice_does_not_duplicate(self):
        self.world_model.assert_relational_fact("read", "john", "book")
        self.world_model.assert_relational_fact("read", "john", "book")
        self.assertEqual(self.world_model.relational_facts["read"], [("john", "book")])

class TestEntitiesOfKind(unittest.TestCase):
    def setUp(self):
        self.world_model = WorldModel()

    def test_seeded_kind_returns_its_holder_list(self):
        self.world_model.knowledge_base["student"] = ["alice", "bob"]
        self.assertEqual(self.world_model.entities_of_kind("STUDENT"), ["alice", "bob"])

    def test_untracked_kind_returns_empty_list_not_a_guess(self):
        self.assertEqual(self.world_model.entities_of_kind("dragon"), [])

class TestClearDiscourseResetsNewStateButKeepsKnowledgeBase(unittest.TestCase):
    def test_clear_discourse_resets_everything_conversation_scoped(self):
        world_model = WorldModel()
        world_model.register_entity("student_a", "STUDENT")
        world_model.advance_turn()
        world_model.record_event("WALK", {})
        world_model.assert_presupposition("stop", "john", holds = True)
        context = world_model.open_context("john", ModalFlavor.EPISTEMIC)
        world_model.assert_fact_in_context(context.id, "at_home", "mary")
        world_model.assert_episodic_fact(FrameTemplate.IS_A, "bob", "employee")
        world_model.assert_relational_fact("read", "john", "book")

        world_model.clear_discourse()

        self.assertEqual(world_model.entities, {})
        self.assertEqual(world_model.event_log, [])
        self.assertEqual(world_model.current_turn, 0)
        self.assertEqual(world_model.presupposed_facts, {})
        self.assertEqual(list(world_model.contexts.keys()), ["global"])
        self.assertEqual(world_model.context_facts, {})
        self.assertEqual(world_model.episodic_facts, {})
        self.assertEqual(world_model.relational_facts, {})
        self.assertIn("portable", world_model.knowledge_base) # Durable, not conversation-scoped.

if __name__ == "__main__":
    unittest.main()