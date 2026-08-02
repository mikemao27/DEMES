"""
Unit tests for core/figurative.py: idiom rewriting, metonymic coercion, conceptual-metaphor mapping, and the ordered repair chain that ties them together.
"""

import unittest

from core.figurative import (
    IdiomMeaning,
    rewrite_idiom,
    CoercionPattern,
    attempt_coercion,
    ConceptualDomain,
    attempt_metaphor_mapping,
    resolve_figurative_meaning,
)

class TestIdiomRewrite(unittest.TestCase):
    def test_known_idiom_resolves(self):
        meaning = rewrite_idiom("IDIOM:kick_bucket")
        self.assertIsInstance(meaning, IdiomMeaning)
        self.assertEqual(meaning.explication.slots["event"], "DIE")

    def test_non_idiom_predicate_returns_none(self):
        self.assertIsNone(rewrite_idiom("PORTABLE"))

    def test_idiom_tag_with_unregistered_meaning_returns_none(self):
        self.assertIsNone(rewrite_idiom("IDIOM:unregistered_phrase"))

    def test_syntactic_flexibility_is_tracked_per_idiom(self):
        kick_bucket = rewrite_idiom("IDIOM:kick_bucket")
        spill_beans = rewrite_idiom("IDIOM:spill_beans")
        self.assertFalse(kick_bucket.passivizable)
        self.assertTrue(spill_beans.passivizable)

class TestMetonymicCoercion(unittest.TestCase):
    def test_licensed_container_for_person_satisfies_person_requirement(self):
        sandwich_entry = {"category": "noun", "metonymy_licenses": ["container_for_person"]}
        result = attempt_coercion("PERSON", sandwich_entry)
        self.assertIsNotNone(result)
        pattern, coerced_type = result
        self.assertEqual(pattern, CoercionPattern.CONTAINER_FOR_PERSON)
        self.assertEqual(coerced_type, "PERSON")

    def test_unlicensed_entry_does_not_coerce(self):
        plain_entry = {"category": "noun"}
        self.assertIsNone(attempt_coercion("PERSON", plain_entry))

    def test_licensed_pattern_that_does_not_match_required_type_fails(self):
        sandwich_entry = {"category": "noun", "metonymy_licenses": ["container_for_person"]}
        self.assertIsNone(attempt_coercion("PRODUCT", sandwich_entry))

    def test_unrecognized_pattern_name_is_ignored_gracefully(self):
        entry = {"category": "noun", "metonymy_licenses": ["not_a_real_pattern"]}
        self.assertIsNone(attempt_coercion("PERSON", entry))

    def test_place_for_institution_covers_both_person_and_institution_requirements(self):
        wall_street_entry = {"category": "noun", "metonymy_licenses": ["place_for_institution_or_inhabitants"]}
        self.assertIsNotNone(attempt_coercion("PERSON", wall_street_entry))
        self.assertIsNotNone(attempt_coercion("INSTITUTION", wall_street_entry))


class TestConceptualMetaphorMapping(unittest.TestCase):
    def test_digest_maps_to_comprehend_for_information_domain(self):
        metaphor = attempt_metaphor_mapping("digest", ConceptualDomain.INFORMATION.value)
        self.assertIsNotNone(metaphor)
        self.assertEqual(metaphor.target_predicate, "COMPREHEND")

    def test_digest_does_not_map_for_unrelated_domain(self):
        self.assertIsNone(attempt_metaphor_mapping("digest", ConceptualDomain.ORGANIZATION.value))

    def test_bleed_maps_to_lose_resources_for_organization_domain(self):
        metaphor = attempt_metaphor_mapping("bleed", ConceptualDomain.ORGANIZATION.value)
        self.assertEqual(metaphor.target_predicate, "LOSE_RESOURCES")

    def test_unregistered_predicate_returns_none(self):
        self.assertIsNone(attempt_metaphor_mapping("walk", ConceptualDomain.INFORMATION.value))

class TestFigurativeRepairChain(unittest.TestCase):
    def test_idiom_takes_priority_over_everything_else(self):
        result = resolve_figurative_meaning("IDIOM:kick_bucket", required_type="PERSON", argument_entries=[])
        self.assertEqual(result, "HAPPENS-TO:DIE")

    def test_coercion_applies_when_no_idiom_tag_present(self):
        result = resolve_figurative_meaning(
            "leave",
            required_type = "PERSON",
            argument_entries = [{"metonymy_licenses": ["container_for_person"]}],
        )
        self.assertEqual(result, "COERCED:PERSON")

    def test_metaphor_applies_when_coercion_does_not(self):
        result = resolve_figurative_meaning(
            "digest",
            required_type = "PERSON", # Doesn't match any coercion pattern for this argument.
            argument_entries = [{"conceptual_domain": "information"}],
        )
        self.assertEqual(result, "COMPREHEND")

    def test_coercion_is_tried_before_metaphor_when_both_could_apply(self):
        # An argument that licenses both a matching coercion and a matching metaphor should resolve via coercion (the narrower repair), per the documented ordering.
        result = resolve_figurative_meaning(
            "leave",
            required_type = "PERSON",
            argument_entries = [{
                "metonymy_licenses": ["container_for_person"],
                "conceptual_domain": "information",
            }],
        )
        self.assertEqual(result, "COERCED:PERSON")

    def test_returns_none_when_nothing_applies(self):
        result = resolve_figurative_meaning("walk", required_type = "PERSON", argument_entries = [{"category": "noun"}])
        self.assertIsNone(result)

    def test_returns_none_when_required_type_is_unknown_and_no_idiom(self):
        result = resolve_figurative_meaning("walk", required_type = None, argument_entries = [{"category": "noun"}])
        self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()