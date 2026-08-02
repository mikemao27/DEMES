"""
Unit tests for the closed NSM primitive vocabulary in core/primitives.py.
"""

import unittest
from core.primitives import (
    ALL_PRIMITIVES,
    primitive_categories,
    primitives_in_category,
    is_valid_primitive,
    validate_primitives,
    assert_valid_primitives,
    InvalidPrimitiveError,
)

class TestPrimitiveVocabulary(unittest.TestCase):

    def test_known_primitives_are_valid(self):
        for name in ("MOVE", "THINK", "KNOW", "WANT", "GOOD", "BAD", "BIG", "SMALL", "IF", "CAN"):
            self.assertTrue(is_valid_primitive(name), f"{name} should be a recognized primitive.")

    def test_made_up_primitive_is_rejected(self):
        # The exact bug that motivated this file in the first place.
        self.assertFalse(is_valid_primitive("MUSHROOM_OR_PORTABLE"))
        self.assertFalse(is_valid_primitive("LEGS"))
        self.assertFalse(is_valid_primitive("random_lowercase"))

    def test_vocabulary_size_is_bounded(self):
        # Not an exact NSM headcount (different published tables split a couple of glosses differently) but this pins down 
        # that the set is small and closed, not open-ended.
        self.assertGreater(len(ALL_PRIMITIVES), 50)
        self.assertLess(len(ALL_PRIMITIVES), 80)

    def test_categories_partition_the_full_set_without_overlap(self):
        seen = set()
        for category in primitive_categories():
            names = primitives_in_category(category)
            self.assertTrue(names, f"category {category} should not be empty.")

            for name in names:
                self.assertNotIn(name, seen, f"{name} appears in more than one category.")
                seen.add(name)

        self.assertEqual(seen, set(ALL_PRIMITIVES))

    def test_unknown_category_returns_empty_tuple(self):
        self.assertEqual(primitives_in_category("not_a_real_category"), ())

    def test_validate_primitives_reports_only_invalid_entries(self):
        primitives = [
            {"name": "MOVE", "category": "action"},
            {"name": "NOT_A_PRIMITIVE", "category": "action"},
            {"name": "GOOD", "category": "property"},
        ]
        self.assertEqual(validate_primitives(primitives), ["NOT_A_PRIMITIVE"])

    def test_validate_primitives_empty_list_is_valid(self):
        self.assertEqual(validate_primitives([]), [])

    def test_assert_valid_primitives_passes_silently_when_valid(self):
        assert_valid_primitives([{"name": "MOVE"}, {"name": "GOOD"}])

    def test_assert_valid_primitives_raises_on_invalid(self):
        with self.assertRaises(InvalidPrimitiveError):
            assert_valid_primitives([{"name": "MUSHROOM_OR_PORTABLE"}])

if __name__ == "__main__":
    unittest.main()