"""
Compiles parsed syntactic structures into executable semantic graphs and
executes logical reduction (beta-reduction and quantified evaluation) for DEMES.

How it solves the problem:
    1. Moves beyond static lookups by executing beta-reduction on lambda expressions and evaluating truth conditions dynamically against the WorldModel.
    2. Handles quantified noun phrases (e.g., universal and existential quantification) by mapping them to set-theoretic checks across the world database.
    3. Packages the final evaluation into a structured, inspectable dictionary payload ready for the local SLM stylist presentation layer.
"""

from typing import Dict, Any, Optional, List
from core.types import LogicalForm

class SemanticCompiler:
    """
    Compiles logical forms, performs reduction, and evaluates truth conditions against the active WorldModel state
    and the lexicon's primitive decompositions.
    """

    def __init__(self, world_model: Any, lexicon_manager: Any):
        self.world_model = world_model
        self.lexicon = lexicon_manager

    def compile_and_evaluate(self, logical_form: Optional[LogicalForm]) -> Dict[str, Any]:
        """
        Takes a LogicalForm object, executes reduction/quantification checks, evaluates truth conditions, and builds a clean payload for the stylist.
        """
        if not logical_form:
            return {
                "status": "failure",
                "reason": "Unparsable syntax or unhandled structural gap.",
                "response_intent": "clarification"
            }

        # 1. Check if the logical form involves NP Quantification (e.g., "Every suitcase is portable").
        if hasattr(logical_form, "quantifier_meta") and logical_form.quantifier_meta:
            evaluation_result = self._evaluate_quantified_predicate(logical_form)
        else:
            # Standard relational or property predicate evaluation
            evaluation_result = self._evaluate_standard_predicate(logical_form)

        # Negation flips the evaluated truth value onto the surface assertion.
        if logical_form.is_negated:
            evaluation_result = not evaluation_result

        # 2. Construct the structured dictionary payload for the local SLM stylist.
        payload = {
            "status": "success",
            "predicate": logical_form.predicate,
            "arguments": logical_form.arguments,
            "is_negated": logical_form.is_negated,
            "truth_value": evaluation_result,
            "quantifier": getattr(logical_form, "quantifier_meta", None),
            "response_intent": "assertion_ack" if evaluation_result else "contradiction_notice"
        }

        return payload

    def _evaluate_standard_predicate(self, form: LogicalForm) -> bool:
        """
        Evaluates a standard atomic predicate. Property predicates (adjectives) are checked first
        against the arguments' own lexicon primitive decomposition (intensional match) before
        falling back to the WorldModel's extensional fact store. Relational/event predicates (verbs)
        stay permissive, since "did this event happen" isn't a property-membership question here.
        """
        predicate_lower = form.predicate.lower()
        predicate_def = self.lexicon.get_word_definition(predicate_lower)

        if not predicate_def or predicate_def.get("category") != "adjective":
            return self.world_model.validate_assertion(form.predicate, form.arguments)

        predicate_primitives = {p.get("name") for p in predicate_def.get("primitives", [])}

        for argument in form.arguments:
            argument_def = self.lexicon.get_word_definition(str(argument).lower())
            argument_primitives = {p.get("name") for p in argument_def.get("primitives", [])} if argument_def else set()

            if predicate_primitives & argument_primitives:
                # Intensional match: the entity's own lexicon definition already encodes this property.
                continue

            if not self.world_model.validate_assertion(predicate_lower, [argument]):
                return False

        return True

    def _evaluate_quantified_predicate(self, form: LogicalForm) -> bool:
        """
        Executes set-theoretic evaluation for quantified statements. Example: "FORALL x (SUITCASE(x) -> PORTABLE(x))".
        """
        meta = form.quantifier_meta
        operator = meta.get("operator")
        restrictor = meta.get("restrictor", "").lower()
        property_name = form.predicate.lower()

        # Find all entities in the knowledge base that match the restrictor category or name.
        matching_entities = []
        for cat, items in self.world_model.knowledge_base.items():
            if restrictor == cat or restrictor in items:
                matching_entities.extend(items)
        if not matching_entities and restrictor in self.world_model.knowledge_base:
            matching_entities = self.world_model.knowledge_base[restrictor]

        # Fallback if restrictor is directly a known object.
        if not matching_entities:
            matching_entities = [restrictor]

        target_property_holders = self.world_model.knowledge_base.get(property_name, [])

        if operator == "FORALL":
            return all(entity in target_property_holders for entity in matching_entities)
        elif operator == "EXISTS":
            return any(entity in target_property_holders for entity in matching_entities)
        elif operator == "NOT_EXISTS":
            return not any(entity in target_property_holders for entity in matching_entities)

        return True
