"""
Compiles parsed sentence structure into evaluated meaning: this is where DEMES decides not just what a sentence claims, but whether that claim holds.

WHAT LIVES HERE, IN TWO PARTS: Part one is the PRACTICAL evaluator (SemanticCompiler) that actually runs every turn: it takes the
LogicalForm core/parser.py produced, checks the claim against what the lexicon says a word means and what the world model records as fact, and returns a plain, 
inspectable true/false payload. This is what makes the terminal's "Truth Val" line show something real.

Part two is real compositional semantics: variable binding and function application (formally, beta-reduction) built out of an actual small term language 
(Variable, Constant, Application) rather than a stub, plus Cooper Storage for quantifier scope. These are implemented here as correct, independently-tested mechanisms: 
honestly, they are not yet fed by core/parser.py end to end. The parser currently builds a LogicalForm by reading facts off a finished derivation tree rather than
composing lambda expressions live as it combines words, and it doesn't yet track more than one quantifier per sentence. Wiring the parser to build real compositional 
meaning as it parses, so a sentence like "Every student read a book" can genuinely carry two different scope readings instead of just being flagged as one, is real 
follow-up work, not something this file can claim to have finished by itself. What's here is the correct engine for that work to plug into.

WHY BETA-REDUCTION IS BEING BUILT NOW, NOT LEFT AS A STUB: An earlier, narrower pass through this codebase deliberately deferred real beta-reduction to avoid
turning a bug-fixing pass into a full semantics rewrite. The architecture plan approved since then commits to it explicitly as part of this layer's job. That later, 
more deliberated decision is what this file follows: the earlier deferral doesn't quietly override it.
"""

from itertools import permutations
from typing import Dict, Any, Optional, List, Union

from core.types import LogicalForm, LambdaExpression, SemanticType, StoredQuantifier, Explication, FrameTemplate, Aktionsart

# A small, real term language for compositional meaning.

# Formal semantics needs somewhere to distinguish "a placeholder waiting to be filled in" from "an actual thing" from "one thing being applied to another": that's all 
# these three classes are. They're deliberately minimal: enough to make beta-reduction a genuine, correct operation instead of a Python string trick, without inventing 
# more term-language machinery than DEMES's sentences currently need.
class Variable:
    """
    A placeholder inside a LambdaExpression's body, waiting to be substituted for.
    """
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other) -> bool:
        return isinstance(other, Variable) and self.name == other.name

    def __hash__(self) -> int:
        return hash(("Variable", self.name))

    def __repr__(self) -> str:
        return self.name

class Constant:
    """
    A fixed, already-known piece of meaning: a word, a primitive name, an entity.
    """
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other) -> bool:
        return isinstance(other, Constant) and self.name == other.name

    def __hash__(self) -> int:
        return hash(("Constant", self.name))

    def __repr__(self) -> str:
        return self.name

class Application:
    """
    One term applied to another: the shape "functor(argument)" takes before it's reduced.
    """
    def __init__(self, functor: Any, argument: Any):
        self.functor = functor
        self.argument = argument

    def __eq__(self, other) -> bool:
        return isinstance(other, Application) and self.functor == other.functor and self.argument == other.argument

    def __hash__(self) -> int:
        return hash(("Application", self.functor, self.argument))

    def __repr__(self) -> str:
        return f"{self.functor!r}({self.argument!r})"

Term = Union[Variable, Constant, Application, LambdaExpression]

def substitute(term: Term, variable_name: str, replacement: Term) -> Term:
    """
    Returns a copy of `term` with every free occurrence of the variable named `variable_name` replaced by `replacement`. This is the mechanical heart of beta-reduction: 
    "plug this value in everywhere the placeholder appears." A nested LambdaExpression that rebinds the same variable name shadows it (its own body is left alone), exactly 
    like a nested function definition reusing a parameter name in ordinary programming languages.
    """
    if isinstance(term, Variable):
        return replacement if term.name == variable_name else term

    if isinstance(term, Constant):
        return term

    if isinstance(term, Application):
        return Application(
            substitute(term.functor, variable_name, replacement),
            substitute(term.argument, variable_name, replacement),
        )

    if isinstance(term, LambdaExpression):
        if term.variable == variable_name:
            return term # Shadowed: the inner variable of the same name binds first.
        return LambdaExpression(
            variable = term.variable,
            body = substitute(term.body, variable_name, replacement),
            semantic_type = term.semantic_type,
        )

    return term

def beta_reduce(functor: LambdaExpression, argument: Term) -> Term:
    """
    Applies a LambdaExpression to an argument: substitutes the argument for the expression's bound variable throughout its body, and returns the resulting 
    (possibly still-unreduced) term. E.g. reducing "the property of being a thing X such that PORTABLE(X)" applied to Constant("suitcase") yields 
    Application(Constant("PORTABLE"), Constant("suitcase")): a fully formed claim, ready to be checked against the world.
    """
    return substitute(functor.body, functor.variable, argument)

def evaluate_term(term: Term) -> Any:
    """
    Fully reduces a term by repeatedly applying beta-reduction wherever an Application's functor is itself a LambdaExpression, until nothing more can be reduced. 
    Returns the final term: typically an Application of Constants once every variable has been filled in.
    """
    if isinstance(term, Application):
        functor = evaluate_term(term.functor)
        argument = evaluate_term(term.argument)
        if isinstance(functor, LambdaExpression):
            return evaluate_term(beta_reduce(functor, argument))
        return Application(functor, argument)

    if isinstance(term, LambdaExpression):
        return LambdaExpression(term.variable, evaluate_term(term.body), term.semantic_type)

    return term

# Cooper Storage: setting quantifiers aside instead of guessing their scope during composition.
def store_quantifier(store: List[StoredQuantifier], quantifier: StoredQuantifier) -> List[StoredQuantifier]:
    """
    Adds a quantified noun phrase to the store, leaving the choice of its scope for later.
    """
    return store + [quantifier]

def enumerate_scope_readings(store: List[StoredQuantifier]) -> List[List[StoredQuantifier]]:
    """
    Returns every possible relative ordering of the stored quantifiers, outermost-scoping first in each ordering: one ordering per distinct reading. For "every student read a book", 
    a store of [EVERY-student, A-book] yields two orderings: [EVERY, A] (every student, possibly a different book each time) and [A, EVERY] (one particular book, read by every student). 
    Returning every permutation rather than picking one is the point: DEMES doesn't have to guess which reading was meant during parsing, only enumerate the genuine possibilities for 
    a later layer (discourse context, or eventually the neural bridge) to choose between.
    """
    return [list(ordering) for ordering in permutations(store)]

# Aktionsart: a verb's inherent shape in time, derived from its explication rather than tagged by hand.
def derive_aktionsart(explication: Explication) -> Optional[Aktionsart]:
    """
    Classifies an event description into one of Vendler's four categories purely from the shape of its explication frame (see core/types.py's Aktionsart for what each category means), 
    rather than requiring every verb to be manually tagged with one. A bare property (HAS-PROPERTY) is a STATE. An action with no encoded result (DOES, nothing else) is an ACTIVITY. 
    An action frame that also CAUSES a resulting state is an ACCOMPLISHMENT. A HAPPENS frame (something occurring, not being done by an agent) is treated as an ACHIEVEMENT: punctual, 
    no build-up.

    This function is correct and tested against the Explication shape, but no lexicon entry currently has real explication data yet (that migration is separately tracked, 
    not done here): so it has nothing to classify in the live system yet. It's ready for the moment that data exists.
    """
    if explication.frame == FrameTemplate.HAS_PROPERTY:
        return Aktionsart.STATE
    
    if explication.frame == FrameTemplate.HAPPENS_TO:
        return Aktionsart.ACHIEVEMENT
    
    if explication.frame == FrameTemplate.DOES:
        if "result" in explication.slots or "causes" in explication.slots:
            return Aktionsart.ACCOMPLISHMENT
        return Aktionsart.ACTIVITY
    
    if explication.frame == FrameTemplate.CAUSES:
        return Aktionsart.ACCOMPLISHMENT
    
    return None

# The practical evaluator: what actually runs every turn.
class SemanticCompiler:
    """
    Evaluates a LogicalForm's truth against the lexicon's own definitions (intensional evaluation) and, where that's not enough, the world model's recorded facts 
    (extensional evaluation), and packages the result into the plain dictionary payload the terminal displays.
    """
    def __init__(self, world_model: Any, lexicon_manager: Any):
        self.world_model = world_model
        self.lexicon = lexicon_manager

    def compile_and_evaluate(self, logical_form: Optional[LogicalForm]) -> Dict[str, Any]:
        """
        Evaluates a full turn: dispatches to quantified or standard-predicate evaluation, applies negation, and builds the inspectable result payload.
        """
        if not logical_form:
            return {
                "status": "failure",
                "reason": "Unparsable syntax or unhandled structural gap.",
                "response_intent": "clarification",
            }

        if logical_form.quantifier_meta:
            evaluation_result = self._evaluate_quantified_predicate(logical_form)
        else:
            evaluation_result = self._evaluate_standard_predicate(logical_form)

        if logical_form.is_negated:
            evaluation_result = not evaluation_result

        return {
            "status": "success",
            "predicate": logical_form.predicate,
            "arguments": logical_form.arguments,
            "is_negated": logical_form.is_negated,
            "truth_value": evaluation_result,
            "quantifier": logical_form.quantifier_meta,
            "response_intent": "assertion_ack" if evaluation_result else "contradiction_notice",
        }

    def _evaluate_standard_predicate(self, form: LogicalForm) -> bool:
        """
        Property predicates (adjectives) are checked first against the argument's own lexicon definition (intensional match: does the entity's own 
        primitive decomposition already include this property?) before falling back to the world model's recorded facts (extensional match). Relational/event predicates 
        (verbs) stay permissive here, since "did this event happen" isn't a property-membership question the way "is this adjective true of this entity" is.
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
                continue # Intensional match: the entity's own definition already encodes this.

            if not self.world_model.validate_assertion(predicate_lower, [argument]):
                return False

        return True

    def _evaluate_quantified_predicate(self, form: LogicalForm) -> bool:
        """
        Set-theoretic evaluation for a quantified statement, e.g. FORALL x (SUITCASE(x) -> PORTABLE(x)). Handles the single-quantifier sentences core/parser.py 
        currently produces; see the module docstring for the gap between this and full Cooper Storage scope resolution.
        """
        meta = form.quantifier_meta
        operator = meta.get("operator")
        restrictor = meta.get("restrictor", "").lower()
        property_name = form.predicate.lower()

        matching_entities = []
        for category, items in self.world_model.knowledge_base.items():
            if restrictor == category or restrictor in items:
                matching_entities.extend(items)

        if not matching_entities and restrictor in self.world_model.knowledge_base:
            matching_entities = self.world_model.knowledge_base[restrictor]
        if not matching_entities:
            matching_entities = [restrictor]

        target_property_holders = self.world_model.knowledge_base.get(property_name, [])

        if operator == "FORALL":
            return all(entity in target_property_holders for entity in matching_entities)
        if operator == "EXISTS":
            return any(entity in target_property_holders for entity in matching_entities)
        if operator == "NOT_EXISTS":
            return not any(entity in target_property_holders for entity in matching_entities)

        return True