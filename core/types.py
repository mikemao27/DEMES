"""
The shared vocabulary of data shapes used across every layer of DEMES.

WHAT PROBLEM THIS FILE SOLVES: DEMES is built out of many separate pieces (a grammar that figures out sentence structure, a
lexicon that stores word meanings, a world model that remembers facts and events, and a discourse tracker that resolves pronouns) 
and all of those pieces need to hand information back and forth in a shape they all agree on. Without one shared place for those shapes to live, 
each file would end up inventing its own slightly-different version of "what a fact looks like" or "what a pronoun
placeholder looks like," and keeping them in sync by hand becomes its own source of bugs.

HOW THIS FILE FIXES THAT: Every class in this file is a plain data container (Python's "dataclass": a class whose only job
is to hold a fixed set of named fields, with no behavior beyond a couple of very small, obviously correct helper checks). None of the actual 
thinking happens here: no parsing, no truth-checking, no resolving of ambiguity. That logic lives in the files that are actually responsible for it 
(the grammar, the world model, and the discourse tracker) and those files import the shapes defined here instead of each defining their own. 
If you're reading this file to understand DEMES, think of it as the list of "nouns" the rest of the system's "verbs" operate on.

A NOTE ON WHAT'S DEFINED HERE BUT NOT YET USED: A few of the shapes below (EventRecord, ContextIndex, StoredQuantifier, DerivationNode,
UnboundPronoun) aren't referenced by any other file yet: they're the data shapes that later, not-yet-built parts of DEMES (the real grammar engine, 
the world model's event memory, and the pronoun-resolution logic) are already known to need, based on the project's architecture plan.
Defining their shape now, before the logic that uses them exists, means those future files don't each have to invent their own version of 
"what does an event's timing look like."
"""

from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional, Tuple, Any
from enum import Enum

class SemanticType(Enum):
    """
    Basic and complex types in DEMES's typed grammar (e.g. "e" for a single entity, "t" for a truth value). This is the same small type system 
    used in formal semantics generally: it exists so the grammar can check that words are being combined in ways that actually produce
    a sentence meaning, rather than nonsense.
    """
    ENTITY = "e"
    TRUTH_VALUE = "t"
    PROPERTY = "<e, t>"
    RELATION = "<e, <e, t>>"
    MODIFIER = "<t, t>"
    COMPLEX = "complex"

class Gender(Enum):
    """
    The grammatical gender categories DEMES's pronoun agreement checks use (e.g. deciding that "he" cannot refer back to an entity already known to be feminine). 
    This is about grammar, not biology: it's the same three-way distinction English pronouns themselves make.
    """
    MASCULINE = "masc"
    FEMININE = "fem"
    NEUTER = "neut"

class GrammaticalNumber(Enum):
    """
    Whether something is being talked about as one thing or as more than one: used both for ordinary pronoun agreement ("it" vs. "they") and for Layer 6's plural/mass-noun handling.
    """
    SINGULAR = "sing"
    PLURAL = "plur"

class ModalFlavor(Enum):
    """
    The three "flavors" of possibility DEMES's Modal & Attitude layer distinguishes: EPISTEMIC is about what's known or believed to be true ("it might be raining"), 
    DEONTIC is about what's permitted or required ("you must leave"), and BOULETIC is about what's wanted or hoped for ("I want it to rain"). 
    Keeping these three separate matters because the same primitive (like CAN) means something different depending on which flavor of possibility is in play.
    """
    EPISTEMIC = "epistemic"
    DEONTIC = "deontic"
    BOULETIC = "bouletic"

class Aktionsart(Enum):
    """
    The four classic categories (from linguist Zeno Vendler) describing the inherent shape of an event or state in time, independent of its tense: 
    a STATE just holds (knowing something), an ACTIVITY has no natural finishing point (running), an ACCOMPLISHMENT builds toward one and
    stops there (building a house), and an ACHIEVEMENT happens all at once with no duration (winning a race). This is what lets DEMES tell 
    "start knowing" (odd) apart from "start running" (fine): the verb's own inherent shape determines which coercions make sense.
    """
    STATE = "state"
    ACTIVITY = "activity"
    ACCOMPLISHMENT = "accomplishment"
    ACHIEVEMENT = "achievement"

class FrameTemplate(Enum):
    """
    The closed set of relation shapes a word's meaning can be built out of, one level up from the primitives themselves. Knowing that "portable" involves the 
    primitive CAN isn't yet a meaning: HAS-PROPERTY(suitcase, portable) is. Restricting explications to this fixed list of relation shapes (rather than letting 
    any relation name be invented per word) is what keeps word definitions themselves closed, the same way core/primitives.py keeps the atoms closed.
    """
    IS_A = "IS-A"
    HAS_PART = "HAS-PART"
    DOES = "DOES"
    HAPPENS_TO = "HAPPENS-TO"
    CAUSES = "CAUSES"
    AT_PLACE = "AT-PLACE"
    HAS_PROPERTY = "HAS-PROPERTY"
    CAN = "CAN"
    BEFORE = "BEFORE"
    AFTER = "AFTER"
    PART_OF = "PART-OF"
    WANTS = "WANTS"
    THINKS = "THINKS"
    KNOWS = "KNOWS"
    FEELS = "FEELS"

@dataclass
class Explication:
    """
    One word's meaning, expressed as a single relation from the closed FrameTemplate list plus whatever fills its slots. A slot filler is either a primitive name (a string from
    core/primitives.py, e.g. "CAN"), or the name of another lexicon entry whose own explication supplies the rest of the meaning (e.g. "portable"'s explication might fill a HAS-PROPERTY slot
    with the word "portable" itself, whose own explication is CAN + MOVE). Nesting bottoms out eventually at primitives: that bottoming-out is exactly what keeps the whole lexicon's meanings
    traceable back to the closed vocabulary instead of definitions referring to each other forever.
    """
    frame: FrameTemplate
    slots: Dict[str, str] = field(default_factory = dict)

@dataclass
class Primitive:
    """
    One atomic building block of word meaning: a single entry from the closed vocabulary defined in core/primitives.py (e.g. MOVE, GOOD, BEFORE), paired with 
    a loose category label for readability (e.g. "action", "property"). A word's full meaning is built out of a small collection of these, never out of a single one.
    """
    name: str
    category: str

    def __repr__(self) -> str:
        return f"[{self.name}:{self.category}]"

@dataclass
class LambdaExpression:
    """
    A piece of unevaluated sentence meaning waiting to be combined with something else: the "x" in an expression like "the property of being a thing that walks, 
    still waiting for a walker to be plugged in for x". This class only stores the pieces (what variable is waiting to be filled in, what expression fills it in 
    once it is, and what type of meaning results); it does not perform the actual combination itself. That combination process (called beta-reduction) is a
    genuine algorithm involving matching up variables across possibly many nested expressions, and it belongs in core/semantics.py, the file responsible for compositional meaning:
    keeping it out of this file keeps this file limited to data shapes, with zero behavior to get wrong.
    """
    variable: str
    body: Any
    semantic_type: SemanticType = SemanticType.PROPERTY

@dataclass
class StoredQuantifier:
    """
    A quantified noun phrase ("every suitcase", "a book") that has been pulled out of the sentence it appeared in and set aside, rather than immediately deciding 
    what it scopes over. This is Cooper Storage: a sentence like "Every student read a book" is genuinely ambiguous between one shared book and a different book per student, 
    and deciding which reading is meant by guessing during parsing would mean either picking wrong half the time or the parser's work exploding combinatorially trying to track 
    every possible reading at once. Instead, each quantifier found during composition becomes one of these, and the choice of which one gets scope over which
    happens once, later, when a full sentence meaning is being assembled.

    context_id links a stored quantifier to the ContextIndex (below) it was found inside: this is what lets DEMES tell apart "there exists a specific unicorn John wants" 
    from "John wants there to exist some unicorn or other, any will do", by tracking whether the quantifier was captured inside or outside the mental state ("wants") that 
    introduced a hypothetical context.
    """
    operator: str
    restrictor: str
    bound_variable: str
    context_id: str

@dataclass
class ContextIndex:
    """
    One node in DEMES's tree of "who's imagining what". Ordinary sentences are evaluated against a single root context representing shared, agreed-upon reality. 
    But a sentence like "John thinks Mary is at home" describes something true only inside John's belief (not in reality) and asserting it into the same store as real 
    facts would corrupt what DEMES actually knows to be true. Every time a sentence introduces a belief, desire, permission, or hypothetical ("thinks", "wants", "can", "if"), 
    a new ContextIndex is created as a child of whichever context the sentence was already being evaluated in, and everything inside that clause is evaluated
    against the new child context instead of the root.

    holder is who the context belongs to (e.g. "john"), or None for the shared root context. modal_flavor records which of the three kinds of possibility (see ModalFlavor) 
    this context represents, or None for the root. parent_id links back to the enclosing context, so a chain of contexts (what John thinks Mary believes) can be followed 
    back up to shared reality.
    """
    id: str
    holder: Optional[str] = None
    modal_flavor: Optional[ModalFlavor] = None
    parent_id: Optional[str] = None

    def is_root(self) -> bool:
        return self.parent_id is None

class ThematicRole(Enum):
    """
    The closed set of "who did what to whom" roles an event's participants can fill (standard case-grammar roles). Storing "John kicked the ball" as AGENT = john, PATIENT = ball 
    instead of a flat, order-dependent argument list means "who did the kicking" is never ambiguous the way a positional list would leave it once sentences get more complex than simple SVO.
    """
    AGENT = "agent"
    PATIENT = "patient"
    THEME = "theme"
    INSTRUMENT = "instrument"
    LOCATION = "location"
    TIME = "time"

@dataclass
class Entity:
    """
    Something DEMES's world model can refer to: which, in English, is not always just one thing. "The suitcase" refers to a single, atomic entity, but "the students" 
    refers to a group, and "the students" can be talked about either collectively (they gathered, as one group) or distributively (they each read a book, one action per member). 
    This is Godehard Link's lattice-based treatment of plurals: rather than inventing separate machinery for singular vs. plural nouns, every entity is either atomic 
    (is_sum is False, members is empty) or a sum of other entities (is_sum is True, members lists which ones), and collective vs. distributive meaning is worked out by checking 
    membership against this same structure, not by a special plural-only code path.

    kind records what category of thing this is (e.g. "SUITCASE"), which is what degree semantics uses to build a comparison class ("big for a suitcase" compares against other SUITCASE-kind
    entities currently known).
    """
    id: str
    kind: str
    is_sum: bool = False
    members: List[str] = field(default_factory = list)

    def is_atomic(self) -> bool:
        return not self.is_sum

@dataclass
class EventRecord:
    """
    DEMES's memory of one thing that happened or one state that held, stored in a much richer shape than a single sentence's logical form. Two pieces of information live here that a flat
    "predicate plus arguments" representation loses:

    First, thematic roles (roles) record how each participant was involved, not just that they were mentioned: "John kicked the ball" fills AGENT with John and PATIENT with the ball, so
    that "who did the kicking" is never ambiguous the way a plain argument list would leave it.

    Second, timing is stored as three separate points rather than one tense label, following linguist Hans Reichenbach's analysis of tense: speech_time is "now" (when the sentence is
    uttered), event_time is when the described event actually happened, and reference_time is the vantage point the sentence is describing it from. These are stored as plain integers on a
    shared discourse timeline, where a smaller number simply means "earlier": DEMES only needs relative order, not real calendar time. This is what tells apart "When John arrived, Mary had
    left" (event_time before reference_time before speech_time: she was already gone) from "When John arrived, Mary was leaving" (event_time overlapping reference_time: they crossed paths):
    both are grammatically "past tense", but only this three-point structure keeps them distinct.

    aktionsart records the event's inherent shape (see the Aktionsart enum) independently of its tense: "was living in Paris" (a STATE) and "was running" (an ACTIVITY) are both past
    progressive, but only one of them can be coerced by "started".
    """
    predicate: str
    roles: Dict[str, str] = field(default_factory = dict)
    speech_time: int = 0
    reference_time: int = 0
    event_time: int = 0
    aktionsart: Optional[Aktionsart] = None
    context_id: Optional[str] = None

    def is_past(self) -> bool:
        return self.event_time < self.speech_time

    def is_pluperfect(self) -> bool:
        # Event happened, then some later reference point was described, before now: "Mary HAD LEFT (event) by the time John arrived (reference)."
        return self.event_time < self.reference_time < self.speech_time

    def is_progressive_overlap(self) -> bool:
        # The event was still ongoing at the reference point being described: "Mary WAS LEAVING (event, ongoing) when John arrived (reference)."
        return self.event_time <= self.reference_time <= self.event_time + 1 and self.reference_time <= self.speech_time

@dataclass
class DerivationNode:
    """
    One node in the tree of how a sentence's words combined into a full sentence, built up as the grammar (core/parser.py) applies its combination rules. 
    This is kept around after parsing specifically so later checks (most importantly, deciding whether a pronoun is allowed to refer to a name that comes later 
    in the same sentence ("Before he entered the room, John took off his coat" is fine; "He entered the room before John did", meaning the same John, is not))
    have real sentence structure to check against, instead of guessing from word order alone.

    label is the node's grammatical category as a plain string (e.g. "S", "NP"): deliberately not tied to any one grammar implementation's category class, so 
    this shape can be shared across files without those files needing to import each other's internals. children holds the sub-trees that were combined to build this node 
    (empty for a single word). token is the actual word this node represents, only set on leaf nodes. span is the range of word positions this node covers, used to 
    quickly check whether one node's words fall entirely inside another's.
    """
    label: str
    children: Tuple["DerivationNode", ...] = ()
    token: Optional[str] = None
    span: Tuple[int, int] = (0, 0)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def dominates(self, other: "DerivationNode") -> bool:
        """
        True if `other` is this node itself, or is reachable by following children downward from it: i.e. this node's part of the sentence structure fully contains the other one's. 
        This is the basic structural building block a Principle C check is built from; deciding what to do with that fact (block or allow a pronoun reference) is a linguistic policy 
        decision that belongs in the discourse layer, not a plain structural fact like this one.
        """
        if other is self:
            return True
        
        return any(child.dominates(other) for child in self.children)

@dataclass
class UnboundPronoun:
    """
    A pronoun DEMES has read but has not yet been able to connect to anything: specifically the case where the pronoun appeared *before* the name or noun phrase it turns out to refer to
    ("Before HE entered the room, JOHN took off his coat"). Rather than failing outright, or guessing at the nearest available entity and possibly guessing wrong, DEMES holds the pronoun
    open as one of these, remembering only what it structurally requires of whatever eventually fills it in (its grammatical gender/number/animacy and where in the sentence tree it was
    found). The moment a candidate entity appears later in the same sentence, the discourse layer checks whether that entity satisfies these requirements and whether doing so is even
    grammatically allowed at that position (see DerivationNode.dominates) before binding the two together: never before both of those checks pass.
    """
    text: str
    origin_node: Optional[DerivationNode] = None
    required_gender: Optional[Gender] = None
    required_number: Optional[GrammaticalNumber] = None
    required_animate: Optional[bool] = None

@dataclass
class DiscourseReferent:
    """
    An entity that has actually been introduced into the conversation and can now be pointed back to: the resolved counterpart to UnboundPronoun above. Created the moment a noun phrase like
    "a suitcase" or "John" is understood, and consulted whenever a later pronoun needs an antecedent. gender/number/animate are filled in where known, so that pronoun resolution can
    rule out mismatched candidates ("she" cannot resolve to a referent already tagged masculine) instead of just grabbing whichever entity was mentioned most recently regardless of fit.
    """
    id: str
    name: str
    type: str
    properties: List[str] = field(default_factory = list)
    gender: Optional[Gender] = None
    number: Optional[GrammaticalNumber] = None
    animate: Optional[bool] = None

@dataclass
class LogicalForm:
    """
    The complete formal meaning DEMES has assigned to one sentence: the main "output" of parsing and the main "input" to truth evaluation. predicate and arguments capture the basic
    who-did-what-to-what; is_negated and tense capture the two grammatical operators that flip or shift that basic meaning without changing what the predicate and arguments themselves are.
    quantifier_meta, when present, records that this sentence involves a quantified noun phrase ("every", "some", "no") and how it should be evaluated: it is an explicit, always-present
    field (rather than something only sometimes attached after the fact) specifically so every other file can check it the same simple way, `if logical_form.quantifier_meta:`, without first
    needing to check whether the field exists at all. It is populated only when the sentence has exactly one quantifier: the shape core/semantics.py's existing single-quantifier evaluator
    already knows how to check. quantifier_store holds every quantifier the parser actually found (one entry even in the single-quantifier case, more when a sentence like "every student
    read a book" has more than one): this is the field Cooper Storage's scope-reading enumeration (core/semantics.py's enumerate_scope_readings) is meant to consume. plural_arguments
    names which entries in `arguments` were grammatically plural nouns, for core/pipeline.py to register as summed entities (Link's lattice, core/world_model.py's join_entities) instead
    of ordinary atomic ones. is_passive records that the sentence was a passive construction ("The suitcase was kicked [by John]"), meaning `arguments[0]` is semantically the PATIENT
    of `predicate`, not the AGENT the same position would mean for an ordinary active sentence: relevant to core/world_model.py's thematic-role recording once that's populated from
    real parses (not yet true today: record_event is still always called with an empty roles dict), but recorded here now so that later wiring has something real to read.
    """
    predicate: str
    arguments: List[Union[str, "LogicalForm", Primitive]] = field(default_factory = list)
    is_negated: bool = False
    tense: str = "present"
    quantifier_meta: Optional[Dict[str, Any]] = None
    quantifier_store: List[StoredQuantifier] = field(default_factory = list)
    plural_arguments: List[str] = field(default_factory=list)
    is_passive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this logical form into a plain dictionary payload, the shape the presentation layer (interface/stylist.py) expects to receive rather than a Python object.
        """
        return {
            "predicate": self.predicate,
            "arguments": [arg.to_dict() if isinstance(arg, LogicalForm) else str(arg) for arg in self.arguments],
            "is_negated": self.is_negated,
            "tense": self.tense,
            "quantifier_meta": self.quantifier_meta,
            "plural_arguments": self.plural_arguments,
            "is_passive": self.is_passive,
        }