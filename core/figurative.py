"""
Handles the three ways English routinely means something other than the literal, word-by-word combination of its parts: and why treating 
them as three separate mechanisms, tried in a fixed order, matters.

WHAT PROBLEM THIS FILE SOLVES: Ordinary composition (core/semantics.py) assumes a sentence's meaning is built strictly out of what
its words literally denote combined together. That assumption breaks, in three genuinely different ways, constantly in real English:
    1. Idioms: "kick the bucket" does not mean anything about kicking, or buckets. The whole phrase has a meaning that isn't built 
    from its parts at all: it has to be swapped in wholesale.
  
    2. Metonymy: "the ham sandwich left" doesn't claim a sandwich walked out: "the ham sandwich" stands in for the person who ordered it. 
    Nothing about the sentence's STRUCTURE changed; one argument's reference quietly shifted to something associated with it.
    
    3. Metaphor: "he digested the book" doesn't claim literal biological digestion: the whole PATTERN of physical consumption is being reused 
    to describe understanding. Unlike metonymy, it's not one argument's reference shifting; it's the predicate's entire frame being reused in
    a different domain.

These are not interchangeable repairs for "the sentence doesn't literally make sense": each targets a different part of the sentence (the whole predicate-argument unit, 
one argument's reference, or the whole predicate's frame) and licensing one doesn't tell you the others apply. Conflating them would mean getting the wrong one to fire, 
or firing one that produces a nonsense combination that happens to satisfy a looser check. This file keeps them as three distinct, individually closed mechanisms, 
tried in a fixed order (idiom, then metonymy, then metaphor) matching how "unusual but wrong" a sentence would have to be to need each one.

WHAT THIS FILE DOES NOT DO: None of these three mechanisms invents a new interpretation on the fly: each one only fires when the specific word or word-combination 
involved has been explicitly registered as licensing it (an idiom in the idiom table, a noun explicitly marked as licensing a metonymy pattern, a predicate
explicitly registered against a conceptual-metaphor mapping). A sentence that doesn't match any of the three closed registries is correctly left unrepaired: this file does not, 
and structurally cannot, guess at figurative meaning nobody has told it about. Genuinely novel, on-the-fly metaphor invention is explicitly out of scope for any closed system 
(see the architecture plan's "what this doesn't claim" section) and is the neural bridge's job, not this file's.

This module is also standalone and tested in isolation, honestly: nothing in core/semantics.py's SemanticCompiler calls into it yet, and no current lexicon entry declares 
the metonymy/metaphor licensing data it reads. Wiring it into live evaluation, and adding real licensing data to the lexicon, are both tracked as follow-up work, not done here.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from core.types import Explication, FrameTemplate

# Idioms: a whole predicate-argument unit is replaced, not built compositionally.
@dataclass
class IdiomMeaning:
    """
    The real, NSM-grounded meaning behind an idiom core/parser.py has already syntactically tagged (see parser.py's _IDIOM_OBJECT_TRIGGERS: that table decides WHEN the idiomatic reading
    is syntactically available; this one decides WHAT it means). passivizable and modifiable record which surface transformations this specific idiom tolerates without breaking its
    idiomatic reading: idioms vary here ("the beans were spilled" still means what "spilled the beans" means; "*the bucket was kicked" does not mean what "kicked the bucket" means), so this
    is tracked per idiom rather than assumed uniform for all of them.
    """
    explication: Explication
    passivizable: bool = False
    modifiable: bool = False

# Keyed by the same "trigger-lemma_object" key core/parser.py builds when it detects an idiom ("IDIOM:kick_bucket" -> "kick_bucket"). Deliberately tiny: proves the mechanism, not a real
# idiom dictionary.
_IDIOM_MEANINGS: Dict[str, IdiomMeaning] = {
    "kick_bucket": IdiomMeaning(
        explication = Explication(frame = FrameTemplate.HAPPENS_TO, slots = {"x": "SUBJECT", "event": "DIE"}),
        passivizable = False,
        modifiable = False,
    ),
    "spill_beans": IdiomMeaning(
        explication = Explication(frame = FrameTemplate.DOES, slots = {"x": "SUBJECT", "action": "SAY", "content": "SECRET"}),
        passivizable = True,
        modifiable = True,
    ),
}

def rewrite_idiom(idiom_tagged_predicate: str) -> Optional[IdiomMeaning]:
    """
    Given a predicate string as core/parser.py tags it when an idiomatic combination was recognized (e.g. "IDIOM:kick_bucket"), returns the idiom's real meaning, or None if the string
    isn't an idiom tag at all or isn't one this file has a registered meaning for yet.
    """
    prefix = "IDIOM:"
    if not idiom_tagged_predicate.startswith(prefix):
        return None
    return _IDIOM_MEANINGS.get(idiom_tagged_predicate[len(prefix):])

# Metonymy: one argument's reference shifts to something associated with it.
class CoercionPattern(Enum):
    """
    A closed, cataloged inventory of productive metonymy patterns (after Lakoff & Johnson, and Nunberg's work on deferred reference): not an open-ended "anything can stand for anything"
    rule, which would make almost any sentence trivially "repairable" into meaning anything.
    """
    CONTAINER_FOR_PERSON = "container_for_person" # "the ham sandwich left".
    PLACE_FOR_INSTITUTION_OR_INHABITANTS = "place_for_institution_or_inhabitants" # "Wall Street panicked".
    PRODUCER_FOR_PRODUCT = "producer_for_product" # "I bought a Ford".

def attempt_coercion(required_type: str, argument_entry: Dict) -> Optional[Tuple[CoercionPattern, str]]:
    """
    Checks whether an argument's own lexicon entry licenses a metonymic shift that would satisfy a predicate's required argument type. Licensing is opt-in and explicit: 
    an entry only qualifies if it declares itself under a "metonymy_licenses" field (a list of CoercionPattern values), nothing is inferred about a word's suitability for metonymy 
    from its category or primitives alone. Returns the pattern used and the type the argument coerces to if one applies, otherwise None (a "hamster" has no reason to license CONTAINER_FOR_PERSON 
    just because it's a noun).
    """
    licensed_pattern_names = argument_entry.get("metonymy_licenses", [])

    for pattern_name in licensed_pattern_names:
        try:
            pattern = CoercionPattern(pattern_name)
        except ValueError:
            continue

        if pattern == CoercionPattern.CONTAINER_FOR_PERSON and required_type == "PERSON":
            return pattern, "PERSON"
        if pattern == CoercionPattern.PLACE_FOR_INSTITUTION_OR_INHABITANTS and required_type in ("PERSON", "INSTITUTION"):
            return pattern, required_type
        if pattern == CoercionPattern.PRODUCER_FOR_PRODUCT and required_type == "PRODUCT":
            return pattern, "PRODUCT"

    return None

# Metaphor: the predicate's whole frame is reused from a different conceptual domain.
class ConceptualDomain(Enum):
    """
    The closed set of source/target domains DEMES's registered conceptual metaphors map between.
    """
    PHYSICAL_CONSUMPTION = "physical_consumption"
    INFORMATION = "information"
    BODY = "body"
    ORGANIZATION = "organization"

@dataclass
class ConceptualMetaphor:
    """
    One registered structural correspondence between a physical source domain and an abstract target domain (Lakoff & Johnson's conceptual metaphor theory): e.g. UNDERSTANDING IS EATING
    licenses reusing the pattern of physical consumption (source_predicate) to describe comprehension (target_predicate) whenever the argument actually belongs to the target domain.
    """
    name: str
    source_domain: ConceptualDomain
    target_domain: ConceptualDomain
    source_predicate: str
    target_predicate: str

_CONCEPTUAL_METAPHORS: List[ConceptualMetaphor] = [
    ConceptualMetaphor(
        name = "UNDERSTANDING_IS_EATING",
        source_domain = ConceptualDomain.PHYSICAL_CONSUMPTION,
        target_domain = ConceptualDomain.INFORMATION,
        source_predicate = "DIGEST",
        target_predicate = "COMPREHEND",
    ),
    ConceptualMetaphor(
        name = "ORGANIZATIONS_ARE_BODIES",
        source_domain = ConceptualDomain.BODY,
        target_domain = ConceptualDomain.ORGANIZATION,
        source_predicate = "BLEED",
        target_predicate = "LOSE_RESOURCES",
    ),
]

def attempt_metaphor_mapping(predicate: str, argument_domain: str) -> Optional[ConceptualMetaphor]:
    """
    Checks the closed conceptual-metaphor registry for a mapping whose source predicate matches the one used and whose target domain matches the argument's own declared conceptual domain
    (again an explicit, opt-in lexicon field: "he digested the book" only maps to COMPREHEND because "book" would need to declare itself as belonging to the INFORMATION domain, not
    because DEMES inferred that on its own). Returns the matching metaphor, or None.
    """
    predicate_upper = predicate.upper()
    for metaphor in _CONCEPTUAL_METAPHORS:
        if metaphor.source_predicate == predicate_upper and metaphor.target_domain.value == argument_domain:
            return metaphor
    return None

# The repair chain: idiom, then metonymy, then metaphor: never more than one applied per sentence.
def resolve_figurative_meaning(
    predicate: str,
    required_type: Optional[str],
    argument_entries: List[Dict],
) -> Optional[str]:
    """
    Attempts to explain a predicate that ordinary literal composition couldn't make sense of, by trying each of the three mechanisms above in order and stopping at the first one that applies.
    This ordering matters and is not arbitrary: idiom-tagging (from parsing) is checked first because it's already a confirmed, syntax-licensed non-literal reading, not a guess. Metonymy is
    checked before metaphor because it's the narrower repair (shifting one argument's reference): trying the broader, whole-frame metaphor repair first could paper over what's actually just a
    referential shift. Returns a replacement predicate/meaning label if some repair applies, or None if the predicate is genuinely ungrounded and none of the three closed registries cover it.
    """
    idiom_meaning = rewrite_idiom(predicate)
    if idiom_meaning is not None:
        return f"{idiom_meaning.explication.frame.value}:{idiom_meaning.explication.slots.get('event', predicate)}"

    if required_type is not None:
        for argument_entry in argument_entries:
            coercion = attempt_coercion(required_type, argument_entry)
            if coercion is not None:
                _pattern, coerced_type = coercion
                return f"COERCED:{coerced_type}"

        for argument_entry in argument_entries:
            argument_domain = argument_entry.get("conceptual_domain")
            if argument_domain is None:
                continue
            metaphor = attempt_metaphor_mapping(predicate, argument_domain)
            if metaphor is not None:
                return metaphor.target_predicate

    return None