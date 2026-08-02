"""
Manages the DEMES dictionary: what DEMES knows about individual words, and how it learns more.

WHAT PROBLEM THIS FILE SOLVES: Every other layer of DEMES eventually needs to answer the question "what does this word mean, and 
what kind of word is it?" This file is the single place that question gets answered from. It is also the single gatekeeper for a rule 
the rest of the system depends on: a word's meaning may only ever be built out of the closed ~65-concept vocabulary defined in core/primitives.py. 
Nothing else in DEMES is allowed to invent a new fundamental concept: if a definition doesn't fit using what core/primitives.py already provides, 
this file rejects it rather than quietly accepting it.

WHAT LIVES HERE, AND WHY THEY'RE TOGETHER RATHER THAN SPREAD ACROSS SEPARATE FILES:
    1. Loading, saving, and validating the lexicon (checking every word's definition against the closed primitive vocabulary).

    2. Word-sense disambiguation: when a word has more than one meaning (like "bank"), picking the one that fits its sentence. 
    This is fundamentally a lexicon-lookup decision ("which of this word's stored definitions applies here") so it lives with the 
    definitions themselves rather than in a separate file.

    3. Morphology: recognizing that "walked" and "walk" are the same word wearing a different grammatical ending, so a sentence using either form 
    can still be looked up successfully.

    4. Minting proper nouns: when a capitalized word like "John" isn't in the dictionary because it isn't a common word at all but a name, this file is 
    where that gets recognized and handled differently from an ordinary unknown word.

    5. Learning new words at runtime, carefully: if this file cannot find a word anywhere, it can accept a proposed definition (from a small offline reference, or 
    eventually a language model): but only provisionally, and only after checking that proposed definition against the same closed vocabulary as everything else. 
    A guess that turns out to be wrong should never be able to quietly become a permanent, trusted fact about what a word means.

WHAT DOES NOT LIVE HERE: This file never reaches out to a language model itself. If a word truly cannot be found anywhere locally, the right move is for 
the caller (eventually interface/neural_bridge.py) to ask a model for a candidate definition and hand it back here to be checked: this file's job stops at "check
what you're given, store it if it passes, and never trust it blindly." Keeping model calls physically out of this file is what makes it possible to trust that nothing 
in here can silently introduce meaning DEMES didn't actually verify.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from core.primitives import validate_primitives, assert_valid_primitives
from core.types import DiscourseReferent

# Morphology (Layer 3): recognizing inflected word forms.

# English mostly builds inflected forms (plurals, past tense, comparatives, ...) by adding a small, fixed set of endings to a base word, with a handful of spelling adjustments 
# (doubling a final consonant, dropping a silent "e") along the way. Rather than trying to predict exactly which spelling rule applies to which word, this file takes a simpler 
# and more robust approach: for an unrecognized word, generate every plausible base-form candidate the regular rules could produce, and let the dictionary itself decide which candidate 
# (if any) is real by checking whether it's an actual entry. Irregular forms ("went", "children", "better") don't follow any spelling rule at all, so they're handled separately, via 
# a small lookup table rather than a rule.

# Irregular inflected forms that no suffix rule could ever derive. This table is meant to grow by adding more entries (data), never by adding more code to handle "special" words.
_IRREGULAR_LEMMAS: Dict[str, str] = {
    "went": "go", "gone": "go", "goes": "go", "going": "go",
    "was": "be", "were": "be", "been": "be", "am": "be", "is": "be", "are": "be",
    "had": "have", "has": "have",
    "did": "do", "done": "do",
    "said": "say",
    "saw": "see", "seen": "see",
    "took": "take", "taken": "take",
    "gave": "give", "given": "give",
    "knew": "know", "known": "know",
    "thought": "think",
    "felt": "feel",
    "heard": "hear",
    "made": "make",
    "ate": "eat", "eaten": "eat",
    "ran": "run",
    "children": "child",
    "people": "person",
    "better": "good", "best": "good",
    "worse": "bad", "worst": "bad",
}

# Regular suffixes tried, in order, against an unrecognized word. Each is paired with a plain-language label describing what that ending signals grammatically, used by detect_inflection().
_REGULAR_SUFFIX_RULES: Tuple[Tuple[str, str], ...] = (
    ("ies", "plural_or_third_person"), # cities -> city.
    ("es", "plural_or_third_person"), # boxes -> box.
    ("s", "plural_or_third_person"), # walks -> walk, suitcases -> suitcase.
    ("ing", "progressive"), # walking -> walk.
    ("ed", "past"), # walked -> walk.
    ("est", "superlative"), # biggest -> big.
    ("er", "comparative"), # bigger -> big.
)

def _generate_lemma_candidates(word: str) -> List[Tuple[str, str]]:
    """
    Produces a list of (candidate base word, grammatical label) pairs for an inflected word, most likely candidates first. This does not know or claim which candidate is actually a real word:
    that check happens wherever this is consulted (get_word_definition, detect_inflection), by seeing which candidate is actually in the dictionary.
    """
    candidates: List[Tuple[str, str]] = []

    if word in _IRREGULAR_LEMMAS:
        candidates.append((_IRREGULAR_LEMMAS[word], "irregular"))

    for suffix, label in _REGULAR_SUFFIX_RULES:
        if not word.endswith(suffix) or len(word) <= len(suffix):
            continue

        stem = word[: -len(suffix)]

        if suffix == "ies":
            candidates.append((stem + "y", label))
            continue

        # Plain stem, e.g. "walked" -> "walk".
        candidates.append((stem, label))

        # Undo a doubled final consonant, e.g. "running" -> "runn" (plain) -> "run".
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            candidates.append((stem[:-1], label))

        # Undo a dropped silent "e", e.g. "making" -> "mak" (plain) -> "make".
        candidates.append((stem + "e", label))

    return candidates

class LexiconManager:
    """
    Handles loading, querying, updating, and growing the DEMES lexicon store. Acts as the single source of truth for word meaning, valency, and (via the closed primitive vocabulary) for what
    counts as a legitimate word definition at all.
    """

    PROPER_NOUN_TYPES = ("PERSON", "PLACE", "ORGANIZATION", "ENTITY")

    def __init__(self, store_path: str = "data/lexicon.json"):
        self.store_path = store_path
        self.lexicon: Dict[str, Dict] = {}

        # Words learned at runtime (via induce_word) live here, separately from the permanent store, until a human explicitly promotes one: see the module docstring and
        # promote_to_permanent() for why this separation exists.
        self.provisional_lexicon: Dict[str, Dict] = {}

        # Populated during load_lexicon() with any entries that were rejected for referencing a primitive outside the closed vocabulary, so a caller can choose to surface them instead
        # of the problem disappearing silently.
        self.load_warnings: List[str] = []

        self._proper_noun_counter = 0

        self.load_lexicon()

    # Loading, saving, and validating the permanent store.
    def load_lexicon(self) -> None:
        """
        Loads the persistent lexicon from disk, handling empty or missing files gracefully, and checking every entry's primitives against the closed vocabulary. An entry that references
        a primitive outside that vocabulary is left out of the loaded lexicon rather than trusted, this is what "closure is enforced at load time" actually means in practice: a bad entry on
        disk cannot re-enter the working vocabulary just by being present in the file.
        """
        self.load_warnings = []
        raw_lexicon: Optional[Dict] = None

        if os.path.exists(self.store_path) and os.path.getsize(self.store_path) > 0:
            try:
                with open(self.store_path, "r", encoding = "utf-8") as file:
                    raw_lexicon = json.load(file)

            except json.JSONDecodeError:
                pass # Fall back to seeding if JSON is corrupt.

        if raw_lexicon is None:
            self.lexicon = self._seed_lexicon()
            self.save_lexicon()
            return

        self.lexicon = {}
        for word, entry in raw_lexicon.items():
            invalid_names = validate_primitives(entry.get("primitives", []))

            if invalid_names:
                self.load_warnings.append(
                    f"'{word}' excluded: references primitive(s) not in the closed vocabulary: {invalid_names}."
                )
                continue

            self.lexicon[word] = entry

    def save_lexicon(self) -> None:
        """
        Persists the permanent lexicon store back to disk. Deliberately only ever writes self.lexicon (the permanent, verified store): self.provisional_lexicon is never touched
        here, so a provisional guess can never end up on disk just because a save happened to run.
        """
        os.makedirs(os.path.dirname(self.store_path), exist_ok = True)
        with open(self.store_path, "w", encoding = "utf-8") as file:
            json.dump(self.lexicon, file, indent = 4)

    def _seed_lexicon(self) -> Dict[str, Dict]:
        """
        The minimal built-in vocabulary used when no lexicon file exists yet, kept small and deliberately valid against the closed primitive vocabulary. These are coarse, honest
        placeholder glosses (e.g. "portable" as CAN + MOVE: it can be moved), not full formal explications: writing real, careful explications for a working vocabulary is its own
        piece of linguistic work, tracked separately rather than rushed here.
        """
        return {
            "walk": {
                "category": "verb",
                "semantic_type": "<e, t>",
                "primitives": [{"name": "MOVE", "category": "action"}, {"name": "BODY", "category": "entity"}],
                "valency": "intransitive",
            },
            "suitcase": {
                "category": "noun",
                "semantic_type": "e",
                "primitives": [{"name": "SOMETHING", "category": "entity"}],
                "valency": "none",
            },
            "portable": {
                "category": "adjective",
                "semantic_type": "<e, t>",
                "primitives": [{"name": "CAN", "category": "logical"}, {"name": "MOVE", "category": "action"}],
                "valency": "none",
            },
        }

    # Looking words up, including inflected forms.
    def get_word_definition(self, word: str) -> Optional[Dict]:
        """
        Looks up a word's definition. Tries the word exactly as given first (against both the permanent and provisional stores); if that fails, tries the base-form candidates
        _generate_lemma_candidates produces, so an inflected form like "walked" or "suitcases" resolves to the same entry as its base word without needing its own separate dictionary
        entry.
        """
        clean_word = word.lower()

        direct = self.lexicon.get(clean_word) or self.provisional_lexicon.get(clean_word)
        if direct:
            return direct

        for candidate, _label in _generate_lemma_candidates(clean_word):
            found = self.lexicon.get(candidate) or self.provisional_lexicon.get(candidate)
            if found:
                return found

        return None

    def lemmatize(self, word: str) -> Optional[str]:
        """
        Returns the dictionary base form of a word (e.g. "walked" -> "walk"), or the word itself if it's already a recognized base form, or None if no known base form can be found at all.
        """
        clean_word = word.lower()
        if clean_word in self.lexicon or clean_word in self.provisional_lexicon:
            return clean_word

        for candidate, _label in _generate_lemma_candidates(clean_word):
            if candidate in self.lexicon or candidate in self.provisional_lexicon:
                return candidate

        return None

    def detect_inflection(self, word: str) -> Optional[str]:
        """
        Reports what grammatical ending, if any, was recognized on a word ("past", "progressive", "plural_or_third_person", "comparative", "superlative", "irregular"), without deciding
        what that means for a sentence's overall tense: that interpretation is a later layer's job (the world model's event records). Returns None if the word is already a base form, or
        if nothing about it was recognized at all.
        """
        clean_word = word.lower()
        if clean_word in self.lexicon or clean_word in self.provisional_lexicon:
            return None

        for candidate, label in _generate_lemma_candidates(clean_word):
            if candidate in self.lexicon or candidate in self.provisional_lexicon:
                return label

        return None

    # Word-sense disambiguation.
    def disambiguate_sense(self, word: str, argument_types: List[str]) -> str:
        """
        For a word with more than one stored sense (like "bank", which can mean a financial institution or a riverbank), picks the sense whose selectional constraint is actually
        present among argument_types: the selectional-constraint tags of the other, unambiguous words in the same sentence. Returns the winning sense's key (e.g.
        "bank.n.financial_institution"), or just the word itself if it has zero or one sense (no real ambiguity to resolve), or the first listed sense if none of its constraints match the
        given context (a defined, predictable fallback rather than a guess).
        """
        clean_word = word.lower()
        word_def = self.get_word_definition(clean_word)
        if not word_def:
            return clean_word

        senses = word_def.get("senses", [])
        if len(senses) <= 1:
            return clean_word

        for sense in senses:
            required_constraint = sense.get("selectional_constraint")
            if not required_constraint or required_constraint in argument_types:
                return sense.get("sense_key", clean_word)

        return senses[0].get("sense_key", clean_word)

    # Proper nouns (Layer 1b): names are not concepts, so they are never decomposed.
    def mint_proper_noun(self, word: str, syntactic_role: Optional[str] = None) -> DiscourseReferent:
        """
        Creates a fresh discourse referent for a capitalized word that isn't in the lexicon at all: the right response to encountering a name like "John" or "Seattle", which was never
        going to have a dictionary entry because it isn't a common word with a decomposable meaning. No explication is attempted or stored; a proper noun is treated as a rigid label
        for one specific individual, not a concept to define.

        syntactic_role is an optional hint about how the word was used in its sentence (e.g. "subject_of_animate_verb", "object_of_locative_preposition"): this file doesn't determine
        that itself, since it requires seeing sentence structure, but it knows how to turn such a hint into a coarse type guess (PERSON, PLACE, ORGANIZATION) once the grammar layer supplies
        one. With no hint, the referent is minted as a generic ENTITY rather than guessed at.

        This does not register the referent into any conversation-wide state: that's the calling code's responsibility, once it decides the referent should actually enter the discourse.
        """
        entity_type = self._infer_proper_noun_type(syntactic_role)
        self._proper_noun_counter += 1
        referent_id = f"proper_{self._proper_noun_counter}"

        return DiscourseReferent(
            id = referent_id,
            name = word,
            type = entity_type,
            animate = (entity_type == "PERSON"),
        )

    def _infer_proper_noun_type(self, syntactic_role: Optional[str]) -> str:
        if syntactic_role == "subject_of_animate_verb":
            return "PERSON"
        if syntactic_role == "object_of_locative_preposition":
            return "PLACE"
        
        return "ENTITY"
    
    # Learning new words at runtime (the word half of Layer 8), carefully.
    def induce_word(
        self,
        word: str,
        category: str,
        primitives: List[Dict],
        provenance: str = "induced_unverified",
    ) -> None:
        """
        Accepts a proposed definition for a word DEMES has never seen before: a "temporary semantic molecule". The proposed primitives are checked against the closed vocabulary
        exactly as strictly as everything else in this file (a bad guess is refused outright, it raises rather than being quietly accepted), but even a definition that passes that check is
        stored only in the provisional store, never written to disk, and never merged into the permanent lexicon automatically. It stays available for the rest of the session, and it
        stays exactly as trustworthy as "provisional" (no more) until a person deliberately reviews it and calls promote_to_permanent().
        """
        assert_valid_primitives(primitives)

        clean_word = word.lower()
        self.provisional_lexicon[clean_word] = {
            "category": category,
            "semantic_type": "<e, t>" if category == "verb" else "e",
            "primitives": primitives,
            "valency": "induced",
            "provenance": provenance,
        }

    def promote_to_permanent(self, word: str) -> None:
        """
        The only path by which a provisionally-learned word becomes a trusted, permanent part of the lexicon: moves it out of the provisional store, into the permanent one, and persists
        it to disk. This is meant to be called deliberately (by a person reviewing what DEMES has learned, not automatically by anything else in the system) which is the actual mechanism
        behind the rule "a guess never quietly becomes a fact".
        """
        clean_word = word.lower()
        if clean_word not in self.provisional_lexicon:
            raise KeyError(f"'{word}' has no provisional definition to promote.")

        entry = self.provisional_lexicon.pop(clean_word)
        entry["provenance"] = "authoritative"
        self.lexicon[clean_word] = entry
        self.save_lexicon()