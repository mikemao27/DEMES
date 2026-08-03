"""
DEMES's memory: what it currently believes is true, what's been said in this conversation, who believes what, and how to tell 
"true in reality" apart from "true inside someone's head."

WHAT PROBLEM THIS FILE SOLVES: A naive world model is just a flat table of facts. That breaks the moment a sentence describes
something other than plain, agreed-upon reality: "John thinks Mary is at home, but she isn't" is not a contradiction to store 
as one fact; it's two different facts belonging to two different places (what's real, and what John separately believes). This 
file is where DEMES keeps those places properly separated, along with several other kinds of state a flat table can't represent:
plural entities that need to support both "the students gathered" (true of the group) and "the students each read a book" (true 
of every member) readings, events with real relative timing (not just "past" as a single flag), and facts DEMES wasn't told directly 
but has to assume are true for a sentence to make sense at all ("my sister is coming over" presupposes a sister exists).

WHAT LIVES HERE:
    1. The original extensional fact store (`knowledge_base`) and discourse-referent tracking (`active_referents`), unchanged from 
    before: this file adds capability, it doesn't remove any.

    2. Link's lattice: entities that can be atomic or sums-of-entities, so collective and distributive readings of plurals resolve 
    against real structure instead of guessing.

    3. An episodic timeline of events, each carrying a real Reichenbach speech/reference/event-time triple rather than a flat tense string.
    
    4. Presupposition tracking, kept in a channel completely separate from ordinary asserted facts, specifically so negating a sentence 
    never accidentally negates what it presupposes.

    5. The Modal & Attitude layer: a tree of belief/desire/permission contexts, so what's true "in John's head" never gets confused with 
    what's actually true.

    6. The Episodic Fact Graph: contingent, one-off facts ("Bob works at Google") that were either stated directly in conversation or 
    fetched on demand when something presupposed them but wasn't yet known: kept provenance-tagged, so a fetched-not-stated fact is never trusted as
    much as one the user actually said.

WHAT DOES NOT LIVE HERE: This file stores and checks facts; it does not decide what a sentence means (core/parser.py, core/semantics.py) or 
whether some non-literal reading needs to be tried first (core/figurative.py). It also does not call out to a language model itself:
`query_external_knowledge_graph` is an honest, empty interface point for whatever later file (interface/neural_bridge.py, or a real knowledge-graph client) 
ends up filling it in; it does not fabricate an answer today.
"""

from enum import Enum
from typing import Dict, List, Any, Optional

from core.types import DiscourseReferent, Entity, EventRecord, ContextIndex, ModalFlavor, Explication, FrameTemplate

# Presupposition triggers: a closed set of words whose use presupposes something beyond what they assert outright, each declaring what that is via the same 
# frame grammar core/lexicon.py uses for ordinary word meanings.
PRESUPPOSITION_TRIGGERS: Dict[str, Explication] = {
    "stop": Explication(frame = FrameTemplate.HAPPENS_TO, slots = {"presupposed": "USED_TO_DO"}),
    "start": Explication(frame = FrameTemplate.HAPPENS_TO, slots = {"presupposed": "DID_NOT_DO_BEFORE"}),
    "again": Explication(frame = FrameTemplate.HAPPENS_TO, slots = {"presupposed": "HAPPENED_BEFORE"}),
    "know": Explication(frame = FrameTemplate.KNOWS, slots = {"presupposed": "CONTENT_IS_TRUE"}),
    "realize": Explication(frame = FrameTemplate.KNOWS, slots = {"presupposed": "CONTENT_IS_TRUE"}),
}

def get_presupposition_trigger(word: str) -> Optional[Explication]:
    """
    Returns the closed-vocabulary explication of what `word` presupposes, or None if it isn't a trigger.
    """
    return PRESUPPOSITION_TRIGGERS.get(word.lower())

class FactProvenance(Enum):
    """
    How confident DEMES should be in an episodic fact. AUTHORITATIVE facts came directly from the conversation (the user said it) 
    or a terminal-search lookup: trusted indefinitely. PROVISIONAL facts came from a language model's parametric guess when nothing 
    local had the answer: trusted only for the turn they were introduced, unless something later corroborates them. This mirrors 
    the temporary-molecule safeguard in core/lexicon.py: a guess should never quietly become a permanent fact just by sitting around unchallenged.
    """
    AUTHORITATIVE = "authoritative"
    PROVISIONAL = "provisional"

class WorldModel:
    """
    The central store for factual knowledge, plural entity structure, episodic event timing, presupposed content, nested belief/attitude contexts, 
    and one-off conversational facts, plus the active discourse referents needed for pronoun resolution.
    """
    def __init__(self):
        # Unchanged from the original design: durable, hand-seeded property facts.
        self.knowledge_base: Dict[str, List[str]] = {
            "portable": ["suitcase", "trophy"],
            "container": ["suitcase"],
            "heavy": ["suitcase", "trophy"],
        }

        # Unchanged: active discourse referents for pronoun resolution.
        self.active_referents: Dict[str, DiscourseReferent] = {}
        self._referent_counter = 0

        # Link's lattice: every entity DEMES has registered, atomic or plural.
        self.entities: Dict[str, Entity] = {}

        # The episodic event timeline (Reichenbach-timed event/state records).
        self.event_log: List[EventRecord] = []
        self.current_turn = 0

        # Presupposed content, kept as its own channel: never touched by negation handling.
        self.presupposed_facts: Dict[str, bool] = {}

        # The Modal & Attitude context tree. "global" is shared, agreed-upon reality; every other context is scoped to one attitude-holder 
        # and one modal flavor.
        self.contexts: Dict[str, ContextIndex] = {"global": ContextIndex(id="global")}
        self.context_facts: Dict[str, Dict[str, List[str]]] = {}
        self._context_counter = 0

        # The Episodic Fact Graph: one-off, contingent facts, each provenance-tagged.
        self.episodic_facts: Dict[str, Dict[str, Any]] = {}

    # Original interface, unchanged: extensional validation and discourse referents.
    def validate_assertion(self, predicate: str, arguments: List[Any]) -> bool:
        """
        Validates a global-reality assertion. If the predicate is tracked in the knowledge base, every argument must be a recorded holder of it; 
        an untracked predicate stays permissive.
        """
        clean_pred = predicate.lower()
        known_holders = self.knowledge_base.get(clean_pred)
        if known_holders is None:
            return True
        return all(str(arg).lower() in known_holders for arg in arguments)

    def register_referent(self, name: str, entity_type: str, properties: List[str] = None) -> str:
        """
        Registers a new discourse entity into the active DRS and returns its unique ID.
        """
        self._referent_counter += 1
        ref_id = f"ref_{self._referent_counter}"
        self.active_referents[ref_id] = DiscourseReferent(
            id = ref_id, name = name, type = entity_type, properties = properties or []
        )
        return ref_id

    def resolve_pronoun(self, pronoun: str) -> Optional[DiscourseReferent]:
        """
        Resolves a pronoun to the most recently registered matching discourse referent.
        """
        if not self.active_referents:
            return None
        recent_ids = list(self.active_referents.keys())
        return self.active_referents[recent_ids[-1]]

    def clear_discourse(self) -> None:
        """
        Resets everything scoped to "this conversation" (active referents, registered entities, the event timeline, non-global attitude contexts, 
        and episodic facts) when the conversation window resets. `knowledge_base` (durable, hand-seeded world facts) is deliberately left untouched: 
        it isn't something this conversation introduced.
        """
        self.active_referents.clear()
        self._referent_counter = 0
        self.entities.clear()
        self.event_log.clear()
        self.current_turn = 0
        self.presupposed_facts.clear()
        self.contexts = {"global": ContextIndex(id = "global")}
        self.context_facts.clear()
        self._context_counter = 0
        self.episodic_facts.clear()

    # Link's lattice: plural entities, and collective vs. distributive checks.
    def register_entity(self, entity_id: str, kind: str, is_sum: bool = False) -> Entity:
        """
        Registers a single entity: atomic by default (e.g. one particular suitcase). Passing is_sum = True registers it as an unenumerated plurality instead 
        (e.g. a bare "the suitcases" mention, where grammatical number is known but individual members aren't): real individual-member enumeration (via join_entities) 
        still requires those members to have been introduced some other way; this only records that the mention itself was plural.
        """
        entity = Entity(id = entity_id, kind = kind, is_sum = is_sum)
        self.entities[entity_id] = entity
        return entity

    def join_entities(self, entity_ids: List[str]) -> Entity:
        """
        Godehard Link's sum/join operation: combines several entities (atomic or already-plural) into one plural entity representing all of them together, 
        flattening any nested sums so a sum's members are always atoms, never other sums. Creating the same sum twice returns the
        same Entity rather than a duplicate.
        """
        members: List[str] = []
        for entity_id in entity_ids:
            existing = self.entities.get(entity_id)
            if existing and existing.is_sum:
                members.extend(existing.members)
            else:
                members.append(entity_id)

        sum_id = "+".join(sorted(set(members)))
        if sum_id not in self.entities:
            first_member = self.entities.get(members[0])
            kind = first_member.kind if first_member else "UNKNOWN"
            self.entities[sum_id] = Entity(id = sum_id, kind = kind, is_sum = True, members = sorted(set(members)))
        return self.entities[sum_id]

    def holds_collectively(self, predicate: str, entity: Entity) -> bool:
        """
        True if `predicate` is recorded as holding of the plural entity AS A WHOLE (e.g. "the students gathered").
        """
        holders = self.knowledge_base.get(predicate.lower(), [])
        return entity.id in holders

    def holds_distributively(self, predicate: str, entity: Entity) -> bool:
        """
        True if `predicate` is recorded as holding of EVERY member individually (e.g. "the students each read a book").
        """
        holders = self.knowledge_base.get(predicate.lower(), [])
        members = entity.members if entity.is_sum else [entity.id]
        return all(member in holders for member in members)

    # The episodic timeline: events with real Reichenbach speech/reference/event time.
    def advance_turn(self) -> int:
        """
        Moves the conversation's clock forward by one turn. Called once per user turn by the pipeline.
        """
        self.current_turn += 1
        return self.current_turn

    def record_event(self, predicate: str, roles: Dict[str, str], tense: str = "present", aktionsart = None) -> EventRecord:
        """
        Adds an event or state to the episodic timeline, deriving a Reichenbach speech/reference/event-time triple from the sentence's tense. 
        This is a coarse mapping honestly: a flat "past"/"present"/"future" label (all core/parser.py currently detects) can't distinguish
        pluperfect from simple past the way real Reichenbach analysis requires: that needs the parser to recognize richer tense/aspect marking 
        than it does today. What's implemented here is correct for the tense distinctions actually available right now.
        """
        speech_time = self.current_turn
        if tense == "past":
            event_time, reference_time = speech_time - 1, speech_time
        elif tense == "future":
            event_time, reference_time = speech_time + 1, speech_time
        else:
            event_time, reference_time = speech_time, speech_time

        record = EventRecord(
            predicate = predicate,
            roles = roles,
            speech_time = speech_time,
            reference_time = reference_time,
            event_time = event_time,
            aktionsart = aktionsart,
        )
        self.event_log.append(record)
        return record

    # Presupposition: a channel kept deliberately separate from at-issue assertions.
    def check_presupposition(self, trigger_word: str, subject: str) -> Optional[bool]:
        """
        Checks whether a presupposition trigger's presupposed content is already known to hold for `subject`. Returns True or False if DEMES has a 
        recorded answer, or None if it genuinely doesn't know yet: a None result is exactly the signal presupposition accommodation
        (core/discourse.py, not yet built) needs to decide whether to assume the presupposition rather than fail outright. This store never consults 
        or is consulted by is_negated: keeping it out of that check entirely is what makes it negation-transparent.
        """
        if get_presupposition_trigger(trigger_word) is None:
            return None
        key = f"{trigger_word.lower()}:{subject.lower()}"
        return self.presupposed_facts.get(key)

    def assert_presupposition(self, trigger_word: str, subject: str, holds: bool = True) -> None:
        """
        Records whether a presupposition trigger's presupposed content holds for `subject`.
        """
        key = f"{trigger_word.lower()}:{subject.lower()}"
        self.presupposed_facts[key] = holds

    # The Modal & Attitude layer: nested belief/desire/permission contexts.
    def open_context(self, holder: str, modal_flavor: ModalFlavor, parent_id: str = "global") -> ContextIndex:
        """
        Opens a new context scoped to one attitude-holder and one modal flavor (epistemic, deontic, or bouletic), as a child of `parent_id`. 
        Everything asserted "inside" this context (via assert_fact_in_context) stays isolated from global reality and from every other context.
        """
        self._context_counter += 1
        context_id = f"{holder}_{modal_flavor.value}_{self._context_counter}"
        context = ContextIndex(id = context_id, holder = holder, modal_flavor = modal_flavor, parent_id = parent_id)
        self.contexts[context_id] = context
        return context

    def assert_fact_in_context(self, context_id: str, predicate: str, argument: str) -> None:
        """
        Records that `predicate` holds of `argument`: but only within `context_id`, not in global reality, unless `context_id` is "global" itself. 
        This is the actual mechanism that keeps "John thinks Mary is at home" from corrupting what DEMES knows to be globally true: the
        fact is written into John's belief context's own local store, never into knowledge_base.
        """
        clean_pred = predicate.lower()
        clean_arg = argument.lower()

        if context_id == "global":
            self.knowledge_base.setdefault(clean_pred, [])
            if clean_arg not in self.knowledge_base[clean_pred]:
                self.knowledge_base[clean_pred].append(clean_arg)
            return

        local_facts = self.context_facts.setdefault(context_id, {})
        local_facts.setdefault(clean_pred, [])
        if clean_arg not in local_facts[clean_pred]:
            local_facts[clean_pred].append(clean_arg)

    def validate_assertion_in_context(self, context_id: str, predicate: str, arguments: List[Any]) -> bool:
        """
        The context-aware counterpart to validate_assertion: checks a context's own local facts, not global reality.
        """
        if context_id == "global":
            return self.validate_assertion(predicate, arguments)

        local_facts = self.context_facts.get(context_id, {})
        known_holders = local_facts.get(predicate.lower())
        if known_holders is None:
            return True
        return all(str(arg).lower() in known_holders for arg in arguments)

    # The Episodic Fact Graph: one-off, contingent facts, each provenance-tagged.
    def assert_episodic_fact(
        self,
        relation: FrameTemplate,
        subject: str,
        obj: str,
        provenance: FactProvenance = FactProvenance.AUTHORITATIVE,
        confidence: float = 1.0,
    ) -> None:
        """
        Discourse-driven assertion: the moment a user states a fact ("Bob works at Google"), it's written straight into working memory: no lookup needed, 
        since it was just given directly.
        """
        key = f"{relation.value}:{subject.lower()}:{obj.lower()}"
        self.episodic_facts[key] = {
            "relation": relation,
            "subject": subject.lower(),
            "object": obj.lower(),
            "provenance": provenance,
            "confidence": confidence,
            "turn": self.current_turn,
        }

    def query_episodic_fact(self, relation: FrameTemplate, subject: str, obj: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Looks up a previously-asserted episodic fact, either the exact (relation, subject, object) triple or, if `obj` is omitted, the first fact matching 
        (relation, subject) regardless of object. Used both by ordinary evaluation and by presupposition accommodation's on-demand retrieval.
        """
        if obj is not None:
            return self.episodic_facts.get(f"{relation.value}:{subject.lower()}:{obj.lower()}")
        for fact in self.episodic_facts.values():
            if fact["relation"] == relation and fact["subject"] == subject.lower():
                return fact
        return None

    def corroborate_fact(self, relation: FrameTemplate, subject: str, obj: str) -> None:
        """
        Upgrades a provisional (model-guessed) fact to authoritative: called on terminal-search confirmation or user restatement.
        """
        key = f"{relation.value}:{subject.lower()}:{obj.lower()}"
        if key in self.episodic_facts:
            self.episodic_facts[key]["provenance"] = FactProvenance.AUTHORITATIVE

    def expire_provisional_facts(self, before_turn: int) -> None:
        """
        Removes provisional facts older than `before_turn` that were never corroborated: they don't get to linger as if trusted.
        """
        stale_keys = [
            key for key, fact in self.episodic_facts.items()
            if fact["provenance"] == FactProvenance.PROVISIONAL and fact["turn"] < before_turn
        ]
        for key in stale_keys:
            del self.episodic_facts[key]

    def query_external_knowledge_graph(self, relation: FrameTemplate, subject: str) -> Optional[Dict[str, Any]]:
        """
        The interface point for consulting an external, indexable knowledge source, restricted to DEMES's own closed relation vocabulary 
        (FrameTemplate) so the open-vocabulary problem this whole architecture exists to avoid doesn't just reappear one layer further out. This is
        deliberately unimplemented (it always returns None) rather than faking a data source that doesn't exist. Wiring this to a real source 
        (or to interface/neural_bridge.py's controlled model-lookup fallback) is later, separately tracked work; this is only the contract it will
        need to satisfy: closed relation types in, a fact or nothing out, never a fabricated answer.
        """
        return None