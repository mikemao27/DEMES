"""
The main terminal REPL (Read-Eval-Print Loop) for DEMES.

How everything fits together:
    1. User Input -> DEMESPipeline (core/pipeline.py): tokenizes the input, queries the LexiconManager
    (core/lexicon.py), resolves pragmatic intent (core/pragmatics.py) and lexical ambiguity (core/wsd.py),
    and builds a formal LogicalForm (core/types.py) via the ChartParser (core/parser.py).
    2. LogicalForm -> SemanticCompiler (core/semantics.py): evaluates the logical form against the WorldModel
    (core/world_model.py) to verify the truth conditions and consistency.
    3. WorldModel -> Active State & DRS Update: updates active discourse referents and relational knowledge.
    4. Evaluation Payload -> LocalStylist (interface/stylist.py): converts the rigid truth payload into a warm, natural English
    response via the local SLM (or fallback).
    5. Transparency View: the terminal displays the raw logical breakdown FIRST, followed by the styled response, allowing full
    inspection of the engine's internal understanding.
"""

import sys
from typing import Dict, Any

from core.pipeline import DEMESPipeline
from interface.stylist import LocalStylist

# Terminal Color Setup:
# Color escape codes are disabled automatically if stdout is piped to a file or non-TTY (non-terminal).
_IS_TTY = sys.stdout.isatty()

def _code(seq: str) -> str:
    return seq if _IS_TTY else ""

RESET = _code("\033[0m")
BOLD = _code("\033[1m")
DIM = _code("\033[2m")
CYAN = _code("\033[36m")
GREEN = _code("\033[32m")
YELLOW = _code("\033[33m")
MAGENTA = _code("\033[35m")
RED = _code("\033[31m")

INDENT = "  "
MAX_CONVERSATION_TURNS = 20
EXIT_COMMANDS = {"exit", "quit"}

def print_banner() -> None:
    """
    Displays the welcome banner and system architectural mode.
    """
    print(f"\n{BOLD}{MAGENTA}======================================================================{RESET}")
    print(f"{BOLD}{CYAN} DEMES {RESET}{DIM}— Deconstructive Meaning Encoding & Expressive Syntax{RESET}")
    print(f"\n{BOLD}{MAGENTA}======================================================================{RESET}")
    print(f"{DIM}Neuro-Symbolic NLU Engine | Local Symbolic Core + Local SLM Stylist{RESET}")
    print(f"{DIM}Type 'exit' or 'quit' to end the session.{RESET}\n")


def print_transparency_view(logical_form: Any, payload: Dict[str, Any], pragmatics: Dict[str, Any]) -> None:
    """
    Prints an explicit, color-coded inspection view of what DEMES structurally
    understood from the utterance BEFORE displaying the styled output.
    """
    print(f"{DIM}[DEMES Core Inspection]{RESET}")
    print(f"{INDENT}{CYAN}Speech Act{RESET}: {DIM}{pragmatics.get('speech_act')} / {pragmatics.get('intent')}{RESET}")

    if logical_form:
        print(f"{INDENT}{CYAN}Predicate{RESET}: {BOLD}{logical_form.predicate}{RESET}")
        print(f"{INDENT}{CYAN}Arguments{RESET}: {DIM}{logical_form.arguments}{RESET}")

        truth = payload.get("truth_value", False)
        truth_str = f"{GREEN}Validated (True){RESET}" if truth else f"{RED}Conflict / Unmapped (False){RESET}"
        print(f"{INDENT}{CYAN}Truth Val{RESET}: {truth_str}")
    else:
        reason = payload.get("reason", "Unknown parse error")
        print(f"{INDENT}{RED}Parse Failure{RESET} : {DIM}{reason}{RESET}")


def main() -> None:
    """
    Main execution loop for the DEMES terminal chatbot.
    """
    # 1. Initialize the symbolic core pipeline (lexicon, parser, WSD, pragmatics, semantics, world model).
    pipeline = DEMESPipeline()

    # 2. Initialize the presentation layer
    stylist = LocalStylist()

    print_banner()
    turn_count = 0

    while True:
        try:
            user_input = input(f"{BOLD}User > {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Session closed.{RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in EXIT_COMMANDS:
            print(f"{DIM}Exiting DEMES. Goodbye!{RESET}")
            break

        turn_count += 1
        print()

        # Step 1: Run the full NLU pipeline (parse, pragmatics, WSD, semantic evaluation).
        result = pipeline.process_utterance(user_input)
        logical_form = result["logical_form"]
        payload = result["semantics"]

        # Step 2: Print transparency inspection of internal understanding
        print_transparency_view(logical_form, payload, result["pragmatics"])

        # Step 3: Pass structured payload to presentation layer for natural styling
        rendered_response = stylist.render(payload)
        print(f"\n{BOLD}{GREEN}DEMES > {RESET}{rendered_response}\n")

        # Step 4: Check 20-message turn constraint for discourse clarity
        if turn_count >= MAX_CONVERSATION_TURNS:
            print(f"{YELLOW}[DEMES Notice] Reached {MAX_CONVERSATION_TURNS}-turn limit. Clearing active discourse referents.{RESET}\n")
            pipeline.world_model.clear_discourse()
            turn_count = 0

if __name__ == "__main__":
    main()
