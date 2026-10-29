"""Turn one piece of user text (generic name, Vietnamese brand, STITCH ID, typo) into a drug ID.

Order: STITCH ID -> exact normalized name/synonym/brand -> fuzzy match.
A fuzzy match is accepted automatically only at score >= ACCEPT_SCORE; between
SUGGEST_SCORE and ACCEPT_SCORE the caller gets candidates to confirm.
Finding several drugs inside a chat message is in chat_message_drug_extraction.py.
"""

from graphrag.drug_names.chat_message_drug_extraction import extract_drugs_from_message
from graphrag.drug_names.fuzzy_name_matching import find_fuzzy_candidates
from graphrag.drug_names.name_index import NameIndex
from graphrag.drug_names.name_resolution_results import DrugNameResolution, MatchCandidate
from graphrag.drug_names.text_normalization import normalize_for_matching


ACCEPT_SCORE = 90.0
# Plain character similarity required for automatic acceptance. It separates
# typos ("ibuprofn", ratio 94) from different drugs ("esomeprazole" vs
# "omeprazole", ratio 91) and partial words ("vitamin c" vs "vitamin d2").
ACCEPT_RATIO = 92.0
# The best candidate must beat the second one by at least this many points.
ACCEPT_MARGIN_OVER_RUNNER_UP = 3
SUGGEST_SCORE = 75.0
# Known names shorter than this (after normalization) are not indexed.
MIN_KEY_LENGTH = 3
# PubChem synonyms that are ordinary words or abbreviations, never a drug on
# their own ("vitamin" -> Ergocalciferol, "CoM" -> Mesna, "delta" -> Dronabinol).
GENERIC_SYNONYMS = {
    "vitamin", "delta", "com", "stim", "hex", "cas", "mer", "dup", "aldo", "oxis",
}
# A single word inside a sentence counts as a drug only if it is a primary
# name, a curated brand, or a synonym at least this long without digits.
SCAN_MIN_SYNONYM_LENGTH = 5

# Priority of a known name when two names normalize to the same key.
BRAND_ALIAS_PRIORITY = 2
PRIMARY_NAME_PRIORITY = 1
SYNONYM_PRIORITY = 0


class DrugResolver:
    def __init__(self, names: NameIndex, drug_ids: set[str]) -> None:
        self.names = names
        self.drug_ids = drug_ids
        # Normalized known name -> (stitch_id, original text, method).
        self.exact_entries: dict[str, tuple[str, str, str]] = {}
        # Keys shared by several drugs ("penicillin", "bisphosphonate"): the
        # user must choose; the first drug in ID order must not silently win.
        self.ambiguous_entries: dict[str, list[str]] = {}
        self._build_name_entries()
        self.known_keys = list(self.exact_entries)
        self.keys_safe_as_single_word = self._keys_safe_as_single_word()

    def _build_name_entries(self) -> None:
        # Normalized key -> stitch_id -> (priority, original text).
        best_name_by_key: dict[str, dict[str, tuple[int, str]]] = {}
        for stitch_id in sorted(self.drug_ids):
            drug = self.names.drugs_by_id.get(stitch_id)
            if drug is None:
                continue
            known_names = [(BRAND_ALIAS_PRIORITY, alias) for alias in drug.vietnamese_brand_aliases]
            known_names.append((PRIMARY_NAME_PRIORITY, drug.name))
            known_names += [(SYNONYM_PRIORITY, synonym) for synonym in drug.synonyms]
            for priority, text in known_names:
                key = normalize_for_matching(text)
                if len(key) < MIN_KEY_LENGTH:
                    continue
                if priority == SYNONYM_PRIORITY and key in GENERIC_SYNONYMS:
                    continue
                current = best_name_by_key.setdefault(key, {}).get(stitch_id)
                if current is None or priority > current[0]:
                    best_name_by_key[key][stitch_id] = (priority, text)

        for key, names_by_drug in best_name_by_key.items():
            highest_priority = max(priority for priority, _text in names_by_drug.values())
            drugs_with_highest_priority = sorted(
                stitch_id
                for stitch_id, (priority, _text) in names_by_drug.items()
                if priority == highest_priority
            )
            if len(drugs_with_highest_priority) > 1:
                self.ambiguous_entries[key] = drugs_with_highest_priority
                continue
            stitch_id = drugs_with_highest_priority[0]
            priority, text = names_by_drug[stitch_id]
            method = "brand_alias" if priority == BRAND_ALIAS_PRIORITY else "exact"
            self.exact_entries[key] = (stitch_id, text, method)

    def _keys_safe_as_single_word(self) -> set[str]:
        safe_keys = set()
        for key, (stitch_id, text, method) in self.exact_entries.items():
            is_long_synonym_without_digits = len(key) >= SCAN_MIN_SYNONYM_LENGTH and not any(
                character.isdigit() for character in key
            )
            if (
                " " in key
                or method == "brand_alias"
                or text == self.names.drugs_by_id[stitch_id].name
                or is_long_synonym_without_digits
            ):
                safe_keys.add(key)
        return safe_keys

    def resolve(self, query: str) -> DrugNameResolution:
        raw_text = query.strip()
        upper_text = raw_text.upper()
        if upper_text in self.drug_ids:
            return DrugNameResolution(
                raw_text, upper_text, self.names.drug_name(upper_text), "stitch_id", 100.0
            )

        key = normalize_for_matching(raw_text)
        if not key:
            return DrugNameResolution(raw_text, None, None, "not_found", 0.0)
        exact_entry = self.exact_entries.get(key)
        if exact_entry:
            stitch_id, _text, method = exact_entry
            return DrugNameResolution(
                raw_text, stitch_id, self.names.drug_name(stitch_id), method, 100.0
            )
        drugs_sharing_key = self.ambiguous_entries.get(key)
        if drugs_sharing_key:
            candidates = [
                MatchCandidate(stitch_id, self.names.drug_name(stitch_id), raw_text, 100.0, 100.0)
                for stitch_id in drugs_sharing_key
            ]
            return DrugNameResolution(raw_text, None, None, "ambiguous", 100.0, candidates)

        candidates = find_fuzzy_candidates(key, self.known_keys, self.exact_entries, self.names)
        if not candidates:
            return DrugNameResolution(raw_text, None, None, "not_found", 0.0, [])
        best = candidates[0]
        if best.score < SUGGEST_SCORE:
            return DrugNameResolution(raw_text, None, None, "not_found", best.score, candidates[:3])
        runner_up_score = candidates[1].score if len(candidates) > 1 else 0.0
        # Accept only a clear winner; two close candidates must be confirmed.
        is_clear_winner = (
            best.score >= ACCEPT_SCORE
            and best.ratio >= ACCEPT_RATIO
            and best.score - runner_up_score >= ACCEPT_MARGIN_OVER_RUNNER_UP
        )
        if is_clear_winner:
            return DrugNameResolution(
                raw_text, best.stitch_id, best.name, "fuzzy", best.score, candidates
            )
        return DrugNameResolution(raw_text, None, None, "ambiguous", best.score, candidates)

    def extract(self, message: str) -> list[DrugNameResolution]:
        """All drugs mentioned in a chat message (see chat_message_drug_extraction.py)."""
        return extract_drugs_from_message(self, message)
