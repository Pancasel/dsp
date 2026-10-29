"""Find every drug mentioned in a free-text chat message.

"tôi đang uống panadol với ibuprofen, wafarin" -> Acetaminophen, Ibuprofen, Warfarin.
The message is split into clauses. Each clause is matched exactly first, then
by the longest known name inside it, and only then by fuzzy matching on the
words left after removing chat filler words.
"""

import re
from typing import TYPE_CHECKING

from graphrag.drug_names.name_resolution_results import DrugNameResolution
from graphrag.drug_names.text_normalization import normalize_for_matching, plain_words

if TYPE_CHECKING:
    from graphrag.drug_names.drug_name_resolver import DrugResolver


# Separators between drug names: punctuation and "và / với / cùng / and / with".
DRUG_LIST_SEPARATOR_PATTERN = re.compile(
    r"\s*(?:[,;+/&\n.?!]|\bva\b|\band\b|\bvà\b|\bvới\b|\bvoi\b|\bwith\b|\bcùng\b)\s*",
    re.IGNORECASE,
)
STITCH_ID_PATTERN = re.compile(r"\bCID\d{9}\b", re.IGNORECASE)
# Filler words of chat messages, in normalize_for_matching() form (no accents).
# They are never treated as drug names and are dropped before fuzzy matching.
CHAT_FILLER_WORDS = {
    "toi", "minh", "em", "anh", "chi", "ban", "me", "bo", "ong", "ba", "con",
    "dang", "se", "da", "vua", "moi", "hay", "thuong", "nen", "them", "bot",
    "uong", "dung", "su", "an", "tiem", "boi", "xai",
    "thuoc", "loai", "vien", "lieu", "ngay", "lan",
    "voi", "cung", "va", "hoac", "kem", "chung", "nhau",
    "co", "khong", "ko", "k", "sao", "the", "nao", "gi", "la", "bi", "duoc",
    "ket", "hop", "phoi", "tac", "phu", "nguy", "hiem", "toan",
    "cho", "hoi", "xem", "kiem", "tra", "giup", "nhe", "a", "oi",
    "cua", "mot", "hai", "luc", "khi", "neu", "thi", "nay", "do", "kia",
    "xin", "chao", "hello", "hi", "cam", "on",
    "i", "im", "am", "taking", "take", "takes", "use", "using", "with", "plus",
    "please", "check", "also", "is", "it", "safe", "my", "can", "combination",
    "drug", "drugs", "medicine", "medicines", "pill", "pills", "and", "or",
    "what", "about", "side", "effects", "effect",
    # meals, time and symptoms/diseases mentioned around the drug names
    "com", "xong", "sau", "truoc", "bua", "sang", "trua", "dem", "ngu",
    "dau", "sot", "ho", "cam", "cum", "gut", "cao", "huyet", "ap", "tieu",
    "duong", "phe", "bung", "benh", "tim", "mach", "viem", "ruou", "bia",
    "lung", "khop", "day", "hong", "mui", "rang", "met", "tong",
}
# Longest drug name, in words, looked for inside a clause.
LONGEST_NAME_IN_WORDS = 5
CERTAIN_METHODS = {"exact", "brand_alias", "stitch_id"}


def split_into_drug_mentions(text: str) -> list[str]:
    """Split "Paracetamol + Codeine, ibuprofen" into separate drug strings."""
    parts = DRUG_LIST_SEPARATOR_PATTERN.split(text)
    return [part.strip() for part in parts if part and part.strip()]


def extract_drugs_from_message(resolver: "DrugResolver", message: str) -> list[DrugNameResolution]:
    found: list[DrugNameResolution] = []
    for stitch_id in STITCH_ID_PATTERN.findall(message):
        found.append(resolver.resolve(stitch_id))
    message = STITCH_ID_PATTERN.sub(" ", message)

    for clause in split_into_drug_mentions(message):
        whole_clause = resolver.resolve(clause)
        if whole_clause.method in CERTAIN_METHODS:
            found.append(whole_clause)
            continue
        # Scan the words before salt/form removal so that "potassium
        # chloride" is still seen as one name inside a sentence.
        names_in_clause = find_known_names_in_words(resolver, plain_words(clause))
        if names_in_clause:
            found.extend(names_in_clause)
            continue
        words = normalize_for_matching(clause).split()
        remaining_words = [word for word in words if word not in CHAT_FILLER_WORDS]
        if not remaining_words:
            continue
        remaining_phrase = resolver.resolve(" ".join(remaining_words))
        if remaining_phrase.method == "not_found" and len(remaining_words) > 1:
            # "gut colchicine": a leftover symptom word must not hide the drug.
            single_word_matches = []
            for word in remaining_words:
                resolution = resolver.resolve(word)
                if resolution.method != "not_found":
                    single_word_matches.append(resolution)
            if single_word_matches:
                found.extend(single_word_matches)
                continue
        found.append(remaining_phrase)
    return remove_repeated_drugs(found)


def remove_repeated_drugs(found: list[DrugNameResolution]) -> list[DrugNameResolution]:
    """Keep the first mention of each drug (or of each unresolved text)."""
    unique: list[DrugNameResolution] = []
    seen_keys: set[str] = set()
    for resolution in found:
        if resolution.stitch_id:
            key = resolution.stitch_id
        else:
            key = f"?{normalize_for_matching(resolution.query)}"
        if key not in seen_keys:
            seen_keys.add(key)
            unique.append(resolution)
    return unique


def find_known_names_in_words(
    resolver: "DrugResolver", words: list[str], longest: int = LONGEST_NAME_IN_WORDS
) -> list[DrugNameResolution]:
    """Greedy longest match of known drug names inside a list of words."""
    results = []
    word_position = 0
    while word_position < len(words):
        matched = False
        for size in range(min(longest, len(words) - word_position), 0, -1):
            phrase = normalize_for_matching(" ".join(words[word_position : word_position + size]))
            exact_entry = resolver.exact_entries.get(phrase)
            if (
                exact_entry
                and phrase not in CHAT_FILLER_WORDS
                and (size > 1 or phrase in resolver.keys_safe_as_single_word)
            ):
                stitch_id, original_text, method = exact_entry
                results.append(
                    DrugNameResolution(
                        original_text, stitch_id, resolver.names.drug_name(stitch_id), method, 100.0
                    )
                )
                word_position += size
                matched = True
                break
        if not matched:
            word_position += 1
    return results
