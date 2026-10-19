"""Turn user text (generic name, Vietnamese brand, STITCH ID, typo) into a drug ID.

Order: STITCH ID -> exact normalized name/synonym/brand -> fuzzy match.
A fuzzy match is accepted automatically only at score >= ACCEPT_SCORE; between
SUGGEST_SCORE and ACCEPT_SCORE the caller gets candidates to confirm.
"""

import re
from dataclasses import asdict, dataclass, field

from rapidfuzz import fuzz, process

from graphrag.names import NameIndex, normalize, tokenize


ACCEPT_SCORE = 90.0
# Plain character similarity required for automatic acceptance. It separates
# typos ("ibuprofn", ratio 94) from different drugs ("esomeprazole" vs
# "omeprazole", ratio 91) and partial words ("vitamin c" vs "vitamin d2").
ACCEPT_RATIO = 92.0
SUGGEST_SCORE = 75.0
MIN_KEY_LENGTH = 3
_SPLIT = re.compile(
    r"\s*(?:[,;+/&\n.?!]|\bva\b|\band\b|\bvà\b|\bvới\b|\bvoi\b|\bwith\b|\bcùng\b)\s*",
    re.IGNORECASE,
)
_STITCH = re.compile(r"\bCID\d{9}\b", re.IGNORECASE)
# Filler words of chat messages, in normalize() form (no accents). They are
# never treated as drug names and are dropped before fuzzy matching.
CHAT_STOPWORDS = {
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
# PubChem synonyms that are ordinary words or abbreviations, never a drug on
# their own ("vitamin" -> Ergocalciferol, "CoM" -> Mesna, "delta" -> Dronabinol).
GENERIC_SYNONYMS = {
    "vitamin", "delta", "com", "stim", "hex", "cas", "mer", "dup", "aldo", "oxis",
}
# A single word inside a sentence counts as a drug only if it is a primary
# name, a curated brand, or a synonym at least this long without digits.
SCAN_MIN_SYNONYM_LENGTH = 5


@dataclass
class Candidate:
    stitch_id: str
    name: str
    matched: str
    score: float
    ratio: float = 0.0


@dataclass
class ResolveResult:
    query: str
    stitch_id: str | None
    name: str | None
    method: str  # stitch_id | exact | brand_alias | fuzzy | ambiguous | not_found
    score: float
    candidates: list[Candidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def split_query(text: str) -> list[str]:
    """Split "Paracetamol + Codeine, ibuprofen" into separate drug strings."""
    return [part.strip() for part in _SPLIT.split(text) if part and part.strip()]


class DrugResolver:
    def __init__(self, names: NameIndex, drug_ids: set[str]) -> None:
        self.names = names
        self.drug_ids = drug_ids
        # normalized key -> (stitch_id, original text, method)
        self._exact: dict[str, tuple[str, str, str]] = {}
        # Keys shared by several drugs ("penicillin", "bisphosphonate"): the
        # user must choose, the first drug in ID order must not silently win.
        self._ambiguous: dict[str, list[str]] = {}
        # Priority: curated brand > primary name > PubChem synonym.
        sources: dict[str, dict[str, tuple[int, str]]] = {}
        for stitch_id in sorted(drug_ids):
            drug = names.drugs.get(stitch_id)
            if drug is None:
                continue
            entries = [(2, alias) for alias in drug.aliases_vi]
            entries += [(1, drug.name)] + [(0, text) for text in drug.synonyms]
            for rank, text in entries:
                key = normalize(text)
                if len(key) < MIN_KEY_LENGTH or (rank == 0 and key in GENERIC_SYNONYMS):
                    continue
                current = sources.setdefault(key, {}).get(stitch_id)
                if current is None or rank > current[0]:
                    sources[key][stitch_id] = (rank, text)
        for key, owners in sources.items():
            best = max(rank for rank, _ in owners.values())
            top = sorted(stitch_id for stitch_id, (rank, _) in owners.items() if rank == best)
            if len(top) > 1:
                self._ambiguous[key] = top
                continue
            rank, text = owners[top[0]]
            self._exact[key] = (top[0], text, "brand_alias" if rank == 2 else "exact")
        self._keys = list(self._exact)
        self._scan_safe = {
            key
            for key, (stitch_id, text, method) in self._exact.items()
            if " " in key
            or method == "brand_alias"
            or text == names.drugs[stitch_id].name
            or (len(key) >= SCAN_MIN_SYNONYM_LENGTH and not any(c.isdigit() for c in key))
        }

    def resolve(self, query: str) -> ResolveResult:
        raw = query.strip()
        upper = raw.upper()
        if upper in self.drug_ids:
            return ResolveResult(raw, upper, self.names.drug_name(upper), "stitch_id", 100.0)

        key = normalize(raw)
        if not key:
            return ResolveResult(raw, None, None, "not_found", 0.0)
        hit = self._exact.get(key)
        if hit:
            stitch_id, _, method = hit
            return ResolveResult(raw, stitch_id, self.names.drug_name(stitch_id), method, 100.0)
        shared = self._ambiguous.get(key)
        if shared:
            candidates = [
                Candidate(stitch_id, self.names.drug_name(stitch_id), raw, 100.0, 100.0)
                for stitch_id in shared
            ]
            return ResolveResult(raw, None, None, "ambiguous", 100.0, candidates)

        candidates = self._fuzzy(key)
        if not candidates or candidates[0].score < SUGGEST_SCORE:
            return ResolveResult(raw, None, None, "not_found", candidates[0].score if candidates else 0.0, candidates[:3])
        best = candidates[0]
        runner_up = candidates[1].score if len(candidates) > 1 else 0.0
        # Accept only a clear winner; two close candidates must be confirmed.
        if (
            best.score >= ACCEPT_SCORE
            and best.ratio >= ACCEPT_RATIO
            and best.score - runner_up >= 3
        ):
            return ResolveResult(raw, best.stitch_id, best.name, "fuzzy", best.score, candidates)
        return ResolveResult(raw, None, None, "ambiguous", best.score, candidates)

    def extract(self, text: str) -> list[ResolveResult]:
        """Find the drugs mentioned in a free-text chat message.

        "tôi đang uống panadol với ibuprofen, wafarin" -> Acetaminophen,
        Ibuprofen, Warfarin. Each clause is matched exactly first, then by the
        longest known name inside it, and only then by fuzzy matching on the
        words left after removing chat filler words.
        """
        found: list[ResolveResult] = []
        for stitch_id in _STITCH.findall(text):
            found.append(self.resolve(stitch_id))
        text = _STITCH.sub(" ", text)
        for segment in split_query(text):
            whole = self.resolve(segment)
            if whole.method in {"exact", "brand_alias", "stitch_id"}:
                found.append(whole)
                continue
            # Scan the words before salt/form removal so that "potassium
            # chloride" is still seen as one name inside a sentence.
            matches = self._scan_known_names(tokenize(segment))
            if matches:
                found.extend(matches)
                continue
            words = normalize(segment).split()
            remaining = [word for word in words if word not in CHAT_STOPWORDS]
            if not remaining:
                continue
            phrase = self.resolve(" ".join(remaining))
            if phrase.method == "not_found" and len(remaining) > 1:
                # "gut colchicine": a leftover symptom word must not hide the drug.
                singles = [self.resolve(word) for word in remaining]
                singles = [item for item in singles if item.method != "not_found"]
                if singles:
                    found.extend(singles)
                    continue
            found.append(phrase)
        unique: list[ResolveResult] = []
        seen: set[str] = set()
        for result in found:
            key = result.stitch_id or f"?{normalize(result.query)}"
            if key not in seen:
                seen.add(key)
                unique.append(result)
        return unique

    def _scan_known_names(self, words: list[str], longest: int = 5) -> list[ResolveResult]:
        """Greedy longest match of known drug names inside a list of words."""
        results = []
        index = 0
        while index < len(words):
            for size in range(min(longest, len(words) - index), 0, -1):
                phrase = normalize(" ".join(words[index : index + size]))
                hit = self._exact.get(phrase)
                if (
                    hit
                    and phrase not in CHAT_STOPWORDS
                    and (size > 1 or phrase in self._scan_safe)
                ):
                    stitch_id, original, method = hit
                    results.append(
                        ResolveResult(original, stitch_id, self.names.drug_name(stitch_id), method, 100.0)
                    )
                    index += size
                    break
            else:
                index += 1
        return results

    def _fuzzy(self, key: str, limit: int = 5) -> list[Candidate]:
        matches = process.extract(
            key,
            self._keys,
            scorer=fuzz.WRatio,
            limit=40,
            score_cutoff=60,
        )
        best: dict[str, Candidate] = {}
        for matched_key, _, index in matches:
            # WRatio rewards short substrings; blend with plain ratio so that
            # "aspirin" does not match "aspirin pH8" better than a real typo fix.
            ratio = fuzz.ratio(key, matched_key)
            score = round(0.5 * fuzz.WRatio(key, matched_key) + 0.5 * ratio, 1)
            stitch_id, original, _ = self._exact[self._keys[index]]
            current = best.get(stitch_id)
            if current is None or score > current.score:
                best[stitch_id] = Candidate(
                    stitch_id, self.names.drug_name(stitch_id), original, score, round(ratio, 1)
                )
        return sorted(best.values(), key=lambda item: -item.score)[:limit]
