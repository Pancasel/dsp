"""Decide whether a side effect is serious, and whether it is life-threatening ("critical").

- serious: label from the EMA Important Medical Event list (resources/serious_effects.csv);
  effects without a label fall back to the critical keywords.
- critical: a keyword of resources/serious_keywords.csv (bleeding, arrhythmia,
  organ failure, ...) found in the effect name or its MedDRA term. Used only
  to order effects that are already serious.
"""

import re
from typing import Callable


def build_keyword_pattern(
    keywords: list[str], allow_plural: bool = False
) -> re.Pattern[str] | None:
    """Whole-word regular expression for any keyword; None when there are no keywords.

    With allow_plural, "thrombosis" also matches "thromboses" and other words
    may end in "s" or "es". Whole-word matching keeps "coma" from matching "glaucoma".
    """
    if not keywords:
        return None
    plural_ending = r"(?:es|s)?" if allow_plural else ""

    def keyword_alternative(keyword: str) -> str:
        # Greek plurals: thrombosis -> thromboses, agranulocytosis -> -oses.
        if allow_plural and keyword.endswith("sis"):
            return re.escape(keyword[:-3]) + "s[ie]s"
        return re.escape(keyword)

    alternatives = "|".join(keyword_alternative(keyword) for keyword in keywords)
    return re.compile(r"\b(?:" + alternatives + r")" + plural_ending + r"\b")


class SideEffectSeriousness:
    def __init__(
        self,
        *,
        critical_keywords: list[str],
        excluded_keywords: list[str],
        serious_labels: dict[str, bool],
        meddra_terms: dict[str, str],
        english_display_name: Callable[[str], str],
    ) -> None:
        self.critical_keywords = critical_keywords
        self.excluded_keywords = excluded_keywords
        # Lower-case Decagon name -> True/False from the EMA list.
        self.serious_labels = serious_labels
        # Lower-case Decagon name -> MedDRA term ("excess potassium" -> Hyperkalaemia).
        self.meddra_terms = meddra_terms
        self.english_display_name = english_display_name
        self._critical_pattern = build_keyword_pattern(critical_keywords, allow_plural=True)
        # Lab tests, procedures and social terms ("glasgow coma scale",
        # "death of parent") are never critical by keyword.
        self._excluded_pattern = build_keyword_pattern(excluded_keywords)

    def is_serious(self, effect_name: str) -> bool:
        label = self.serious_labels.get(effect_name.casefold())
        if label is not None:
            return label
        return self.is_critical(effect_name)

    def is_critical(self, effect_name: str) -> bool:
        if self._critical_pattern is None:
            return False
        texts_to_check = [self.english_display_name(effect_name).casefold()]
        meddra_term = self.meddra_terms.get(effect_name.casefold())
        if meddra_term:
            texts_to_check.append(meddra_term.casefold())
        for text in texts_to_check:
            has_critical_keyword = self._critical_pattern.search(text)
            has_excluded_keyword = (
                self._excluded_pattern is not None and self._excluded_pattern.search(text)
            )
            if has_critical_keyword and not has_excluded_keyword:
                return True
        return False
