"""Human-readable names for drugs, genes and side effects.

All files are optional: when a resource is missing the index falls back to the
source identifiers, so the API still works on the raw Decagon data.
"""

import csv
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


# Strength and dosage-form words that appear on packaging but are not part of
# the drug name ("Panadol 500mg viên nén" -> "panadol").
_STRENGTH = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ug|g|ml|iu|ui|%)\b(?:\s*/\s*\d*\s*(?:ml|g))?",
    re.IGNORECASE,
)
_FORM_WORDS = {
    "vien", "nen", "nang", "sui", "bao", "phim", "film", "coated", "tablet",
    "tablets", "tab", "capsule", "capsules", "cap", "caps", "siro", "syrup",
    "ong", "tiem", "injection", "hop", "vi", "goi", "sr", "xr", "er", "retard",
    "plus", "forte", "extra", "kid", "kids", "tre", "em",
    # salt forms: "losartan kali" -> "losartan"
    "kali", "natri", "sodium", "potassium", "hydrochloride", "hcl", "maleate",
    "besylate", "besilat", "mesylate", "tartrate", "succinate", "citrate",
}
_SALT_WORDS = {
    "kali", "natri", "sodium", "potassium", "hydrochloride", "hcl", "maleate",
    "besylate", "besilat", "mesylate", "tartrate", "succinate", "citrate",
}
# When only a counter-ion is left the salt itself is the drug: "sodium
# chloride" and "potassium chloride" must not both become "chloride".
_ION_WORDS = {
    "chloride", "clorid", "bicarbonate", "bicarbonat", "carbonate", "carbonat",
    "sulfate", "sulphate", "sulfat", "phosphate", "phosphat", "iodide", "iodid",
    "fluoride", "fluorid", "oxide", "hydroxide", "acetate", "lactate", "gluconate",
    "monochloride", "citrate", "chloric",
}


def tokenize(text: str) -> list[str]:
    """Lower-case words without accents, strengths or bare numbers."""
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("®", " ").replace("™", " ")
    text = _STRENGTH.sub(" ", text.casefold())
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    # Bare numbers are strengths without a unit ("Hapacol 250").
    return [word for word in text.split() if not word.isdigit()]


def normalize(text: str) -> str:
    """Lower-case, strip Vietnamese accents, strengths and dosage-form words."""
    words = tokenize(text)
    kept = [word for word in words if word not in _FORM_WORDS]
    if kept and all(word in _ION_WORDS for word in kept):
        kept = [word for word in words if word not in _FORM_WORDS - _SALT_WORDS]
    return " ".join(kept).strip()


def _word_pattern(words: list[str], plural: bool = False) -> re.Pattern[str] | None:
    if not words:
        return None
    suffix = r"(?:es|s)?" if plural else ""

    def alternative(word: str) -> str:
        # Greek plurals: thrombosis -> thromboses, agranulocytosis -> -oses.
        if plural and word.endswith("sis"):
            return re.escape(word[:-3]) + "s[ie]s"
        return re.escape(word)

    alternatives = "|".join(alternative(word) for word in words)
    return re.compile(r"\b(?:" + alternatives + r")" + suffix + r"\b")


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


@dataclass
class DrugName:
    stitch_id: str
    name: str
    synonyms: list[str] = field(default_factory=list)
    aliases_vi: list[str] = field(default_factory=list)


class NameIndex:
    def __init__(
        self,
        drugs: dict[str, DrugName],
        genes: dict[str, str],
        side_effects_vi: dict[str, tuple[str, str]],
        serious_keywords: list[str],
        serious_labels: dict[str, bool] | None = None,
        exclude_keywords: list[str] | None = None,
        meddra_terms: dict[str, str] | None = None,
    ) -> None:
        self.drugs = drugs
        self.genes = genes
        self.side_effects_vi = side_effects_vi
        self.serious_keywords = serious_keywords
        # Labels from the EMA Important Medical Event list
        # (scripts/build_serious_effects.py); they win over the keywords.
        self.serious_labels = serious_labels or {}
        # Decagon name -> MedDRA term it maps to ("excess potassium" -> Hyperkalaemia).
        self.meddra_terms = meddra_terms or {}
        # Whole-word match with an optional plural: "coma" must not flag
        # "glaucoma", "thrombosis" also flags "thromboses".
        self._serious_pattern = _word_pattern(serious_keywords, plural=True)
        # Lab tests, procedures and social terms ("glasgow coma scale",
        # "death of parent") are never serious by keyword.
        self._exclude_pattern = _word_pattern(exclude_keywords or [])

    @classmethod
    def load(
        cls,
        drug_names_file: Path | None,
        brand_aliases_file: Path | None,
        gene_names_file: Path | None,
        side_effects_vi_file: Path | None,
        serious_keywords_file: Path | None,
        serious_effects_file: Path | None = None,
    ) -> "NameIndex":
        drugs: dict[str, DrugName] = {}
        for row in _read_csv(drug_names_file):
            synonyms = [item for item in row.get("synonyms", "").split("|") if item]
            drugs[row["stitch_id"]] = DrugName(row["stitch_id"], row["name"], synonyms)
        for row in _read_csv(brand_aliases_file):
            drug = drugs.get(row.get("stitch_id", ""))
            if drug and row["alias"] not in drug.aliases_vi:
                drug.aliases_vi.append(row["alias"])
        genes = {row["gene_id"]: row["symbol"] for row in _read_csv(gene_names_file)}
        side_effects_vi = {
            row["name_en"].casefold(): (
                row.get("display_en") or row["name_en"],
                row.get("name_vi") or "",
            )
            for row in _read_csv(side_effects_vi_file)
        }
        keyword_rows = [row for row in _read_csv(serious_keywords_file) if row.get("keyword")]
        keywords = [row["keyword"].casefold() for row in keyword_rows if row.get("category") != "exclude"]
        excluded = [row["keyword"].casefold() for row in keyword_rows if row.get("category") == "exclude"]
        label_rows = [row for row in _read_csv(serious_effects_file) if row.get("name_en")]
        labels = {row["name_en"].casefold(): row.get("serious") == "1" for row in label_rows}
        terms = {row["name_en"].casefold(): row["meddra_pt"] for row in label_rows if row.get("meddra_pt")}
        return cls(drugs, genes, side_effects_vi, keywords, labels, excluded, terms)

    def drug_name(self, stitch_id: str) -> str:
        drug = self.drugs.get(stitch_id)
        return drug.name if drug else stitch_id

    def drug_label(self, stitch_id: str) -> str:
        """Name plus ID, the form sent to the LLM so it can cite either."""
        name = self.drug_name(stitch_id)
        return name if name == stitch_id else f"{name} ({stitch_id})"

    def drug_display(self, stitch_id: str) -> dict[str, object]:
        drug = self.drugs.get(stitch_id)
        return {
            "stitch_id": stitch_id,
            "name": drug.name if drug else stitch_id,
            "aliases_vi": list(drug.aliases_vi) if drug else [],
            "synonyms": list(drug.synonyms[:8]) if drug else [],
        }

    def gene_symbol(self, gene_id: str) -> str:
        return self.genes.get(gene_id, gene_id)

    def side_effect_en(self, name: str) -> str:
        entry = self.side_effects_vi.get(name.casefold())
        return entry[0] if entry else name

    def side_effect_vi(self, name: str) -> str:
        """Vietnamese label, or the English label when no translation exists."""
        entry = self.side_effects_vi.get(name.casefold())
        return entry[1] if entry and entry[1] else self.side_effect_en(name)

    def is_serious(self, name: str) -> bool:
        label = self.serious_labels.get(name.casefold())
        if label is not None:
            return label
        return self.is_critical(name)

    def is_critical(self, name: str) -> bool:
        """Life-threatening class (bleeding, arrhythmia, organ failure, ...).

        Matched on resources/serious_keywords.csv, against the Decagon name and
        its MedDRA term, whatever the IME label. Used only to order effects
        that are already serious.
        """
        if self._serious_pattern is None:
            return False
        texts = [self.side_effect_en(name).casefold()]
        term = self.meddra_terms.get(name.casefold())
        if term:
            texts.append(term.casefold())
        return any(
            self._serious_pattern.search(text)
            and not (self._exclude_pattern is not None and self._exclude_pattern.search(text))
            for text in texts
        )

    def node_label(self, node: str, drugs: set[str]) -> str:
        """Readable label for a path endpoint (drug or protein)."""
        if node in drugs or node in self.drugs:
            return self.drug_name(node)
        return self.gene_symbol(node)
