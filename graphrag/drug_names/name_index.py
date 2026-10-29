"""Readable names for drugs, genes and side effects, read from the files in resources/.

Every file is optional: when one is missing the index falls back to the
source identifiers, so the system still works on the raw Decagon data.
"""

from dataclasses import dataclass, field
from pathlib import Path

from graphrag.drug_names.side_effect_seriousness import SideEffectSeriousness
from graphrag.source_data.csv_file_reader import read_all_rows_if_file_exists


@dataclass
class DrugName:
    stitch_id: str
    name: str
    # Other names from PubChem (resources/drug_names.csv).
    synonyms: list[str] = field(default_factory=list)
    # Brand names sold in Vietnam (resources/brand_aliases_vietnamese.csv).
    vietnamese_brand_aliases: list[str] = field(default_factory=list)


class NameIndex:
    def __init__(
        self,
        *,
        drugs_by_id: dict[str, DrugName],
        gene_symbols_by_id: dict[str, str],
        side_effect_translations: dict[str, tuple[str, str]],
        critical_keywords: list[str],
        serious_labels: dict[str, bool] | None = None,
        excluded_keywords: list[str] | None = None,
        meddra_terms: dict[str, str] | None = None,
    ) -> None:
        self.drugs_by_id = drugs_by_id
        # Entrez gene ID -> gene symbol ("5743" -> "PTGS2").
        self.gene_symbols_by_id = gene_symbols_by_id
        # Lower-case Decagon name -> (English display name, Vietnamese name).
        self.side_effect_translations = side_effect_translations
        self.seriousness = SideEffectSeriousness(
            critical_keywords=critical_keywords,
            excluded_keywords=excluded_keywords or [],
            serious_labels=serious_labels or {},
            meddra_terms=meddra_terms or {},
            english_display_name=self.side_effect_english,
        )

    @classmethod
    def load(
        cls,
        *,
        drug_names_file: Path | None,
        brand_aliases_file: Path | None,
        gene_names_file: Path | None,
        side_effect_translations_file: Path | None,
        serious_keywords_file: Path | None,
        serious_effects_file: Path | None = None,
    ) -> "NameIndex":
        drugs_by_id: dict[str, DrugName] = {}
        for row in read_all_rows_if_file_exists(drug_names_file):
            synonyms = [synonym for synonym in row.get("synonyms", "").split("|") if synonym]
            drugs_by_id[row["stitch_id"]] = DrugName(row["stitch_id"], row["name"], synonyms)
        for row in read_all_rows_if_file_exists(brand_aliases_file):
            drug = drugs_by_id.get(row.get("stitch_id", ""))
            if drug and row["alias"] not in drug.vietnamese_brand_aliases:
                drug.vietnamese_brand_aliases.append(row["alias"])

        gene_symbols_by_id = {
            row["gene_id"]: row["symbol"] for row in read_all_rows_if_file_exists(gene_names_file)
        }

        # Columns: name_en (Decagon name), display_en (corrected English), name_vi (Vietnamese).
        side_effect_translations = {}
        for row in read_all_rows_if_file_exists(side_effect_translations_file):
            english_display_name = row.get("display_en") or row["name_en"]
            vietnamese_name = row.get("name_vi") or ""
            side_effect_translations[row["name_en"].casefold()] = (
                english_display_name,
                vietnamese_name,
            )

        # category "exclude" marks words that are never critical; every other category is critical.
        critical_keywords = []
        excluded_keywords = []
        for row in read_all_rows_if_file_exists(serious_keywords_file):
            if not row.get("keyword"):
                continue
            if row.get("category") == "exclude":
                excluded_keywords.append(row["keyword"].casefold())
            else:
                critical_keywords.append(row["keyword"].casefold())

        # Columns: name_en, serious ("1" or "0"), meddra_pt (MedDRA preferred term).
        serious_labels = {}
        meddra_terms = {}
        for row in read_all_rows_if_file_exists(serious_effects_file):
            if not row.get("name_en"):
                continue
            serious_labels[row["name_en"].casefold()] = row.get("serious") == "1"
            if row.get("meddra_pt"):
                meddra_terms[row["name_en"].casefold()] = row["meddra_pt"]

        return cls(
            drugs_by_id=drugs_by_id,
            gene_symbols_by_id=gene_symbols_by_id,
            side_effect_translations=side_effect_translations,
            critical_keywords=critical_keywords,
            serious_labels=serious_labels,
            excluded_keywords=excluded_keywords,
            meddra_terms=meddra_terms,
        )

    def drug_name(self, stitch_id: str) -> str:
        drug = self.drugs_by_id.get(stitch_id)
        if drug:
            return drug.name
        return stitch_id

    def drug_name_with_id(self, stitch_id: str) -> str:
        """"Warfarin (CID000006691)", the form sent to the LLM so that it can cite either."""
        name = self.drug_name(stitch_id)
        if name == stitch_id:
            return name
        return f"{name} ({stitch_id})"

    def drug_identity(self, stitch_id: str) -> dict[str, object]:
        """ID, name, Vietnamese brands and the first PubChem synonyms of a drug."""
        drug = self.drugs_by_id.get(stitch_id)
        if drug is None:
            return {"stitch_id": stitch_id, "name": stitch_id, "aliases_vi": [], "synonyms": []}
        return {
            "stitch_id": stitch_id,
            "name": drug.name,
            "aliases_vi": list(drug.vietnamese_brand_aliases),
            "synonyms": list(drug.synonyms[:8]),
        }

    def gene_symbol(self, gene_id: str) -> str:
        return self.gene_symbols_by_id.get(gene_id, gene_id)

    def side_effect_english(self, effect_name: str) -> str:
        translation = self.side_effect_translations.get(effect_name.casefold())
        if translation:
            return translation[0]
        return effect_name

    def side_effect_vietnamese(self, effect_name: str) -> str:
        """Vietnamese name, or the English name when there is no translation."""
        translation = self.side_effect_translations.get(effect_name.casefold())
        if translation and translation[1]:
            return translation[1]
        return self.side_effect_english(effect_name)

    def is_serious(self, effect_name: str) -> bool:
        return self.seriousness.is_serious(effect_name)

    def is_critical(self, effect_name: str) -> bool:
        return self.seriousness.is_critical(effect_name)

    def path_node_label(self, node: str, drugs: set[str]) -> str:
        """Readable label for one end of a path edge: a drug name or a gene symbol."""
        if node in drugs or node in self.drugs_by_id:
            return self.drug_name(node)
        return self.gene_symbol(node)
