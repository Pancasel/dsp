"""Label every Decagon side-effect name as serious or not (run once, needs internet).

Reference standard: the EMA "Important Medical Event (IME) terms list", the
MedDRA preferred terms (PT) that EU pharmacovigilance treats as medically
important. A Decagon name is serious when it maps to an IME term:

1. exact match after normalisation (case, US -> UK spelling, plural, "NOS");
2. the same words in another order ("Embolism pulmonary");
3. resources/effect_meddra_map.csv, a hand-written map for lay names
   ("heart attack" -> Myocardial infarction). IME membership of the mapped
   term still decides; the map never marks a term serious by itself.

IME terms that FAERS reports usually carry as an indication or comorbidity
rather than a drug effect are excluded (neoplasms, congenital and perinatal
conditions, community infections, chronic diseases listed in EXCLUDED_TERMS).
Names that match no IME term fall back to resources/serious_keywords.csv at
runtime (graphrag/drug_names/side_effect_seriousness.py).

Usage (from the project root):
    python -m resource_builders.build_serious_effects [--ime path/to/ime.xlsx]
Writes resources/serious_effects.csv.
"""

import argparse
import csv
import re
import sqlite3
import tempfile
import urllib.request
import xml.etree.ElementTree as ElementTree
import zipfile
from pathlib import Path

from graphrag.settings.application_settings import Settings
from graphrag.settings.environment_variable_readers import PROJECT_ROOT


IMPORTANT_MEDICAL_EVENT_LIST_URL = (
    "https://www.ema.europa.eu/en/documents/other/"
    "meddra-important-medical-event-terms-list-version-28-0_en.xlsx"
)
OUTPUT_FILE = PROJECT_ROOT / "resources" / "serious_effects.csv"
MANUAL_MEDDRA_MAP_FILE = PROJECT_ROOT / "resources" / "effect_meddra_map.csv"
# Namespace of the XML inside an .xlsx file.
SPREADSHEET_NAMESPACE = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
MINIMUM_EXPECTED_TERMS = 1000

# MedDRA System Organ Classes (SOC) whose terms are left out.
EXCLUDED_ORGAN_CLASSES = {
    "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
    "Congenital, familial and genetic disorders",
    "Pregnancy, puerperium and perinatal conditions",
    "Social circumstances",
    "Product issues",
    "Surgical and medical procedures",
}
# Infections are kept only when drugs commonly cause or unmask them
# (sepsis, opportunistic and aspiration infections, C. difficile colitis).
DRUG_RELATED_INFECTION_PATTERN = re.compile(
    r"sepsis|septic|bacteraem|urosepsis|toxic shock|pseudomembranous|pneumocystis|"
    r"leukoencephalopathy|cryptococc|histoplasm|aspergill|mycobacter|candida|"
    r"cytomegalovirus|pneumonia|gangrene|meningitis aseptic",
    re.IGNORECASE,
)
# Chronic diseases, indications and perinatal terms (lower-case PT names).
EXCLUDED_TERMS = {
    "acromegaly", "alcohol abuse", "substance abuse", "schizophrenia",
    "schizoaffective disorder", "bipolar disorder", "rheumatoid arthritis",
    "ankylosing spondylitis", "multiple sclerosis", "myasthenia gravis",
    "coeliac disease", "inflammatory bowel disease", "scleroderma",
    "mixed connective tissue disease", "dermatomyositis", "polymyositis",
    "polymyalgia rheumatica", "rheumatic fever", "rheumatic heart disease",
    "rickets", "osteochondrosis", "chronic kidney disease", "diabetic nephropathy",
    "iga nephropathy", "hypoglycaemia neonatal", "jaundice neonatal",
    "neonatal respiratory distress syndrome", "retinopathy of prematurity",
    "cerebral palsy", "leriche syndrome", "carotid artery stenosis",
    "aortic stenosis", "pneumoconiosis", "radiation pneumonitis",
    "spinal cord injury", "peripheral nerve injury", "narcolepsy", "cataplexy",
    "carcinoid syndrome", "amyloidosis", "corneal dystrophy", "corneal degeneration",
    "retinal degeneration", "fracture nonunion",
}
# (American spelling pattern, British spelling), anchored so that "chemo" stays intact.
AMERICAN_TO_BRITISH_SPELLING = [
    (r"hemorrh", "haemorrh"), (r"(?<![a-z])an?emia", "anaemia"), (r"(?<![a-z])edema", "oedema"),
    (r"edema", "oedema"), (r"(?<![a-z])esophag", "oesophag"), (r"ischemi", "ischaemi"),
    (r"leukemia", "leukaemia"), (r"glycemia", "glycaemia"), (r"kalemia", "kalaemia"),
    (r"natremia", "natraemia"), (r"calcemia", "calcaemia"), (r"(?<![a-z])apnea", "apnoea"),
    (r"diarrhea", "diarrhoea"), (r"(?<![a-z])hem(?=at|o)", "haem"), (r"(?<![a-z])fetal", "foetal"),
    (r"tumor", "tumour"), (r"pediatric", "paediatric"), (r"anesth", "anaesth"),
    (r"dyspnea", "dyspnoea"), (r"anemia", "anaemia"),
]


def normalize_term(name: str) -> str:
    """Lower case, British spelling, without "NOS" ("not otherwise specified")."""
    text = name.casefold()
    for american_spelling, british_spelling in AMERICAN_TO_BRITISH_SPELLING:
        text = re.sub(american_spelling, british_spelling, text)
    text = re.sub(r"\bnos\b", " ", text)
    return " ".join(text.split())


def singular_form(text: str) -> str:
    return re.sub(r"(?<=[a-z])(?:es|s)$", "", text)


def read_important_medical_events(path: Path) -> dict[str, str]:
    """Preferred term -> organ class, from the EMA spreadsheet (no openpyxl needed)."""
    namespace = SPREADSHEET_NAMESPACE
    text_tag = f"{{{namespace['m']}}}t"
    with zipfile.ZipFile(path) as archive:
        shared_strings_xml = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
        shared_strings = [
            "".join(node.text or "" for node in item.iter(text_tag))
            for item in shared_strings_xml.findall("m:si", namespace)
        ]
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    organ_class_by_term: dict[str, str] = {}
    for row in sheet.iter(f"{{{namespace['m']}}}row"):
        values = []
        for cell in row.findall("m:c", namespace):
            value = cell.find("m:v", namespace)
            if value is None:
                values.append("")
            elif cell.get("t") == "s":  # the cell holds the index of a shared string
                values.append(shared_strings[int(value.text)])
            else:
                values.append(value.text or "")
        if len(values) >= 3 and values[0].isdigit():
            organ_class_by_term[values[1]] = values[2]
    if len(organ_class_by_term) < MINIMUM_EXPECTED_TERMS:
        raise ValueError(f"Unexpected IME file layout: only {len(organ_class_by_term)} terms read")
    return organ_class_by_term


def decagon_side_effect_names(settings: Settings) -> list[tuple[str, str]]:
    """(name, "combo" or "mono") for every drug pair and single-drug side-effect name in the cache."""
    with sqlite3.connect(settings.cache_database_file) as connection:
        pair_names = [name for (name,) in connection.execute("SELECT effect_name FROM effect_pair_counts")]
        single_drug_names = [
            name for (name,) in connection.execute("SELECT DISTINCT effect_name FROM single_drug_side_effects")
        ]
    first_by_lowercase: dict[str, tuple[str, str]] = {}
    for name in pair_names:
        first_by_lowercase.setdefault(name.casefold(), (name, "combo"))
    for name in single_drug_names:
        first_by_lowercase.setdefault(name.casefold(), (name, "mono"))

    def lowercase_name(item: tuple[str, str]) -> str:
        return item[0].casefold()

    return sorted(first_by_lowercase.values(), key=lowercase_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ime", type=Path, help="local copy of the EMA IME .xlsx")
    arguments = parser.parse_args()
    settings = Settings.from_environment()
    if arguments.ime:
        organ_class_by_term = read_important_medical_events(arguments.ime)
    else:
        with tempfile.TemporaryDirectory() as folder:
            downloaded_file = Path(folder) / "ime.xlsx"
            urllib.request.urlretrieve(IMPORTANT_MEDICAL_EVENT_LIST_URL, downloaded_file)
            organ_class_by_term = read_important_medical_events(downloaded_file)
    print(f"IME terms: {len(organ_class_by_term)}")

    term_by_normalized_name = {normalize_term(term): term for term in organ_class_by_term}
    term_by_word_set = {frozenset(normalize_term(term).split()): term for term in organ_class_by_term}
    with MANUAL_MEDDRA_MAP_FILE.open(encoding="utf-8", newline="") as stream:
        manual_term_by_name = {row["name_en"].casefold(): row["meddra_pt"] for row in csv.DictReader(stream)}
    mapped_terms_not_in_list = sorted(
        {term for term in manual_term_by_name.values() if term not in organ_class_by_term}
    )
    print(f"mapped terms that are not IME ({len(mapped_terms_not_in_list)}): {', '.join(mapped_terms_not_in_list)}")

    rows = []
    for name, vocabulary in decagon_side_effect_names(settings):
        normalized_name = normalize_term(name)
        if name.casefold() in manual_term_by_name:
            preferred_term = manual_term_by_name[name.casefold()]
            method = "manual_map"
            if preferred_term not in organ_class_by_term:
                rows.append({"name_en": name, "vocabulary": vocabulary, "serious": 0,
                             "method": "manual_map_not_ime", "meddra_pt": preferred_term, "soc": ""})
                continue
        else:
            preferred_term = (
                term_by_normalized_name.get(normalized_name)
                or term_by_normalized_name.get(singular_form(normalized_name))
                or term_by_word_set.get(frozenset(normalized_name.split()))
            )
            method = "ime_match"
        if preferred_term is None:
            continue  # decided at runtime by the keyword fallback
        organ_class = organ_class_by_term[preferred_term]
        is_excluded = (
            organ_class in EXCLUDED_ORGAN_CLASSES
            or preferred_term.casefold() in EXCLUDED_TERMS
            or (
                organ_class == "Infections and infestations"
                and not DRUG_RELATED_INFECTION_PATTERN.search(preferred_term)
            )
        )
        rows.append({"name_en": name, "vocabulary": vocabulary, "serious": 0 if is_excluded else 1,
                     "method": "excluded" if is_excluded else method, "meddra_pt": preferred_term,
                     "soc": organ_class})

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pair_rows = [row for row in rows if row["vocabulary"] == "combo"]
    print(f"labelled {len(rows)} names ({len(pair_rows)} combo); serious: "
          f"{sum(row['serious'] for row in rows)} ({sum(row['serious'] for row in pair_rows)} combo)")
    print(f"-> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
