"""Label every Decagon side-effect name as serious or not (run once, needs internet).

Reference standard: the EMA "Important Medical Event (IME) terms list", the
MedDRA preferred terms that EU pharmacovigilance treats as medically important.
A Decagon name is serious when it maps to an IME term:

1. exact match after normalisation (case, US -> UK spelling, plural, "NOS");
2. the same words in another order ("Embolism pulmonary");
3. resources/effect_meddra_map.csv, a hand-written map for lay names
   ("heart attack" -> Myocardial infarction). IME membership of the mapped
   term still decides; the map never marks a term serious by itself.

IME terms that FAERS reports usually carry as an indication or comorbidity
rather than a drug effect are excluded (neoplasms, congenital and perinatal
conditions, community infections, chronic diseases listed in EXCLUDED_TERMS).
Names that match no IME term fall back to resources/serious_keywords.csv at
runtime (graphrag/names.py).

Usage:
    python scripts/build_serious_effects.py [--ime path/to/ime.xlsx]
Writes resources/serious_effects.csv.
"""

import argparse
import csv
import re
import sqlite3
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphrag.config import PROJECT_ROOT, Settings  # noqa: E402

IME_URL = (
    "https://www.ema.europa.eu/en/documents/other/"
    "meddra-important-medical-event-terms-list-version-28-0_en.xlsx"
)
OUT = PROJECT_ROOT / "resources" / "serious_effects.csv"
MAP_FILE = PROJECT_ROOT / "resources" / "effect_meddra_map.csv"

EXCLUDED_SOCS = {
    "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
    "Congenital, familial and genetic disorders",
    "Pregnancy, puerperium and perinatal conditions",
    "Social circumstances",
    "Product issues",
    "Surgical and medical procedures",
}
# Infections are kept only when drugs commonly cause or unmask them
# (sepsis, opportunistic and aspiration infections, C. difficile colitis).
DRUG_RELATED_INFECTION = re.compile(
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
_SPELLING = [  # US -> UK spelling, anchored so that "chemo" stays intact
    (r"hemorrh", "haemorrh"), (r"(?<![a-z])an?emia", "anaemia"), (r"(?<![a-z])edema", "oedema"),
    (r"edema", "oedema"), (r"(?<![a-z])esophag", "oesophag"), (r"ischemi", "ischaemi"),
    (r"leukemia", "leukaemia"), (r"glycemia", "glycaemia"), (r"kalemia", "kalaemia"),
    (r"natremia", "natraemia"), (r"calcemia", "calcaemia"), (r"(?<![a-z])apnea", "apnoea"),
    (r"diarrhea", "diarrhoea"), (r"(?<![a-z])hem(?=at|o)", "haem"), (r"(?<![a-z])fetal", "foetal"),
    (r"tumor", "tumour"), (r"pediatric", "paediatric"), (r"anesth", "anaesth"),
    (r"dyspnea", "dyspnoea"), (r"anemia", "anaemia"),
]


def normalize(name: str) -> str:
    text = name.casefold()
    for us, uk in _SPELLING:
        text = re.sub(us, uk, text)
    text = re.sub(r"\bnos\b", " ", text)
    return " ".join(text.split())


def singular(text: str) -> str:
    return re.sub(r"(?<=[a-z])(?:es|s)$", "", text)


def read_ime(path: Path) -> dict[str, str]:
    """PT name -> SOC name from the EMA spreadsheet (no openpyxl needed)."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared = [
            "".join(node.text or "" for node in item.iter(f"{{{ns['m']}}}t"))
            for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", ns)
        ]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    terms: dict[str, str] = {}
    for row in sheet.iter(f"{{{ns['m']}}}row"):
        values = []
        for cell in row.findall("m:c", ns):
            value = cell.find("m:v", ns)
            if value is None:
                values.append("")
            elif cell.get("t") == "s":
                values.append(shared[int(value.text)])
            else:
                values.append(value.text or "")
        if len(values) >= 3 and values[0].isdigit():
            terms[values[1]] = values[2]
    if len(terms) < 1000:
        raise ValueError(f"Unexpected IME file layout: only {len(terms)} terms read")
    return terms


def vocabulary(settings: Settings) -> list[tuple[str, str]]:
    """(name, source) for every combo and mono side-effect name in the cache."""
    with sqlite3.connect(settings.cache_db) as connection:
        combo = [name for (name,) in connection.execute("SELECT effect_name FROM effect_counts")]
        mono = [name for (name,) in connection.execute("SELECT DISTINCT effect_name FROM mono")]
    seen: dict[str, tuple[str, str]] = {}
    for name in combo:
        seen.setdefault(name.casefold(), (name, "combo"))
    for name in mono:
        seen.setdefault(name.casefold(), (name, "mono"))
    return sorted(seen.values(), key=lambda item: item[0].casefold())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ime", type=Path, help="local copy of the EMA IME .xlsx")
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.ime:
        ime = read_ime(args.ime)
    else:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ime.xlsx"
            urllib.request.urlretrieve(IME_URL, path)
            ime = read_ime(path)
    print(f"IME terms: {len(ime)}")

    by_norm = {normalize(term): term for term in ime}
    by_words = {frozenset(normalize(term).split()): term for term in ime}
    with MAP_FILE.open(encoding="utf-8", newline="") as stream:
        manual = {row["name_en"].casefold(): row["meddra_pt"] for row in csv.DictReader(stream)}
    missing = sorted({pt for pt in manual.values() if pt not in ime})
    print(f"mapped terms that are not IME ({len(missing)}): {', '.join(missing)}")

    rows = []
    for name, source in vocabulary(settings):
        key = normalize(name)
        if name.casefold() in manual:
            pt, method = manual[name.casefold()], "manual_map"
            if pt not in ime:
                rows.append({"name_en": name, "vocabulary": source, "serious": 0,
                             "method": "manual_map_not_ime", "meddra_pt": pt, "soc": ""})
                continue
        else:
            pt = by_norm.get(key) or by_norm.get(singular(key)) or by_words.get(frozenset(key.split()))
            method = "ime_match"
        if pt is None:
            continue  # decided at runtime by the keyword fallback
        soc = ime[pt]
        excluded = (
            soc in EXCLUDED_SOCS
            or pt.casefold() in EXCLUDED_TERMS
            or (soc == "Infections and infestations" and not DRUG_RELATED_INFECTION.search(pt))
        )
        rows.append({"name_en": name, "vocabulary": source, "serious": 0 if excluded else 1,
                     "method": "excluded" if excluded else method, "meddra_pt": pt, "soc": soc})

    with OUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    combo = [row for row in rows if row["vocabulary"] == "combo"]
    print(f"labelled {len(rows)} names ({len(combo)} combo); serious: "
          f"{sum(row['serious'] for row in rows)} ({sum(row['serious'] for row in combo)} combo)")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
