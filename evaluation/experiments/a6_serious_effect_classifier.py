"""A6: how well do keyword lists find serious side effects, compared with the EMA IME list?

Evaluated on the side-effect names that match an IME term exactly
(resources/serious_effects.csv, method "ime_match" or "excluded").
"""

from graphrag.drug_names.side_effect_seriousness import build_keyword_pattern
from evaluation.shared.result_csv_files import read_csv_file, write_result_csv
from evaluation.shared.system_loading import RESOURCES_FOLDER


# serious_keywords.csv before the IME labels were introduced (commit a1f2615).
# "|" separates multi-word keywords; single words inside a part are separated by spaces.
LEGACY_KEYWORDS = (
    "haemorrhage hemorrhage bleeding haematemesis melaena arrhythmia fibrillation|tachycardia ventricular|"
    "torsade|qt prolonged|cardiac arrest|cardiac failure|heart failure|myocardial infarction|bradycardia|"
    "heart block|hypotension|renal failure|kidney failure|acute kidney|nephropathy|hepatic failure|"
    "liver failure|hepatitis|hepatotoxicity|jaundice|rhabdomyolysis|serotonin syndrome|neuroleptic malignant|"
    "convulsion|seizure|coma|stroke|cerebrovascular accident|respiratory depression|respiratory arrest|"
    "respiratory failure|apnoea|pulmonary embolism|thrombosis|anaphylactic|anaphylaxis|angioedema|"
    "stevens-johnson|toxic epidermal necrolysis|agranulocytosis|pancytopenia|neutropenia|thrombocytopenia|"
    "aplastic anaemia|hypoglycaemia|hyperkalaemia|hyponatraemia|lactic acidosis|pancreatitis|"
    "gastrointestinal perforation|sepsis|death"
)


def legacy_keyword_list() -> list[str]:
    keywords = []
    for part in LEGACY_KEYWORDS.split("|"):
        if " " in part:
            keywords.append(part)
        else:
            keywords.extend(part.split())
    return keywords


def run_serious_classifier_experiment() -> None:
    print("A6 serious-effect classifier vs EMA IME")
    labelled_names = [
        row
        for row in read_csv_file(RESOURCES_FOLDER / "serious_effects.csv")
        if row["method"] in {"ime_match", "excluded"}
    ]
    if not labelled_names:
        print("  resources/serious_effects.csv missing; run python -m resource_builders.build_serious_effects")
        return
    keyword_rows = read_csv_file(RESOURCES_FOLDER / "serious_keywords.csv")
    current_keywords = [row["keyword"] for row in keyword_rows if row["category"] != "exclude"]
    excluded_keywords = [row["keyword"] for row in keyword_rows if row["category"] == "exclude"]
    # Method label -> (keyword pattern, exclusion pattern or None).
    methods = {
        "Từ khóa cũ (khớp nguyên từ)": (build_keyword_pattern(legacy_keyword_list()), None),
        "Từ khóa mới (số nhiều + loại trừ)": (
            build_keyword_pattern(current_keywords, allow_plural=True),
            build_keyword_pattern(excluded_keywords),
        ),
    }
    rows = []
    for label, (keyword_pattern, exclusion_pattern) in methods.items():
        true_positives = false_positives = false_negatives = true_negatives = 0
        for row in labelled_names:
            text = row["name_en"].casefold()
            is_excluded = exclusion_pattern is not None and exclusion_pattern.search(text)
            predicted_serious = bool(keyword_pattern.search(text)) and not is_excluded
            truly_serious = row["serious"] == "1"
            true_positives += predicted_serious and truly_serious
            false_positives += predicted_serious and not truly_serious
            false_negatives += truly_serious and not predicted_serious
            true_negatives += not predicted_serious and not truly_serious
        predicted_total = true_positives + false_positives
        serious_total = true_positives + false_negatives
        precision = true_positives / predicted_total if predicted_total else 0.0
        recall = true_positives / serious_total if serious_total else 0.0
        if precision + recall:
            f1_score = round(2 * precision * recall / (precision + recall), 4)
        else:
            f1_score = 0.0
        rows.append(
            {
                "method": label, "names": len(labelled_names), "ime_serious": serious_total,
                "precision": round(precision, 4), "recall": round(recall, 4), "f1": f1_score,
                "tp": true_positives, "fp": false_positives, "fn": false_negatives, "tn": true_negatives,
            }
        )
        print(f"  {label}: precision={rows[-1]['precision']} recall={rows[-1]['recall']} f1={rows[-1]['f1']}")
    write_result_csv("a6_serious_classifier.csv", rows)
