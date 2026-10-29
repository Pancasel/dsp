"""Turn a drug name as people type it into a plain form used for matching.

"Panadol 500mg viên nén" -> "panadol", "Losartan kali" -> "losartan".
Vietnamese accents, dose strengths, dosage-form words and salt words are removed.
"""

import re
import unicodedata


# A dose strength with its unit: "500mg", "2,5 mg", "5 mg/5 ml", "0.1%".
DOSE_STRENGTH_PATTERN = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ug|g|ml|iu|ui|%)\b(?:\s*/\s*\d*\s*(?:ml|g))?",
    re.IGNORECASE,
)
# Packaging and dosage-form words (Vietnamese without accents, and English).
DOSAGE_FORM_WORDS = {
    "vien", "nen", "nang", "sui", "bao", "phim", "film", "coated", "tablet",
    "tablets", "tab", "capsule", "capsules", "cap", "caps", "siro", "syrup",
    "ong", "tiem", "injection", "hop", "vi", "goi", "sr", "xr", "er", "retard",
    "plus", "forte", "extra", "kid", "kids", "tre", "em",
}
# Salt forms: "losartan kali" -> "losartan".
SALT_WORDS = {
    "kali", "natri", "sodium", "potassium", "hydrochloride", "hcl", "maleate",
    "besylate", "besilat", "mesylate", "tartrate", "succinate", "citrate",
}
# When only a counter-ion is left, the salt itself is the drug: "sodium
# chloride" and "potassium chloride" must not both become "chloride".
COUNTER_ION_WORDS = {
    "chloride", "clorid", "bicarbonate", "bicarbonat", "carbonate", "carbonat",
    "sulfate", "sulphate", "sulfat", "phosphate", "phosphat", "iodide", "iodid",
    "fluoride", "fluorid", "oxide", "hydroxide", "acetate", "lactate", "gluconate",
    "monochloride", "citrate", "chloric",
}


def plain_words(text: str) -> list[str]:
    """Lower-case words without accents, dose strengths or bare numbers."""
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.replace("®", " ").replace("™", " ")
    text = DOSE_STRENGTH_PATTERN.sub(" ", text.casefold())
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    # Bare numbers are strengths without a unit ("Hapacol 250").
    return [word for word in text.split() if not word.isdigit()]


def normalize_for_matching(text: str) -> str:
    """plain_words() without dosage-form and salt words, joined by single spaces."""
    words = plain_words(text)
    kept_words = [
        word for word in words if word not in DOSAGE_FORM_WORDS and word not in SALT_WORDS
    ]
    only_counter_ions_left = kept_words and all(word in COUNTER_ION_WORDS for word in kept_words)
    if only_counter_ions_left:
        kept_words = [word for word in words if word not in DOSAGE_FORM_WORDS]
    return " ".join(kept_words).strip()
