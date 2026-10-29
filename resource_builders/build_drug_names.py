"""Map Decagon STITCH IDs to PubChem names and synonyms (run once, needs internet).

Decagon uses "flat" STITCH compound IDs such as CID000002244. The numeric part
is the PubChem CID (compound ID), so the name comes straight from PubChem PUG
REST (PubChem's web API). Compounds whose old PubChem CID was merged away are
recovered through their SIDER name or resources/drug_name_overrides.csv.

Usage (from the project root):
    python -m resource_builders.build_drug_names [--out resources/drug_names.csv]
"""

import argparse
import csv
import re
import time
from pathlib import Path

import httpx

from graphrag.settings.application_settings import Settings
from graphrag.settings.environment_variable_readers import PROJECT_ROOT
from graphrag.source_data.csv_file_reader import read_csv_rows_with_required_columns


PUBCHEM_COMPOUND_BY_ID_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"
PUBCHEM_COMPOUND_BY_NAME_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
# SIDER uses the same flat STITCH IDs ("CID1" + 8 digits) and still names some
# compounds whose old PubChem CIDs were merged away.
SIDER_DRUG_NAMES_URL = "http://sideeffects.embl.de/media/download/drug_names.tsv"
NAME_OVERRIDES_FILE = PROJECT_ROOT / "resources" / "drug_name_overrides.csv"
MAX_SYNONYMS = 25
PUBCHEM_BATCH_SIZE = 100
# Registry numbers and database codes are useless for name matching.
DATABASE_CODE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^\d+-\d+-\d$",  # CAS registry number
        r"^(unii|chebi|chembl|dtxsid|dtxcid|nsc|schembl|zinc|akos|mfcd|bdbm|cas|einecs|hsdb|ccris|brn|ec|ncgc|smr|mls|bspbio|spectrum|prestwick|kbio|sr-|hms|tox21|cid|sbi|bidd|db\d)",
        r"^[a-z]{0,4}[-\s]?\d{3,}[a-z]?$",  # lab codes such as "NSC 1234", "AB12345"
        r"\d{4,}",
        r"[\[\]{}=]",
    )
]
# Stereo-chemistry and pharmacopoeia markers removed from PubChem titles.
TITLE_NOISE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^\((?:\+-|\+/-|rac|rs|r|s|e|z|e/z|\+|-)\)-",  # (+-)-Adrenaline, (Rac)-, (RS)-
        r"\s*\((?:usan|usp|inn|ban|tn|jan)[^)]*\)$",
        r"\s*\((?:trihydrate|salt/mix|acetate)\)$",
        r"\s*\((?:1[+-]|[+-])\)$",
        r",\s*\((?:\+-|\+/-|[rsez+-]+|\d?a?[rs])\)-?$",
    )
]
# SIDER names that are ordinary words, never a drug name on their own.
SIDER_GENERIC_WORDS = {"v", "delta", "cas", "vitamin", "copolymer", "retinoic"}
MAX_READABLE_NAME_LENGTH = 25
MAX_SYNONYM_LENGTH = 40


def all_stitch_ids(settings: Settings) -> list[str]:
    stitch_ids: set[str] = set()
    for row in read_csv_rows_with_required_columns(settings.drug_targets_file, ("STITCH",)):
        stitch_ids.add(row["STITCH"])
    if settings.single_drug_side_effects_file:
        for row in read_csv_rows_with_required_columns(settings.single_drug_side_effects_file, ("STITCH",)):
            stitch_ids.add(row["STITCH"])
    for row in read_csv_rows_with_required_columns(settings.drug_pair_side_effects_file, ("STITCH 1", "STITCH 2")):
        stitch_ids.add(row["STITCH 1"])
        stitch_ids.add(row["STITCH 2"])
    return sorted(stitch_id for stitch_id in stitch_ids if stitch_id)


def is_useful_synonym(name: str) -> bool:
    if len(name) > MAX_SYNONYM_LENGTH or len(name) < 3:
        return False
    return not any(pattern.search(name) for pattern in DATABASE_CODE_PATTERNS)


def is_hard_to_read(name: str) -> bool:
    """Empty, an ID, too long, containing digits or brackets, or written in capitals."""
    return (
        not name
        or name.startswith("CID")
        or len(name) > MAX_READABLE_NAME_LENGTH
        or bool(re.search(r"\d|[\[\](),;:]|impurity|standard|mixture|api", name, re.IGNORECASE))
        or (name.isupper() and len(name) > 5)
    )


def remove_title_noise(title: str) -> str:
    for pattern in TITLE_NOISE_PATTERNS:
        title = pattern.sub("", title).strip()
    return title


def display_form(name: str) -> str:
    """"WARFARIN" -> "Warfarin", "warfarin" -> "Warfarin"."""
    if name.isupper() and len(name) > 5:
        return name.capitalize()
    return name[:1].upper() + name[1:]


def choose_drug_name(
    stitch_id: str, pubchem_title: str, sider_name: str, synonyms: list[str], overrides: dict[str, str]
) -> str:
    """Pick a short human-readable generic name for the drug."""
    if stitch_id in overrides:
        return overrides[stitch_id]
    cleaned_title = remove_title_noise(pubchem_title)
    title_is_usable = bool(cleaned_title) and not is_hard_to_read(cleaned_title)
    # SIDER names are lower-case generic names; old PubChem titles are
    # sometimes brand names (e.g. "Lanicor" for digoxin).
    sider_name_is_usable = (
        len(sider_name) >= 4
        and sider_name.casefold() not in SIDER_GENERIC_WORDS
        and not sider_name.isupper()
        and not is_hard_to_read(sider_name)
    )
    title_agrees_with_sider = sider_name.casefold() in cleaned_title.casefold()
    if title_is_usable and (not sider_name_is_usable or title_agrees_with_sider):
        return display_form(cleaned_title)
    if sider_name_is_usable:
        return display_form(sider_name)
    if title_is_usable:
        return display_form(cleaned_title)
    # Generic (INN) names are usually lower-case in PubChem, brand names are not.
    for synonym in synonyms:
        if synonym.islower() and not is_hard_to_read(synonym):
            return display_form(synonym)
    for synonym in synonyms:
        if not is_hard_to_read(synonym):
            return display_form(synonym)
    if cleaned_title and cleaned_title.isupper():
        return cleaned_title.capitalize()
    return cleaned_title or stitch_id


def download_sider_names(client: httpx.Client) -> dict[int, str]:
    """PubChem CID -> SIDER drug name; empty when SIDER cannot be reached."""
    try:
        response = client.get(SIDER_DRUG_NAMES_URL, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as error:
        print(f"SIDER names unavailable: {error}")
        return {}
    names_by_cid = {}
    for line in response.text.splitlines():
        columns = line.split("\t")
        if len(columns) == 2 and columns[0].startswith("CID1"):
            names_by_cid[int(columns[0][4:])] = columns[1]
    return names_by_cid


def pubchem_title_and_synonyms_by_name(client: httpx.Client, name: str) -> tuple[str, list[str]]:
    """Title and synonyms of the current PubChem record for a drug name."""
    url = httpx.URL(f"{PUBCHEM_COMPOUND_BY_NAME_URL}/{name}/synonyms/JSON")
    payload = get_json_with_retries(client, str(url))
    information = (payload or {}).get("InformationList", {}).get("Information", [])
    if not information:
        return "", []
    synonyms = information[0].get("Synonym", [])
    return (synonyms[0] if synonyms else ""), synonyms


def get_json_with_retries(client: httpx.Client, url: str) -> dict | None:
    """None for 404; waits 1, 2, 4, 8, 16 s between failed attempts."""
    for attempt in range(5):
        try:
            response = client.get(url)
        except httpx.HTTPError:
            time.sleep(2**attempt)
            continue
        if response.status_code == 404:
            return None
        if response.status_code in (429, 503):
            time.sleep(2**attempt)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"PubChem kept failing for {url}")


def unique_ignoring_case(names: list[str]) -> list[str]:
    unique: list[str] = []
    seen_keys: set[str] = set()
    for name in names:
        if name.casefold() not in seen_keys:
            seen_keys.add(name.casefold())
            unique.append(name)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="resources/drug_names.csv")
    arguments = parser.parse_args()
    settings = Settings.from_environment()
    stitch_ids = all_stitch_ids(settings)
    cid_by_stitch_id = {stitch_id: int(stitch_id[3:]) for stitch_id in stitch_ids}
    print(f"{len(stitch_ids)} STITCH IDs")

    titles: dict[int, str] = {}
    synonyms_by_cid: dict[int, list[str]] = {}
    cids = sorted(set(cid_by_stitch_id.values()))
    with httpx.Client(timeout=60.0) as client:
        for start in range(0, len(cids), PUBCHEM_BATCH_SIZE):
            batch = cids[start : start + PUBCHEM_BATCH_SIZE]
            joined_cids = ",".join(map(str, batch))
            payload = get_json_with_retries(client, f"{PUBCHEM_COMPOUND_BY_ID_URL}/{joined_cids}/property/Title/JSON")
            for item in (payload or {}).get("PropertyTable", {}).get("Properties", []):
                if item.get("Title"):
                    titles[item["CID"]] = item["Title"]
            payload = get_json_with_retries(client, f"{PUBCHEM_COMPOUND_BY_ID_URL}/{joined_cids}/synonyms/JSON")
            for item in (payload or {}).get("InformationList", {}).get("Information", []):
                useful = [name for name in item.get("Synonym", []) if is_useful_synonym(name)]
                synonyms_by_cid[item["CID"]] = unique_ignoring_case(useful)[:MAX_SYNONYMS]
            print(f"  fetched {min(start + PUBCHEM_BATCH_SIZE, len(cids))}/{len(cids)}")
            time.sleep(0.4)  # PubChem asks for at most 5 requests per second

        overrides: dict[str, str] = {}
        if NAME_OVERRIDES_FILE.is_file():
            with NAME_OVERRIDES_FILE.open(encoding="utf-8", newline="") as stream:
                overrides = {row["stitch_id"]: row["name"] for row in csv.DictReader(stream)}
        sider_names = download_sider_names(client)
        for stitch_id, cid in cid_by_stitch_id.items():
            if cid in titles:
                continue
            fallback_name = sider_names.get(cid) or overrides.get(stitch_id)
            if not fallback_name:
                continue
            title, names = pubchem_title_and_synonyms_by_name(client, fallback_name)
            titles[cid] = title or fallback_name
            synonyms_by_cid[cid] = [name for name in names if is_useful_synonym(name)][:MAX_SYNONYMS]
            known_lowercase = {name.casefold() for name in synonyms_by_cid[cid]}
            if fallback_name.casefold() not in known_lowercase:
                synonyms_by_cid[cid].insert(0, fallback_name)
            print(f"  {stitch_id}: recovered as {titles[cid]} (via {fallback_name})")
            time.sleep(0.3)

    output_file = Path(arguments.out)
    if not output_file.is_absolute():
        output_file = PROJECT_ROOT / output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    unnamed_ids = []
    with output_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stitch_id", "pubchem_cid", "name", "synonyms"])
        for stitch_id in stitch_ids:
            cid = cid_by_stitch_id[stitch_id]
            title = titles.get(cid, "")
            sider_name = sider_names.get(cid, "")
            name = choose_drug_name(stitch_id, title, sider_name, synonyms_by_cid.get(cid, []), overrides)
            if name == stitch_id:
                unnamed_ids.append(stitch_id)
            # Keep the PubChem title and SIDER name as matchable aliases.
            other_names = []
            for candidate in [remove_title_noise(title), sider_name] + synonyms_by_cid.get(cid, []):
                if not candidate or candidate.casefold() == name.casefold():
                    continue
                if candidate.casefold() in {other.casefold() for other in other_names}:
                    continue
                if candidate.startswith("CID") or len(candidate) > MAX_SYNONYM_LENGTH:
                    continue
                other_names.append(candidate)
            writer.writerow([stitch_id, cid, name, "|".join(other_names)])
    print(f"wrote {output_file} ({len(stitch_ids) - len(unnamed_ids)}/{len(stitch_ids)} named)")
    if unnamed_ids:
        print("unnamed:", ", ".join(unnamed_ids))


if __name__ == "__main__":
    main()
