"""Map Decagon STITCH IDs to PubChem names and synonyms (run once, needs internet).

Decagon uses "flat" STITCH compound IDs such as CID000002244. The numeric part
is the PubChem CID, so the name comes straight from PubChem PUG REST.

Usage:
    python scripts/build_drug_names.py [--out resources/drug_names.csv]
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphrag.config import Settings  # noqa: E402
from graphrag.dataset import _rows  # noqa: E402


PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"
PUG_NAME = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
# SIDER uses the same flat STITCH IDs ("CID1" + 8 digits) and still names some
# compounds whose old PubChem CIDs were merged away.
SIDER_NAMES = "http://sideeffects.embl.de/media/download/drug_names.tsv"
OVERRIDES = Path(__file__).resolve().parents[1] / "resources" / "drug_name_overrides.csv"
STEREO_SUFFIX = re.compile(r",\s*\((?:\+-|\+/-|[RSEZ+-]+|cis|trans)\)-?$", re.IGNORECASE)
MAX_SYNONYMS = 25
# Registry numbers and database codes are useless for name matching.
CODE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^\d+-\d+-\d$",  # CAS
        r"^(unii|chebi|chembl|dtxsid|dtxcid|nsc|schembl|zinc|akos|mfcd|bdbm|cas|einecs|hsdb|ccris|brn|ec|ncgc|smr|mls|bspbio|spectrum|prestwick|kbio|sr-|hms|tox21|cid|sbi|bidd|db\d)",
        r"^[a-z]{0,4}[-\s]?\d{3,}[a-z]?$",  # lab codes such as "NSC 1234", "AB12345"
        r"\d{4,}",
        r"[\[\]{}=]",
    )
]


def stitch_ids(settings: Settings) -> list[str]:
    ids: set[str] = set()
    for row in _rows(settings.targets_file, ("STITCH",)):
        ids.add(row["STITCH"])
    if settings.mono_file:
        for row in _rows(settings.mono_file, ("STITCH",)):
            ids.add(row["STITCH"])
    for row in _rows(settings.combo_file, ("STITCH 1", "STITCH 2")):
        ids.add(row["STITCH 1"])
        ids.add(row["STITCH 2"])
    return sorted(item for item in ids if item)


def keep_synonym(name: str) -> bool:
    if len(name) > 40 or len(name) < 3:
        return False
    return not any(pattern.search(name) for pattern in CODE_PATTERNS)


TITLE_NOISE = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^\((?:\+-|\+/-|rac|rs|r|s|e|z|e/z|\+|-)\)-",  # (+-)-Adrenaline, (Rac)-, (RS)-
        r"\s*\((?:usan|usp|inn|ban|tn|jan)[^)]*\)$",
        r"\s*\((?:trihydrate|salt/mix|acetate)\)$",
        r"\s*\((?:1[+-]|[+-])\)$",
        r",\s*\((?:\+-|\+/-|[rsez+-]+|\d?a?[rs])\)-?$",
    )
]
SIDER_NOISE = {"v", "delta", "cas", "vitamin", "copolymer", "retinoic"}


def is_ugly(name: str) -> bool:
    return (
        not name
        or name.startswith("CID")
        or len(name) > 25
        or bool(re.search(r"\d|[\[\](),;:]|impurity|standard|mixture|api", name, re.IGNORECASE))
        or (name.isupper() and len(name) > 5)
    )


def strip_title(title: str) -> str:
    for pattern in TITLE_NOISE:
        title = pattern.sub("", title).strip()
    return title


def display(name: str) -> str:
    if name.isupper() and len(name) > 5:
        return name.capitalize()
    return name[:1].upper() + name[1:]


def choose_name(
    stitch: str, title: str, sider: str, synonyms: list[str], overrides: dict[str, str]
) -> str:
    """Pick a short human-readable generic name for the drug."""
    if stitch in overrides:
        return overrides[stitch]
    cleaned = strip_title(title)
    title_ok = bool(cleaned) and not is_ugly(cleaned)
    # SIDER names are lower-case generic names; old PubChem titles are
    # sometimes brand names (e.g. "Lanicor" for digoxin).
    sider_ok = (
        len(sider) >= 4
        and sider.casefold() not in SIDER_NOISE
        and not sider.isupper()
        and not is_ugly(sider)
    )
    if title_ok and (not sider_ok or sider.casefold() in cleaned.casefold()):
        return display(cleaned)
    if sider_ok:
        return display(sider)
    if title_ok:
        return display(cleaned)
    # Generic (INN) names are usually lower-case in PubChem, brand names are not.
    for name in synonyms:
        if name.islower() and not is_ugly(name):
            return display(name)
    for name in synonyms:
        if not is_ugly(name):
            return display(name)
    return cleaned.capitalize() if cleaned and cleaned.isupper() else (cleaned or stitch)


def sider_names(client: httpx.Client) -> dict[int, str]:
    try:
        response = client.get(SIDER_NAMES, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"SIDER names unavailable: {exc}")
        return {}
    names = {}
    for line in response.text.splitlines():
        parts = line.split("	")
        if len(parts) == 2 and parts[0].startswith("CID1"):
            names[int(parts[0][4:])] = parts[1]
    return names


def by_name(client: httpx.Client, name: str) -> tuple[str, list[str]]:
    """Title and synonyms of the current PubChem record for a drug name."""
    quoted = httpx.URL(f"{PUG_NAME}/{name}/synonyms/JSON")
    payload = get_json(client, str(quoted))
    info = (payload or {}).get("InformationList", {}).get("Information", [])
    if not info:
        return "", []
    synonyms = info[0].get("Synonym", [])
    return (synonyms[0] if synonyms else ""), synonyms


def get_json(client: httpx.Client, url: str) -> dict | None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="resources/drug_names.csv")
    args = parser.parse_args()
    settings = Settings.from_env()
    ids = stitch_ids(settings)
    cid_by_stitch = {stitch: int(stitch[3:]) for stitch in ids}
    print(f"{len(ids)} STITCH IDs")

    titles: dict[int, str] = {}
    synonyms: dict[int, list[str]] = {}
    cids = sorted(set(cid_by_stitch.values()))
    with httpx.Client(timeout=60.0) as client:
        for start in range(0, len(cids), 100):
            batch = cids[start : start + 100]
            joined = ",".join(map(str, batch))
            payload = get_json(client, f"{PUG}/{joined}/property/Title/JSON")
            for item in (payload or {}).get("PropertyTable", {}).get("Properties", []):
                if item.get("Title"):
                    titles[item["CID"]] = item["Title"]
            payload = get_json(client, f"{PUG}/{joined}/synonyms/JSON")
            for item in (payload or {}).get("InformationList", {}).get("Information", []):
                names = [name for name in item.get("Synonym", []) if keep_synonym(name)]
                unique: list[str] = []
                seen: set[str] = set()
                for name in names:
                    key = name.casefold()
                    if key not in seen:
                        seen.add(key)
                        unique.append(name)
                synonyms[item["CID"]] = unique[:MAX_SYNONYMS]
            print(f"  fetched {min(start + 100, len(cids))}/{len(cids)}")
            time.sleep(0.4)  # PubChem asks for at most 5 requests per second

        overrides: dict[str, str] = {}
        if OVERRIDES.is_file():
            with OVERRIDES.open(encoding="utf-8", newline="") as stream:
                overrides = {row["stitch_id"]: row["name"] for row in csv.DictReader(stream)}
        sider = sider_names(client)
        for stitch, cid in cid_by_stitch.items():
            if cid in titles:
                continue
            fallback = sider.get(cid) or overrides.get(stitch)
            if not fallback:
                continue
            title, names = by_name(client, fallback)
            titles[cid] = title or fallback
            synonyms[cid] = [name for name in names if keep_synonym(name)][:MAX_SYNONYMS]
            if fallback.casefold() not in {n.casefold() for n in synonyms[cid]}:
                synonyms[cid].insert(0, fallback)
            print(f"  {stitch}: recovered as {titles[cid]} (via {fallback})")
            time.sleep(0.3)

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[1] / out
    out.parent.mkdir(parents=True, exist_ok=True)
    missing = []
    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stitch_id", "pubchem_cid", "name", "synonyms"])
        for stitch in ids:
            cid = cid_by_stitch[stitch]
            title = titles.get(cid, "")
            name = choose_name(stitch, title, sider.get(cid, ""), synonyms.get(cid, []), overrides)
            if name == stitch:
                missing.append(stitch)
            # Keep the PubChem title and SIDER name as matchable aliases.
            extra = [strip_title(title), sider.get(cid, "")]
            names = []
            for item in extra + synonyms.get(cid, []):
                if item and item.casefold() != name.casefold() and item.casefold() not in {n.casefold() for n in names}:
                    if item.startswith("CID") or len(item) > 40:
                        continue
                    names.append(item)
            writer.writerow([stitch, cid, name, "|".join(names)])
    print(f"wrote {out} ({len(ids) - len(missing)}/{len(ids)} named)")
    if missing:
        print("unnamed:", ", ".join(missing))


if __name__ == "__main__":
    main()
