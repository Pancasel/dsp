"""Translate Decagon side-effect names to Vietnamese once (CALLS THE LLM API).

Collects unique names from the drug pair and single-drug files, sends them in
batches to the OpenAI-compatible endpoint configured in .env, and appends the
results to resources/side_effect_translations_vietnamese.csv. Names already
translated are skipped, so the script can be re-run after a failure.

Usage (from the project root):
    python -m resource_builders.translate_side_effect_names [--batch 80] [--workers 4]
        [--model MODEL] [--only-combo]
"""

import argparse
import asyncio
import csv
import json
import sqlite3

import httpx

from graphrag.settings.application_settings import Settings
from graphrag.settings.environment_variable_readers import PROJECT_ROOT
from graphrag.source_data.csv_file_reader import read_csv_rows_with_required_columns


OUTPUT_FILE = PROJECT_ROOT / "resources" / "side_effect_translations_vietnamese.csv"
# Display fixes for known typos in the source vocabulary. The source data is
# left untouched; only the English display name changes.
ENGLISH_DISPLAY_FIXES = {"neumonia": "pneumonia"}
REQUEST_ATTEMPTS = 4

TRANSLATION_PROMPT = (
    "Bạn là dược sĩ dịch thuật ngữ y khoa (MedDRA/UMLS) sang tiếng Việt. "
    "Dịch từng tên tác dụng phụ tiếng Anh sang thuật ngữ y khoa tiếng Việt chuẩn, "
    "ngắn gọn, dễ hiểu với người bệnh (ví dụ 'arterial pressure NOS decreased' -> "
    "'huyết áp động mạch giảm', 'anaemia' -> 'thiếu máu'). Bỏ các ký hiệu NOS. "
    "Trả về DUY NHẤT một JSON object, khóa là chuỗi tiếng Anh giữ nguyên y hệt, "
    "giá trị là bản dịch tiếng Việt."
)


def drug_pair_effect_names(settings: Settings) -> list[str]:
    """From the SQLite cache when it exists, otherwise from the CSV file."""
    if settings.cache_database_file.is_file():
        with sqlite3.connect(settings.cache_database_file) as connection:
            try:
                return [
                    name
                    for (name,) in connection.execute(
                        "SELECT DISTINCT effect_name FROM drug_pair_side_effects"
                    )
                ]
            except sqlite3.DatabaseError:
                pass
    rows = read_csv_rows_with_required_columns(settings.drug_pair_side_effects_file, ("Side Effect Name",))
    return sorted({row["Side Effect Name"] for row in rows})


def single_drug_effect_names(settings: Settings) -> list[str]:
    if settings.single_drug_side_effects_file is None:
        return []
    rows = read_csv_rows_with_required_columns(settings.single_drug_side_effects_file, ("Side Effect Name",))
    return sorted({row["Side Effect Name"] for row in rows})


def load_existing_translations() -> dict[str, str]:
    if not OUTPUT_FILE.is_file():
        return {}
    with OUTPUT_FILE.open(encoding="utf-8", newline="") as stream:
        return {row["name_en"]: row["name_vi"] for row in csv.DictReader(stream)}


def parse_json_object(text: str) -> dict[str, str]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    payload = json.loads(text[start : end + 1])
    return {str(key): str(value).strip() for key, value in payload.items()}


async def translate_batch(
    client: httpx.AsyncClient, settings: Settings, model: str, batch: list[str]
) -> dict[str, str]:
    """English name -> Vietnamese name; an empty dict after REQUEST_ATTEMPTS failures."""
    for attempt in range(REQUEST_ATTEMPTS):
        try:
            response = await client.post(
                f"{settings.llm_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.llm_api_key}",
                    "HTTP-Referer": "http://127.0.0.1:8000",
                    "X-Title": "Decagon Polypharmacy Graph-RAG",
                },
                json={
                    "model": model,
                    "response_format": {"type": "json_object"},
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": TRANSLATION_PROMPT},
                        {"role": "user", "content": json.dumps(batch, ensure_ascii=False)},
                    ],
                },
            )
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            translations = parse_json_object(message.get("content") or "")
            return {name: translations[name] for name in batch if translations.get(name)}
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as error:
            print(f"  retry {attempt + 1}: {error.__class__.__name__}: {str(error)[:120]}")
            await asyncio.sleep(3 * (attempt + 1))
    return {}


async def translate_all(arguments: argparse.Namespace) -> None:
    settings = Settings.from_environment()
    if not settings.llm_api_key:
        raise SystemExit("LLM_API_KEY is not configured in .env")
    model = arguments.model or settings.llm_model
    names = drug_pair_effect_names(settings)
    if not arguments.only_combo:
        already_listed = set(names)
        names += [name for name in single_drug_effect_names(settings) if name not in already_listed]
    existing = load_existing_translations()
    names_to_translate = [name for name in names if name and name not in existing]
    print(f"{len(names)} names, {len(existing)} already translated, {len(names_to_translate)} to go ({model})")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not OUTPUT_FILE.is_file()
    stream = OUTPUT_FILE.open("a", encoding="utf-8", newline="")
    writer = csv.writer(stream)
    if is_new_file:
        writer.writerow(["name_en", "display_en", "name_vi"])

    batches = [
        names_to_translate[start : start + arguments.batch]
        for start in range(0, len(names_to_translate), arguments.batch)
    ]
    parallel_request_limit = asyncio.Semaphore(arguments.workers)
    finished_batch_count = 0

    async def translate_and_write(batch: list[str]) -> None:
        nonlocal finished_batch_count
        async with parallel_request_limit:
            translations = await translate_batch(client, settings, model, batch)
        for name in batch:
            if name in translations:
                english_display_name = ENGLISH_DISPLAY_FIXES.get(name.casefold(), name)
                writer.writerow([name, english_display_name, translations[name]])
        stream.flush()
        finished_batch_count += 1
        print(f"  batch {finished_batch_count}/{len(batches)}: {len(translations)}/{len(batch)} translated")

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=15.0)) as client:
            await asyncio.gather(*(translate_and_write(batch) for batch in batches))
    finally:
        stream.close()
    still_missing = [name for name in names if name not in load_existing_translations()]
    print(f"done; {len(still_missing)} names still untranslated (re-run to retry)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=80, help="names per request")
    parser.add_argument("--workers", type=int, default=4, help="requests sent in parallel")
    parser.add_argument("--model", default="")
    parser.add_argument("--only-combo", action="store_true", help="skip single-drug side-effect names")
    asyncio.run(translate_all(parser.parse_args()))


if __name__ == "__main__":
    main()
