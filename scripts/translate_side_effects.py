"""Translate Decagon side-effect names to Vietnamese once (needs the LLM API).

Collects unique names from the combo and mono files, sends them in batches to
the OpenAI-compatible endpoint configured in .env, and appends results to
resources/side_effects_vi.csv. Already translated names are skipped, so the
script can be re-run after a failure.

Usage:
    python scripts/translate_side_effects.py [--batch 80] [--workers 4]
        [--model MODEL] [--only-combo]
"""

import argparse
import asyncio
import csv
import json
import sqlite3
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphrag.config import PROJECT_ROOT, Settings  # noqa: E402
from graphrag.dataset import _rows  # noqa: E402


OUT = PROJECT_ROOT / "resources" / "side_effects_vi.csv"
# Display fixes for known typos in the source vocabulary. The source data is
# left untouched; only the English display label changes.
DISPLAY_FIXES = {"neumonia": "pneumonia"}

PROMPT = (
    "Bạn là dược sĩ dịch thuật ngữ y khoa (MedDRA/UMLS) sang tiếng Việt. "
    "Dịch từng tên tác dụng phụ tiếng Anh sang thuật ngữ y khoa tiếng Việt chuẩn, "
    "ngắn gọn, dễ hiểu với người bệnh (ví dụ 'arterial pressure NOS decreased' -> "
    "'huyết áp động mạch giảm', 'anaemia' -> 'thiếu máu'). Bỏ các ký hiệu NOS. "
    "Trả về DUY NHẤT một JSON object, khóa là chuỗi tiếng Anh giữ nguyên y hệt, "
    "giá trị là bản dịch tiếng Việt."
)


def combo_names(settings: Settings) -> list[str]:
    if settings.cache_db.is_file():
        with sqlite3.connect(settings.cache_db) as connection:
            try:
                return [
                    name
                    for (name,) in connection.execute(
                        "SELECT DISTINCT effect_name FROM side_effects"
                    )
                ]
            except sqlite3.DatabaseError:
                pass
    return sorted(
        {
            row["Side Effect Name"]
            for row in _rows(settings.combo_file, ("Side Effect Name",))
        }
    )


def mono_names(settings: Settings) -> list[str]:
    if settings.mono_file is None:
        return []
    return sorted(
        {row["Side Effect Name"] for row in _rows(settings.mono_file, ("Side Effect Name",))}
    )


def load_done() -> dict[str, str]:
    if not OUT.is_file():
        return {}
    with OUT.open(encoding="utf-8", newline="") as stream:
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
    for attempt in range(4):
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
                        {"role": "system", "content": PROMPT},
                        {"role": "user", "content": json.dumps(batch, ensure_ascii=False)},
                    ],
                },
            )
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            result = parse_json_object(message.get("content") or "")
            return {name: result[name] for name in batch if result.get(name)}
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            print(f"  retry {attempt + 1}: {exc.__class__.__name__}: {str(exc)[:120]}")
            await asyncio.sleep(3 * (attempt + 1))
    return {}


async def run(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    if not settings.llm_api_key:
        raise SystemExit("LLM_API_KEY is not configured in .env")
    model = args.model or settings.llm_model
    names = combo_names(settings)
    if not args.only_combo:
        seen = set(names)
        names += [name for name in mono_names(settings) if name not in seen]
    done = load_done()
    todo = [name for name in names if name and name not in done]
    print(f"{len(names)} names, {len(done)} already translated, {len(todo)} to go ({model})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    new_file = not OUT.is_file()
    stream = OUT.open("a", encoding="utf-8", newline="")
    writer = csv.writer(stream)
    if new_file:
        writer.writerow(["name_en", "display_en", "name_vi"])

    batches = [todo[i : i + args.batch] for i in range(0, len(todo), args.batch)]
    semaphore = asyncio.Semaphore(args.workers)
    finished = 0

    async def worker(batch: list[str]) -> None:
        nonlocal finished
        async with semaphore:
            result = await translate_batch(client, settings, model, batch)
        for name in batch:
            if name in result:
                display_en = DISPLAY_FIXES.get(name.casefold(), name)
                writer.writerow([name, display_en, result[name]])
        stream.flush()
        finished += 1
        print(f"  batch {finished}/{len(batches)}: {len(result)}/{len(batch)} translated")

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=15.0)) as client:
            await asyncio.gather(*(worker(batch) for batch in batches))
    finally:
        stream.close()
    remaining = [name for name in names if name not in load_done()]
    print(f"done; {len(remaining)} names still untranslated (re-run to retry)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=80)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default="")
    parser.add_argument("--only-combo", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
