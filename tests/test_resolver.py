import pytest

from graphrag.names import DrugName, NameIndex, normalize
from graphrag.resolver import DrugResolver, split_query


@pytest.fixture
def resolver() -> DrugResolver:
    drugs = {
        "CID000001983": DrugName("CID000001983", "Acetaminophen", ["Paracetamol", "Tylenol"], ["Panadol", "Hapacol"]),
        "CID000003672": DrugName("CID000003672", "Ibuprofen", ["Brufen"], []),
        "CID000006691": DrugName("CID000006691", "Warfarin", [], ["Coumadin"]),
        "CID000004594": DrugName("CID000004594", "Omeprazole", [], ["Losec"]),
        "CID000003249": DrugName("CID000003249", "Ergocalciferol", ["Vitamin D2"], []),
        "CID000003961": DrugName("CID000003961", "Losartan", [], []),
    }
    names = NameIndex(drugs, {}, {}, [])
    return DrugResolver(names, set(drugs))


def test_normalize_strips_accents_strength_and_forms() -> None:
    assert normalize("Panadol 500mg viên nén") == "panadol"
    assert normalize("Hapacol 250") == "hapacol"
    assert normalize("Losartan kali 50 mg") == "losartan"
    assert normalize("Thuốc Đau") == "thuoc dau"


@pytest.mark.parametrize(
    ("query", "expected", "method"),
    [
        ("CID000006691", "CID000006691", "stitch_id"),
        ("ibuprofen", "CID000003672", "exact"),
        ("PARACETAMOL", "CID000001983", "exact"),
        ("Panadol Extra 500mg", "CID000001983", "brand_alias"),
        ("Hapacol 250", "CID000001983", "brand_alias"),
        ("ibuprofn", "CID000003672", "fuzzy"),
        ("wafarin", "CID000006691", "fuzzy"),
        ("losartan kali", "CID000003961", "exact"),
    ],
)
def test_resolves_names(resolver: DrugResolver, query: str, expected: str, method: str) -> None:
    result = resolver.resolve(query)
    assert result.stitch_id == expected
    assert result.method == method


@pytest.mark.parametrize("query", ["esomeprazole", "Vitamin C"])
def test_near_miss_is_not_auto_accepted(resolver: DrugResolver, query: str) -> None:
    result = resolver.resolve(query)
    assert result.stitch_id is None
    assert result.method in {"ambiguous", "not_found"}


def test_unknown_drug_is_not_found(resolver: DrugResolver) -> None:
    assert resolver.resolve("xyzabc").method == "not_found"


def test_split_query_handles_combinations() -> None:
    assert split_query("Paracetamol + Codein, ibuprofen và warfarin") == [
        "Paracetamol",
        "Codein",
        "ibuprofen",
        "warfarin",
    ]


def test_resolve_many(resolver: DrugResolver) -> None:
    results = resolver.resolve_many("Panadol, Brufen; Coumadin")
    assert [item.stitch_id for item in results] == [
        "CID000001983",
        "CID000003672",
        "CID000006691",
    ]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("tôi đang uống panadol với ibuprofen và warfarin", ["CID000001983", "CID000003672", "CID000006691"]),
        ("Mình dùng Hapacol 500mg cùng Losec thì có sao không?", ["CID000001983", "CID000004594"]),
        ("I'm taking Coumadin and Brufen", ["CID000006691", "CID000003672"]),
        ("thêm wafarin", ["CID000006691"]),
        ("CID000003961 và paracetamol", ["CID000003961", "CID000001983"]),
        ("xin chào", []),
    ],
)
def test_extract_drugs_from_chat_messages(resolver: DrugResolver, message: str, expected: list[str]) -> None:
    assert [item.stitch_id for item in resolver.extract(message)] == expected


def test_extract_keeps_ambiguous_names_for_confirmation(resolver: DrugResolver) -> None:
    results = resolver.extract("uống esomeprazole với ibuprofen")
    assert results[0].stitch_id is None and results[0].method in {"ambiguous", "not_found"}
    assert results[1].stitch_id == "CID000003672"


@pytest.fixture
def salt_resolver() -> DrugResolver:
    drugs = {
        "CID000005234": DrugName("CID000005234", "Sodium Chloride", ["Sodium monochloride"], []),
        "CID000004873": DrugName("CID000004873", "Potassium Chloride", ["Potassium monochloride"], []),
        "CID000002349": DrugName("CID000002349", "Benzylpenicillin", ["penicillin", "Penicillin G"], []),
        "CID000004730": DrugName("CID000004730", "Phenoxymethylpenicillin", ["penicillin", "Penicillin V"], []),
        "CID000003446": DrugName("CID000003446", "Mesna", ["CoM"], []),
        "CID000003249": DrugName("CID000003249", "Ergocalciferol", ["vitamin", "Vitamin D2"], []),
        "CID000001983": DrugName("CID000001983", "Acetaminophen", [], ["Panadol"]),
        "CID000002833": DrugName("CID000002833", "Colchicine", [], []),
    }
    return DrugResolver(NameIndex(drugs, {}, {}, []), set(drugs))


def test_salts_that_are_the_drug_keep_their_cation(salt_resolver: DrugResolver) -> None:
    assert normalize("Sodium Chloride") == "sodium chloride"
    assert normalize("Natri clorid 0.9%") == "natri clorid"
    assert salt_resolver.resolve("sodium chloride").stitch_id == "CID000005234"
    assert salt_resolver.resolve("Potassium Chloride").stitch_id == "CID000004873"
    found = [item.stitch_id for item in salt_resolver.extract("tôi uống potassium chloride với panadol")]
    assert found == ["CID000004873", "CID000001983"]


def test_shared_synonym_asks_instead_of_guessing(salt_resolver: DrugResolver) -> None:
    result = salt_resolver.resolve("penicillin")
    assert result.method == "ambiguous" and result.stitch_id is None
    assert {item.stitch_id for item in result.candidates} == {"CID000002349", "CID000004730"}
    assert salt_resolver.resolve("penicillin V").stitch_id == "CID000004730"


def test_chat_words_and_generic_synonyms_are_not_drugs(salt_resolver: DrugResolver) -> None:
    found = [item.stitch_id for item in salt_resolver.extract("tôi ăn cơm xong uống panadol")]
    assert found == ["CID000001983"]
    found = [item.stitch_id for item in salt_resolver.extract("uống vitamin tổng hợp với panadol")]
    assert "CID000003249" not in found
    found = [item.stitch_id for item in salt_resolver.extract("ông tôi bị gút uống colchicine")]
    assert found == ["CID000002833"]
