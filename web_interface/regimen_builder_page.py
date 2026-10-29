"""The "Tạo phác đồ" page: search the drug catalogue, add drugs one by one, remove them."""

import html
from dataclasses import dataclass

import streamlit as st

from graphrag.drug_names.text_normalization import normalize_for_matching
from graphrag.pipeline.polypharmacy_system import PolypharmacySystem
from web_interface.chat_message_handling import analyse_regimen_if_ready
from web_interface.interface_texts import MINIMUM_DRUGS_TO_ANALYSE
from web_interface.session_state import reset_drug_catalog_picker


MINIMUM_SEARCH_LENGTH = 2
MAXIMUM_SEARCH_RESULTS = 30
BRAND_NAMES_IN_LABEL = 2
DRUG_CHIP_COLUMNS = 4


@dataclass
class DrugCatalog:
    # Drug IDs sorted by drug name.
    drug_ids: list[str]
    # Drug ID -> "Name · Brand1, Brand2", with the ID added when two labels are equal.
    labels: dict[str, str]
    # Drug ID -> every normalized name, brand and synonym it can be found by.
    search_terms: dict[str, list[str]]


def build_drug_catalog(system: PolypharmacySystem) -> DrugCatalog:
    def drug_name_lowercase(drug_id: str) -> str:
        return system.names.drug_name(drug_id).casefold()

    drug_ids = sorted(system.graph.drug_ids, key=drug_name_lowercase)
    base_labels: dict[str, str] = {}
    search_terms: dict[str, list[str]] = {}
    for drug_id in drug_ids:
        drug = system.names.drugs_by_id.get(drug_id)
        name = system.names.drug_name(drug_id)
        brand_names = list(dict.fromkeys(drug.vietnamese_brand_aliases if drug else []))
        brand_names = [
            brand for brand in brand_names
            if normalize_for_matching(brand) != normalize_for_matching(name)
        ][:BRAND_NAMES_IN_LABEL]
        if brand_names:
            base_labels[drug_id] = f"{name} · {', '.join(brand_names)}"
        else:
            base_labels[drug_id] = name

        all_terms = [name]
        if drug:
            all_terms += drug.vietnamese_brand_aliases + drug.synonyms
        normalized_terms = [normalize_for_matching(term) for term in all_terms]
        search_terms[drug_id] = list(dict.fromkeys(term for term in normalized_terms if term))

    label_counts: dict[str, int] = {}
    for label in base_labels.values():
        label_counts[label] = label_counts.get(label, 0) + 1
    labels = {}
    for drug_id, label in base_labels.items():
        if label_counts[label] > 1:
            labels[drug_id] = f"{label} ({drug_id})"
        else:
            labels[drug_id] = label
    return DrugCatalog(drug_ids, labels, search_terms)


def search_drug_catalog(catalog: DrugCatalog, system: PolypharmacySystem, query: str) -> list[str]:
    """Drug IDs whose terms contain the query: exact term first, then prefix, then anywhere."""
    if len(query) < MINIMUM_SEARCH_LENGTH:
        return []
    EXACT_MATCH, PREFIX_MATCH, CONTAINS_MATCH = 0, 1, 2
    sort_keys = []
    for drug_id in catalog.drug_ids:
        matching_terms = [term for term in catalog.search_terms[drug_id] if query in term]
        if not matching_terms:
            continue
        best_match_kind = CONTAINS_MATCH
        for term in matching_terms:
            if term == query:
                best_match_kind = min(best_match_kind, EXACT_MATCH)
            elif term.startswith(query):
                best_match_kind = min(best_match_kind, PREFIX_MATCH)
        sort_keys.append((best_match_kind, system.names.drug_name(drug_id).casefold(), drug_id))
    sort_keys.sort()
    return [drug_id for *_, drug_id in sort_keys[:MAXIMUM_SEARCH_RESULTS]]


def render_regimen_builder_page(system: PolypharmacySystem) -> None:
    with st.container(key="regimen-view"):
        st.subheader("Tạo phác đồ")
        st.caption("Tìm thuốc theo tên hoạt chất hoặc biệt dược, rồi thêm vào danh sách.")
        _render_search_and_add(system)
        _render_current_regimen(system)


def _render_search_and_add(system: PolypharmacySystem) -> None:
    catalog = build_drug_catalog(system)
    search_column, select_column, add_column = st.columns([3, 3, 1], vertical_alignment="bottom")
    search_text = search_column.text_input(
        "Tìm thuốc",
        placeholder="Tên hoạt chất hoặc biệt dược…",
        key="drug-search-text",
    )
    query = normalize_for_matching(search_text)
    matching_drug_ids = search_drug_catalog(catalog, system, query)

    selected_drug = None
    if matching_drug_ids:
        selected_drug = select_column.selectbox(
            "Chọn thuốc phù hợp",
            options=matching_drug_ids,
            index=None,
            placeholder=f"{len(matching_drug_ids)} kết quả — chọn một thuốc",
            format_func=lambda drug_id: catalog.labels[drug_id],
            # A new key empties the picker after a drug is added.
            key=f"drug-catalog-{st.session_state.get('drug_catalog_version', 0)}-{query}",
        )
    elif query:
        select_column.caption("Chưa thấy kết quả khớp. Thử tên hoạt chất hoặc biệt dược khác nhé.")
    else:
        select_column.caption("Nhập ít nhất 2 ký tự để tìm thuốc.")

    regimen = st.session_state["regimen"]
    already_added = selected_drug in regimen if selected_drug else False
    clicked_add = add_column.button(
        "Đã thêm" if already_added else "＋ Thêm",
        type="primary",
        disabled=selected_drug is None or already_added,
        width="stretch",
    )
    if not clicked_add:
        return
    regimen[selected_drug] = system.names.drug_name(selected_drug)
    reset_drug_catalog_picker()
    if len(regimen) >= MINIMUM_DRUGS_TO_ANALYSE:
        with st.spinner("Đang tra dữ liệu cho phác đồ…"):
            analyse_regimen_if_ready(system)
    else:
        missing_count = MINIMUM_DRUGS_TO_ANALYSE - len(regimen)
        st.toast(
            f"Đã thêm {system.names.drug_name(selected_drug)}. "
            f"Thêm {missing_count} thuốc nữa để tra phác đồ."
        )
    st.rerun()


def _render_current_regimen(system: PolypharmacySystem) -> None:
    regimen = st.session_state["regimen"]
    st.markdown(f"**Thuốc trong phác đồ** · {len(regimen)}")
    if regimen:
        chip_columns = st.columns(min(len(regimen), DRUG_CHIP_COLUMNS))
        for position, (drug_id, name) in enumerate(list(regimen.items())):
            with chip_columns[position % len(chip_columns)]:
                st.markdown(f"<span class='drug-chip'>💊 {html.escape(name)}</span>", unsafe_allow_html=True)
                if st.button("Bỏ thuốc", key=f"builder-remove-{drug_id}"):
                    regimen.pop(drug_id)
                    analyse_regimen_if_ready(system)
                    st.rerun()
    else:
        st.info("Danh sách đang trống. Tìm rồi thêm thuốc để bắt đầu.")

    missing_count = max(0, MINIMUM_DRUGS_TO_ANALYSE - len(regimen))
    if missing_count:
        st.caption(f"Thêm {missing_count} thuốc nữa thì mình có thể tra dữ liệu cho phác đồ.")
    elif st.session_state.get("result"):
        st.success("Đã có kết quả cho danh sách thuốc hiện tại. Chuyển sang **Trò chuyện** để xem chi tiết.")
    else:
        st.caption("Danh sách đã đủ điều kiện để tra.")
