"""React to what the user does: a chat message, a candidate button, a drug removed or added."""

import unicodedata

import streamlit as st

from graphrag.drug_names.name_resolution_results import DrugNameResolution
from graphrag.drug_names.text_normalization import normalize_for_matching
from graphrag.pipeline.polypharmacy_system import PolypharmacySystem
from graphrag.pipeline.regimen_error import RegimenError
from web_interface.interface_texts import (
    MINIMUM_DRUGS_TO_ANALYSE,
    REMOVE_COMMAND_PREFIXES,
    RESET_COMMANDS,
)
from web_interface.session_state import add_assistant_message


def analyse_regimen_if_ready(system: PolypharmacySystem) -> None:
    """Analyse the regimen when it has enough drugs; otherwise say how many are missing."""
    regimen: dict = st.session_state["regimen"]
    drug_names = ", ".join(regimen.values())
    if len(regimen) < MINIMUM_DRUGS_TO_ANALYSE:
        st.session_state["result"] = None  # an older result no longer matches
        missing_count = MINIMUM_DRUGS_TO_ANALYSE - len(regimen)
        if drug_names:
            current = f"Mình đã ghi nhận {drug_names}"
        else:
            current = "Mình chưa ghi nhận thuốc nào"
        add_assistant_message(
            f"{current}. Thêm ít nhất {missing_count} thuốc nữa thì mình mới tra được phác đồ từ 3 thuốc."
        )
        return
    try:
        result = system.predict(list(regimen))
    except RegimenError as error:
        add_assistant_message(f"Chưa phân tích được: {error}")
        return
    st.session_state["result"] = result
    serious_count = sum(card["serious"] for card in result["interactions"])
    if serious_count:
        response = (
            f"Mình đã tra xong **{drug_names}**. Trong dữ liệu có {serious_count} mục được "
            "xếp nhóm nghiêm trọng để bạn chú ý."
        )
    else:
        response = (
            f"Mình đã tra xong **{drug_names}**. Hiện chưa thấy mục nào được xếp nhóm "
            "nghiêm trọng trong dữ liệu; điều đó không đảm bảo là không có rủi ro."
        )
    add_assistant_message(response, result)


def add_recognized_drugs(found: list[DrugNameResolution]) -> list[str]:
    """Add resolved drugs that are not in the regimen yet; return their descriptions for the reply."""
    added_descriptions = []
    for resolution in found:
        if resolution.stitch_id and resolution.stitch_id not in st.session_state["regimen"]:
            st.session_state["regimen"][resolution.stitch_id] = resolution.name
            typed_differently = normalize_for_matching(resolution.query) != normalize_for_matching(
                resolution.name
            )
            if typed_differently:
                note = f" (từ “{resolution.query}”)"
            else:
                note = ""
            added_descriptions.append(f"**{resolution.name}**{note}")
    return added_descriptions


def handle_user_message(system: PolypharmacySystem, text: str) -> None:
    st.session_state["messages"].append({"role": "user", "content": text})
    if normalize_for_matching(text) in RESET_COMMANDS:
        st.session_state.update(regimen={}, pending=[], result=None)
        add_assistant_message("Mình đã xóa danh sách thuốc rồi. Bạn muốn tra những thuốc nào?")
        return
    lowered_text = unicodedata.normalize("NFC", text).strip().casefold()
    if lowered_text.startswith(REMOVE_COMMAND_PREFIXES):
        _remove_mentioned_drugs(system, lowered_text.split(" ", 1)[1])
        return

    found = system.resolver.extract(text)
    if not found:
        add_assistant_message(
            "Mình chưa nhận ra tên thuốc trong tin nhắn này. Bạn thử nhập tên hoạt chất "
            "hoặc biệt dược, chẳng hạn “Panadol, ibuprofen, warfarin” nhé."
        )
        return
    added_descriptions = add_recognized_drugs(found)
    needs_choice = [
        resolution.to_dict()
        for resolution in found
        if resolution.stitch_id is None and resolution.candidates and resolution.method == "ambiguous"
    ]
    not_found_texts = [
        resolution.query
        for resolution in found
        if resolution.stitch_id is None
        and not (resolution.method == "ambiguous" and resolution.candidates)
    ]
    st.session_state["pending"].extend(needs_choice)

    reply_sentences = []
    if added_descriptions:
        reply_sentences.append("Mình nhận ra " + ", ".join(added_descriptions) + ".")
    if needs_choice:
        reply_sentences.append(
            "Mình chưa chắc “"
            + "”, “".join(pending["query"] for pending in needs_choice)
            + "” là thuốc nào; bạn chọn giúp mình ở bên dưới nhé."
        )
    if not_found_texts:
        reply_sentences.append(
            "Mình chưa tìm thấy "
            + ", ".join(f"“{text}”" for text in not_found_texts)
            + " trong dữ liệu thuốc."
        )
    add_assistant_message(" ".join(reply_sentences) or "Mấy thuốc này đã có trong danh sách rồi.")
    if added_descriptions:
        analyse_regimen_if_ready(system)


def _remove_mentioned_drugs(system: PolypharmacySystem, text_after_command: str) -> None:
    regimen = st.session_state["regimen"]
    removed_names = []
    for resolution in system.resolver.extract(text_after_command):
        if resolution.stitch_id in regimen:
            removed_names.append(regimen.pop(resolution.stitch_id))
    if not removed_names:
        add_assistant_message("Mình chưa thấy thuốc đó trong danh sách hiện tại.")
        return
    add_assistant_message(f"Đã bỏ {', '.join(removed_names)} khỏi danh sách nhé.")
    analyse_regimen_if_ready(system)


def apply_candidate_choice(
    system: PolypharmacySystem, pending_index: int, chosen_candidate: dict | None
) -> None:
    """chosen_candidate is None when the user answers "Không phải" (none of them)."""
    pending = st.session_state["pending"].pop(pending_index)
    if chosen_candidate is None:
        add_assistant_message(f"Được rồi, mình sẽ bỏ qua “{pending['query']}”.")
        return
    st.session_state["regimen"][chosen_candidate["stitch_id"]] = chosen_candidate["name"]
    add_assistant_message(f"Mình hiểu “{pending['query']}” là **{chosen_candidate['name']}** nhé.")
    analyse_regimen_if_ready(system)
