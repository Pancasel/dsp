"""Fixed texts, commands and colours of the interface."""

TIER_COLORS = {1: "#2563eb", 2: "#ea580c", 3: "#7c3aed"}
TIER_EXPLANATIONS = {
    1: "Đã ghi nhận: tác dụng phụ đã được báo cáo cho cặp thuốc này.",
    2: "Khuếch đại: đã ghi nhận, và cặp thuốc có protein đích chung với thuốc thứ ba.",
    3: "Giả thuyết: cặp thuốc chưa có dữ liệu, suy luận qua liên kết protein. Cần kiểm chứng.",
}

# Messages that clear the regimen, compared after normalize_for_matching() (no accents).
RESET_COMMANDS = {"lam lai", "reset", "xoa het", "phac do moi", "bat dau lai", "clear"}
# Matched on the accented text: without accents "bỏ" (remove) and "bố"
# (father, as in "bố tôi uống ...") would both become "bo".
REMOVE_COMMAND_PREFIXES = ("bỏ ", "bớt ", "xóa ", "xoá ", "xoa ", "remove ", "delete ")

WELCOME_MESSAGE = (
    "Chào bạn! Bạn đang dùng những thuốc nào? Cứ nhập tên hoạt chất hoặc biệt dược như "
    "*“Panadol, ibuprofen với warfarin”* nhé. Khi có ít nhất 3 thuốc khác nhau, mình "
    "có thể tra các tác dụng phụ được ghi nhận trong dữ liệu.\n\n"
    "Muốn sửa danh sách, gõ **bỏ <tên thuốc>**; muốn bắt đầu lại, gõ **làm lại**."
)
EXAMPLE_REGIMENS = (
    "Panadol, Ibuprofen, Warfarin",
    "Nurofen, Plavix, Lipitor",
    "Zoloft, Xanax, Seroquel",
)

MINIMUM_DRUGS_TO_ANALYSE = 3
# Serious effect cards shown before "Xem thêm".
EFFECT_CARDS_SHOWN_FIRST = 5
# Ambiguous name: at most this many candidate buttons.
CANDIDATE_BUTTONS_SHOWN = 3

CHAT_PAGE_LABEL = "💬 Trò chuyện"
REGIMEN_BUILDER_PAGE_LABEL = "💊 Tạo phác đồ"
