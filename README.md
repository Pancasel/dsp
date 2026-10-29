# Polypharmacy Graph-RAG

Ứng dụng chat (Streamlit) chạy local, dùng để xem tác dụng phụ khi phối hợp **≥ 3 thuốc** trên đồ thị Decagon (BioSNAP).

**Đầu vào:** chat bình thường, ví dụ *“tôi đang uống Panadol với ibuprofen và warfarin”*. Hệ thống tự nhận tên hoạt chất, biệt dược Việt Nam, tên sai chính tả nhẹ hoặc kèm hàm lượng. Tên nào chưa chắc chắn thì hệ thống hỏi lại bằng nút chọn. Gõ **bỏ <tên thuốc>** để bỏ một thuốc, **làm lại** để bắt đầu phác đồ mới.

Ngoài chat, chọn **Tạo phác đồ** để tìm thuốc theo tên hoạt chất, biệt dược hoặc tên đồng nghĩa, thêm từng thuốc vào danh sách và tra cứu khi đủ ít nhất 3 thuốc.

**Đầu ra:**
- Từng thuốc là gì: tên generic, biệt dược, protein đích, tác dụng phụ khi dùng riêng.
- **Tối đa 10 tác dụng phụ nghiêm trọng nhất** khi phối hợp (các tác dụng phụ khác gom trong một mục thu gọn), chia 3 tầng:
  - **Tier 1**: đã ghi nhận trong dữ liệu.
  - **Tier 2**: khuếch đại qua protein đích chung.
  - **Tier 3**: giả thuyết suy ra qua cầu protein.
- Mỗi kết quả là một thẻ gồm tên tác dụng phụ, cặp thuốc được báo cáo, mô tả ngắn bằng tiếng Việt (do LLM viết, mang tính chung) và mục "Bằng chứng" thu gọn.
- **Chi tiết thêm** (mục thu gọn trong câu trả lời): thông tin từng thuốc, cơ chế protein và **đồ thị quan hệ** tương tác được. Đồ thị gồm thuốc, số tác dụng phụ của từng cặp, protein đích chung, protein cầu nối và các tác dụng phụ nghiêm trọng.

Xem [PLAN.md](PLAN.md) để biết thiết kế, đánh giá dữ liệu và kết quả thí nghiệm. Các thuật ngữ và tên viết tắt (STITCH, PPI, Tier, hub, IME, AUROC…) được giải thích trong [docs/THUAT_NGU.md](docs/THUAT_NGU.md).

## Cài đặt

Cần Python 3.10 trở lên.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Đặt 4 thư mục dữ liệu Decagon **cạnh** thư mục dự án:

```
dsp/
├── bio-decagon-combo/bio-decagon-combo.csv
├── bio-decagon-mono/bio-decagon-mono.csv
├── bio-decagon-ppi/bio-decagon-ppi.csv
├── bio-decagon-targets/bio-decagon-targets.csv
└── polypharmacy-graph-rag/      <- dự án này
```

Nếu để dữ liệu ở chỗ khác, sửa các biến `*_FILE` trong `.env` (`DRUG_TARGETS_FILE`, `PROTEIN_INTERACTIONS_FILE`, `DRUG_PAIR_SIDE_EFFECTS_FILE`, `SINGLE_DRUG_SIDE_EFFECTS_FILE`).

Biến trong `.env`:
- `LLM_API_KEY`, `LLM_BASE_URL` (mặc định OpenRouter), `LLM_MODEL`: dùng cho phần giải thích. Khi không có key, app vẫn chạy và dùng giải thích mẫu, không gọi API nào.
- `MAX_SERIOUS_EFFECTS` (mặc định 10) và `MAX_OTHER_EFFECTS` (mặc định 5): số tác dụng phụ được hiển thị.
- Các biến còn lại (ngưỡng hub, số cầu protein, trọng số `RANKING_WEIGHT_*`) có chú thích trong `.env.example`.

> Nếu `.env` của bạn tạo trước khi đổi tên biến, hãy đổi tên theo `.env.example`: `TARGETS_FILE` → `DRUG_TARGETS_FILE`, `PPI_FILE` → `PROTEIN_INTERACTIONS_FILE`, `COMBO_FILE` → `DRUG_PAIR_SIDE_EFFECTS_FILE`, `MONO_FILE` → `SINGLE_DRUG_SIDE_EFFECTS_FILE`, `CACHE_DB` → `CACHE_DATABASE_FILE`, `MAX_EVIDENCE` → `MAX_EVIDENCE_ITEMS`, `TIER3_ONE_HOP_PER_PAIR` → `TIER3_ONE_HOP_BRIDGES_PER_PAIR`, `TIER3_TWO_HOP_PER_PAIR` → `TIER3_TWO_HOP_BRIDGES_PER_PAIR`, `RANK_W_*` → `RANKING_WEIGHT_*`, `LLM_TIMEOUT` → `LLM_TIMEOUT_SECONDS`.

## Chạy

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

Dùng `python -m streamlit` để chắc chắn Streamlit chạy bằng Python của venv dự án. Nếu gặp `ModuleNotFoundError` (ví dụ `pyvis`), thường là terminal đang kích hoạt venv của dự án khác: chạy `deactivate`, kích hoạt lại `.venv` của dự án này, rồi `python -m pip install -r requirements.txt`. Khi thiếu riêng `pyvis`, app vẫn chạy, chỉ khung đồ thị hiện hướng dẫn cài.

Lần chạy đầu tiên sẽ tạo cache `.cache/decagon.sqlite3`, mất khoảng 2 phút. Mỗi lần phác đồ thay đổi (và có ≥ 3 thuốc), hệ thống phân tích lại bằng đúng 1 request LLM, chứa toàn bộ evidence dưới dạng JSON.

Có thể gọi pipeline trực tiếp từ Python:

```python
from graphrag.pipeline.polypharmacy_system import PolypharmacySystem
system = PolypharmacySystem.load()
found_drugs = system.resolver.extract("tôi đang uống panadol với ibuprofen và warfarin")
result = system.predict([resolution.stitch_id for resolution in found_drugs])
```

## Kiểm thử và thí nghiệm

```powershell
pytest                                              # dữ liệu tổng hợp + mock, không gọi API
python -m evaluation.run_offline_experiments        # E1, E4, E5, E6 -> evaluation/results/
python -m evaluation.run_accuracy_experiments       # E2, A1-A6 + hình + evaluation/results/ACCURACY.md (~1 phút)
python -m evaluation.run_full_coverage_check        # C1-C3: toàn bộ 645 thuốc -> evaluation/results/COVERAGE.md (~2 phút)
python -m evaluation.run_noise_drug_scaling         # A7
python -m evaluation.run_multi_drug_regimens        # A8
python -m evaluation.run_llm_groundedness_experiment --yes-call-api --regimens 10   # E3, CÓ gọi LLM
```

### Đánh giá độ chính xác (cho báo cáo)

Kết quả đầy đủ, bảng và hình nằm trong [evaluation/results/ACCURACY.md](evaluation/results/ACCURACY.md) (hình PNG trong `evaluation/results/figures/`). Chạy `run_offline_experiments` trước rồi `run_accuracy_experiments` để file tổng hợp có đủ E1/E4. Tóm tắt (seed 0):

| Mã | Đánh giá | Kết quả chính |
|---|---|---|
| C1–C3 | Toàn bộ 645 thuốc ([COVERAGE.md](evaluation/results/COVERAGE.md)) | Cache khớp CSV 100% (63,473 cặp, 4,649,441 TDP); 645/645 thuốc có tên và nhận đúng (gõ riêng và trong câu chat); 506 phác đồ phủ mọi thuốc chạy không lỗi, 100% TDP và cạnh bằng chứng hiển thị đúng với CSV |
| E1 | Nhận dạng tên thuốc (71 câu nhập) | Đúng 98.6%, nhận sai 0 |
| A1 | Đối chiếu kết quả với CSV gốc (200 phác đồ) | Tier 1 precision = recall = 100% (52,880 TDP); Tier 2 đúng 72/72; 40,529 đường dẫn bằng chứng đúng 100% |
| A2 | Cầu protein dự đoán cặp có tương tác | AUROC 0.54, gần ngẫu nhiên; baseline độ phổ biến 0.94. Tier 3 chỉ là giả thuyết |
| A3 | Dự đoán từng tác dụng phụ (giao thức Decagon, 963 TDP) | AUROC 0.951 / AUPRC 0.940 / AP@50 0.966 (baseline 0.908 / 0.889 / 0.883; Decagon công bố 0.872 / 0.832 / 0.803) |
| E4 | Nhãn Tier 3 trên 2000 cặp bị ẩn | Precision@10 0.507 (phổ biến 0.418, ngẫu nhiên 0.065) |
| A4 | Validator với dự đoán LLM bị làm sai | Loại 100% (950/950), giữ 100% dự đoán hợp lệ (300/300) |
| E2 | Trích thuốc từ 30 câu chat thực tế | Đúng 30/30 câu, 0 thuốc nhận nhầm, 0 bỏ sót |
| A5 | 26 tương tác kinh điển (warfarin + NSAID, "triple whammy"…) | Tác dụng phụ mong đợi lọt top 10 ở 53.8% số ca, top 3 ở 23.1% (giới hạn của dữ liệu chỉ có/không) |
| A6 | Từ khóa "nghiêm trọng" so với danh sách IME của EMA | Precision 0.92 nhưng recall 0.15 → chuyển sang nhãn IME |
| A7 | Tương tác kinh điển khi thêm thuốc "nhiễu" (`python -m evaluation.run_noise_drug_scaling`) | Hit@10 53.8% (3 thuốc) → 51.5% / 49.2% / 37.7% / 39.2% (4 / 5 / 6 / 8 thuốc); ngẫu nhiên chỉ 10.9% → 4.7%. Hit@3 giảm 23.1% → 9.2% |
| A8 | 20 phác đồ 4–8 thuốc viết tay, mỗi phác đồ có 1 cặp tương tác kinh điển (`evaluation/test_cases/clinical_benchmark_multi_drug.csv`, `python -m evaluation.run_multi_drug_regimens`; chưa được bác sĩ rà) | 18/20 ca chấm được (2 ca không có dữ liệu cho cặp); Hit@10 27.8% (5/18), Hit@3 11.1%; ngẫu nhiên 6.4%. 4–5 thuốc: 33.3%; 6–8 thuốc: 22.2% |

## Cấu trúc thư mục

Mỗi file làm một việc; tên file nói lên việc đó. Thuật ngữ khó hiểu (Tier, hub, IME…) xem [docs/THUAT_NGU.md](docs/THUAT_NGU.md).

```
polypharmacy-graph-rag/
├── app.py                                  # điểm khởi chạy Streamlit, chỉ gọi web_interface/main_page.py
│
├── graphrag/                               # LÕI HỆ THỐNG
│   ├── settings/                           # cấu hình
│   │   ├── application_settings.py         # class Settings: mọi cấu hình, đọc từ .env
│   │   ├── environment_variable_readers.py # đọc và kiểm tra từng biến môi trường (số, đường dẫn, %)
│   │   └── ranking_weights.py              # trọng số của điểm xếp hạng tác dụng phụ
│   ├── source_data/                        # dữ liệu nguồn
│   │   ├── csv_file_reader.py              # đọc CSV Decagon (cả file .zip) và CSV trong resources/
│   │   ├── side_effect_cache_database.py   # tạo / kiểm tra cache SQLite từ CSV tác dụng phụ
│   │   └── data_records.py                 # SideEffect, Evidence: bản ghi dùng chung
│   ├── knowledge_graph/                    # đồ thị và truy xuất bằng chứng
│   │   ├── decagon_graph.py                # DecagonGraph: nạp đồ thị, tra cứu thuốc–protein–tác dụng phụ
│   │   ├── hub_protein_filter.py           # nhận diện và loại protein hub
│   │   ├── shared_target_proteins.py       # protein mà ≥ 3 thuốc cùng tác động (tín hiệu Tier 2)
│   │   ├── protein_bridge_search.py        # tìm cầu protein 1 bước / 2 bước (Tier 3)
│   │   ├── evidence_candidate_collection.py# dựng mọi bằng chứng ứng viên Tier 1/2/3 cho từng cặp
│   │   ├── evidence_budget_selection.py    # chọn bằng chứng vừa giới hạn, đánh số E1, E2…
│   │   ├── regimen_evidence_retrieval.py   # retrieve_regimen_evidence(): ghép các bước trên cho 1 phác đồ
│   │   └── evidence_path_validation.py     # kiểm tra từng cạnh của đường bằng chứng có thật
│   ├── side_effect_ranking/                # xếp hạng
│   │   ├── regimen_side_effect_scoring.py  # tính điểm và sắp xếp tác dụng phụ của cả phác đồ
│   │   └── effects_to_display.py           # chọn mục hiển thị: top nghiêm trọng + vài mục khác
│   ├── drug_names/                         # tên thuốc và nhận dạng
│   │   ├── text_normalization.py           # bỏ dấu, hàm lượng, dạng bào chế, gốc muối
│   │   ├── name_index.py                   # NameIndex: tên thuốc, gene, tác dụng phụ tiếng Việt
│   │   ├── side_effect_seriousness.py      # quyết định tác dụng phụ nghiêm trọng / đe dọa tính mạng
│   │   ├── name_resolution_results.py      # kết quả nhận dạng một chuỗi (DrugNameResolution)
│   │   ├── fuzzy_name_matching.py          # khớp gần đúng tên gõ sai (RapidFuzz)
│   │   ├── drug_name_resolver.py           # DrugResolver.resolve(): một chuỗi → STITCH ID
│   │   └── chat_message_drug_extraction.py # tìm mọi thuốc trong một câu chat
│   ├── llm/                                # mô hình ngôn ngữ lớn
│   │   ├── system_prompt.py                # lời dặn gửi LLM
│   │   ├── prediction_data_models.py       # cấu trúc JSON LLM phải trả về, các lỗi
│   │   ├── prediction_request.py           # gửi 1 request tới nhà cung cấp LLM
│   │   ├── model_response_parsing.py       # đọc JSON trả về, kể cả khi bị cắt ngang
│   │   └── prediction_validation.py        # loại dự đoán không khớp bằng chứng
│   ├── result_presentation/                # đổi mã thành tên đọc được, dựng kết quả
│   │   ├── evidence_path_text.py           # đường bằng chứng → một dòng chữ
│   │   ├── llm_context.py                  # JSON gọn gửi cho LLM
│   │   ├── template_explanation.py         # giải thích mẫu khi không có LLM
│   │   ├── drug_information_cards.py       # thẻ thông tin từng thuốc
│   │   ├── interaction_cards.py            # thẻ từng tác dụng phụ (giải thích + bằng chứng)
│   │   ├── mechanism_summary.py            # cầu protein, số tác dụng phụ từng cặp, protein chung
│   │   ├── coverage_summary.py             # bao nhiêu cặp có dữ liệu
│   │   └── vietnamese_labels.py            # nhãn tier và lời miễn trừ tiếng Việt
│   ├── visualization/
│   │   └── interaction_graph_html.py       # đồ thị quan hệ tương tác được (pyvis)
│   └── pipeline/
│       ├── polypharmacy_system.py          # PolypharmacySystem: nạp mọi thứ, predict() từ tên thuốc ra kết quả
│       └── regimen_error.py                # lỗi khi phác đồ không phân tích được
│
├── web_interface/                          # GIAO DIỆN STREAMLIT
│   ├── main_page.py                        # khung trang, thanh bên, chuyển trang
│   ├── chat_page.py                        # trang "Trò chuyện"
│   ├── regimen_builder_page.py             # trang "Tạo phác đồ"
│   ├── chat_message_handling.py            # xử lý tin nhắn, thêm/bỏ thuốc, chọn gợi ý
│   ├── answer_rendering.py                 # vẽ câu trả lời: thẻ tác dụng phụ, các tab chi tiết
│   ├── session_state.py                    # dữ liệu trang nhớ giữa các lần chạy lại
│   ├── cached_resources.py                 # hệ thống và HTML đồ thị được cache
│   ├── interface_texts.py                  # câu chữ cố định, lệnh, màu tier
│   └── page_styles.py                      # CSS
│
├── evaluation/                             # THÍ NGHIỆM CHO BÁO CÁO
│   ├── run_offline_experiments.py          # chạy E1, E4, E5, E6
│   ├── run_accuracy_experiments.py         # chạy E2, A1–A6 và viết ACCURACY.md
│   ├── run_full_coverage_check.py          # chạy C1–C3 và viết COVERAGE.md
│   ├── run_noise_drug_scaling.py           # chạy A7
│   ├── run_multi_drug_regimens.py          # chạy A8
│   ├── run_llm_groundedness_experiment.py  # chạy E3 (CÓ gọi LLM)
│   ├── experiments/                        # mỗi thí nghiệm một file, tên bắt đầu bằng mã (e1_, a3_, c2_…)
│   ├── shared/                             # nạp hệ thống, metric xếp hạng, ghi CSV, vẽ hình, bảng Markdown
│   ├── reports/                            # viết ACCURACY.md và COVERAGE.md
│   ├── test_cases/                         # bộ ca thử (tên thuốc, câu chat, benchmark lâm sàng)
│   └── results/                            # kết quả: CSV, hình, ACCURACY.md, COVERAGE.md
│
├── resource_builders/                      # SINH FILE TRONG resources/ (chạy 1 lần)
│   ├── build_drug_names.py                 # tên thuốc từ PubChem + SIDER
│   ├── build_gene_names.py                 # ký hiệu gene từ NCBI
│   ├── build_serious_effects.py            # nhãn nghiêm trọng theo danh sách IME của EMA
│   ├── check_brand_aliases.py              # kiểm tra file biệt dược Việt Nam
│   └── translate_side_effect_names.py      # dịch tên tác dụng phụ sang tiếng Việt (gọi LLM)
│
├── resources/                              # CSV tên, biệt dược, gene, bản dịch, từ khóa (xem bảng dưới)
├── tests/                                  # pytest, chia thư mục giống graphrag/
├── docs/THUAT_NGU.md                       # giải thích thuật ngữ và tên viết tắt
└── report_source/                          # mã nguồn báo cáo (LaTeX, script dựng PDF/DOCX)
```

## Tài nguyên (`resources/`)

| File | Nguồn |
|---|---|
| `drug_names.csv` | `python -m resource_builders.build_drug_names` (PubChem + SIDER, cần internet) |
| `drug_name_overrides.csv` | Sửa tay, có ghi bằng chứng |
| `brand_aliases_vietnamese.csv` | Viết tay, kiểm tra bằng `python -m resource_builders.check_brand_aliases --fill` |
| `gene_names.csv` | `python -m resource_builders.build_gene_names` (NCBI) |
| `side_effect_translations_vietnamese.csv` | Dịch thủ công; `python -m resource_builders.translate_side_effect_names` là tùy chọn và **có gọi LLM** |
| `serious_effects.csv` | `python -m resource_builders.build_serious_effects`: nhãn nghiêm trọng theo danh sách Important Medical Event của EMA (cần internet) |
| `effect_meddra_map.csv` | Viết tay: tên thông dụng của Decagon → thuật ngữ MedDRA ("heart attack" → Myocardial infarction) |
| `serious_keywords.csv` | Viết tay: dự phòng cho tên chưa ánh xạ được, nhóm đe dọa tính mạng (`critical`) và các từ loại trừ (`exclude`) |

> Thông tin chỉ mang tính tham khảo ở mức quần thể, không thay thế tư vấn của bác sĩ hoặc dược sĩ.
