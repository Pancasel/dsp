# Polypharmacy Graph-RAG

Ứng dụng chat (Streamlit) chạy local, dùng để xem tác dụng phụ khi phối hợp **≥ 3 thuốc** trên đồ thị Decagon (BioSNAP).

**Đầu vào:** chat bình thường, ví dụ *“tôi đang uống Panadol với ibuprofen và warfarin”*. Hệ thống tự nhận tên hoạt chất, biệt dược Việt Nam, tên sai chính tả nhẹ hoặc kèm hàm lượng. Tên nào chưa chắc chắn thì hệ thống hỏi lại bằng nút chọn. Gõ **bỏ <tên thuốc>** để bỏ một thuốc, **làm lại** để bắt đầu phác đồ mới.

**Đầu ra:**
- Từng thuốc là gì: tên generic, biệt dược, protein đích, tác dụng phụ khi dùng riêng.
- **Tối đa 10 tác dụng phụ nghiêm trọng nhất** khi phối hợp (các tác dụng phụ khác gom trong một mục thu gọn), chia 3 tầng:
  - **Tier 1**: đã ghi nhận trong dữ liệu.
  - **Tier 2**: khuếch đại qua protein đích chung.
  - **Tier 3**: giả thuyết suy ra qua cầu protein.
- Mỗi kết quả kèm đường dẫn bằng chứng và giải thích tiếng Việt.
- **Đồ thị quan hệ** tương tác được, đặt cạnh khung chat (bật/tắt ở thanh bên). Đồ thị gồm thuốc, số tác dụng phụ của từng cặp, protein đích chung, protein cầu nối và các tác dụng phụ nghiêm trọng.

Xem [PLAN.md](PLAN.md) để biết thiết kế, đánh giá dữ liệu và kết quả thí nghiệm.

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

Nếu để dữ liệu ở chỗ khác, sửa các biến `*_FILE` trong `.env`.

Biến trong `.env`:
- `LLM_API_KEY`, `LLM_BASE_URL` (mặc định OpenRouter), `LLM_MODEL`: dùng cho phần giải thích. Khi không có key, app vẫn chạy và dùng giải thích mẫu, không gọi API nào.
- `MAX_SERIOUS_EFFECTS` (mặc định 10) và `MAX_OTHER_EFFECTS` (mặc định 5): số tác dụng phụ được hiển thị.

## Chạy

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

Dùng `python -m streamlit` để chắc chắn Streamlit chạy bằng Python của venv dự án. Nếu gặp `ModuleNotFoundError` (ví dụ `pyvis`), thường là terminal đang kích hoạt venv của dự án khác: chạy `deactivate`, kích hoạt lại `.venv` của dự án này, rồi `python -m pip install -r requirements.txt`. Khi thiếu riêng `pyvis`, app vẫn chạy, chỉ khung đồ thị hiện hướng dẫn cài.

Lần chạy đầu tiên sẽ tạo cache `.cache/decagon.sqlite3`, mất khoảng 2 phút. Mỗi lần phác đồ thay đổi (và có ≥ 3 thuốc), hệ thống phân tích lại bằng đúng 1 request LLM, chứa toàn bộ evidence dưới dạng JSON.

Có thể gọi pipeline trực tiếp từ Python:

```python
from graphrag.pipeline import PolypharmacySystem
system = PolypharmacySystem.load()
drugs = system.resolver.extract("tôi đang uống panadol với ibuprofen và warfarin")
result = system.predict([item.stitch_id for item in drugs])
```

## Kiểm thử và thí nghiệm

```powershell
pytest                                   # dữ liệu tổng hợp + mock, không gọi API
python eval/run_offline.py               # E1, E4, E5, E6 -> eval/results/
python eval/run_accuracy.py              # A1-A4 + hình + eval/results/ACCURACY.md (~1 phút)
python eval/e3_llm_groundedness.py --yes-call-api --regimens 10   # E3, CÓ gọi LLM
```

### Đánh giá độ chính xác (cho báo cáo)

Kết quả đầy đủ, bảng và hình nằm trong [eval/results/ACCURACY.md](eval/results/ACCURACY.md) (hình PNG trong `eval/results/figures/`). Chạy `run_offline.py` trước rồi `run_accuracy.py` để file tổng hợp có đủ E1/E4. Tóm tắt (seed 0):

| Mã | Đánh giá | Kết quả chính |
|---|---|---|
| E1 | Nhận dạng tên thuốc (70 câu nhập) | Đúng 98.6%, nhận sai 0 |
| A1 | Đối chiếu kết quả với CSV gốc (200 phác đồ) | Tier 1 precision = recall = 100% (52,880 TDP); Tier 2 đúng 72/72; 40,529 đường dẫn bằng chứng đúng 100% |
| A2 | Cầu protein dự đoán cặp có tương tác | AUROC 0.54, gần ngẫu nhiên; baseline độ phổ biến 0.94. Tier 3 chỉ là giả thuyết |
| A3 | Dự đoán từng tác dụng phụ (giao thức Decagon, 963 TDP) | AUROC 0.951 / AUPRC 0.940 / AP@50 0.966 (baseline 0.908 / 0.889 / 0.883; Decagon công bố 0.872 / 0.832 / 0.803) |
| E4 | Nhãn Tier 3 trên 2000 cặp bị ẩn | Precision@10 0.507 (phổ biến 0.418, ngẫu nhiên 0.065) |
| A4 | Validator với dự đoán LLM bị làm sai | Loại 100% (950/950), giữ 100% dự đoán hợp lệ (300/300) |

## Tài nguyên (`resources/`)

| File | Nguồn |
|---|---|
| `drug_names.csv` | `scripts/build_drug_names.py` (PubChem + SIDER, cần internet) |
| `drug_name_overrides.csv` | Sửa tay, có ghi bằng chứng |
| `brand_aliases_vi.csv` | Viết tay, kiểm tra bằng `scripts/check_brand_aliases.py --fill` |
| `gene_names.csv` | `scripts/build_gene_names.py` (NCBI) |
| `side_effects_vi.csv` | Dịch thủ công; `scripts/translate_side_effects.py` là tùy chọn và **có gọi LLM** |
| `serious_keywords.csv` | Viết tay, so khớp nguyên từ |

> Thông tin chỉ mang tính tham khảo ở mức quần thể, không thay thế tư vấn của bác sĩ hoặc dược sĩ.
