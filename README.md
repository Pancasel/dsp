# Polypharmacy Graph-RAG

Ứng dụng Streamlit chạy local, dùng để xem tác dụng phụ khi phối hợp **≥ 3 thuốc** trên đồ thị Decagon (BioSNAP).

**Đầu vào:** gõ tên thuốc (hoạt chất, biệt dược Việt Nam, sai chính tả nhẹ, kèm hàm lượng đều được) hoặc chụp ảnh hộp thuốc.

**Đầu ra:**
- Từng thuốc là gì: tên generic, biệt dược, protein đích, tác dụng phụ khi dùng riêng.
- Danh sách tác dụng phụ khi phối hợp, đã xếp hạng và chia 3 tầng:
  - **Tier 1**: đã ghi nhận trong dữ liệu.
  - **Tier 2**: khuếch đại qua protein đích chung.
  - **Tier 3**: giả thuyết suy ra qua cầu protein.
- Mỗi kết quả kèm đường dẫn bằng chứng và giải thích tiếng Việt.

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

Biến LLM trong `.env`:
- `LLM_API_KEY`, `LLM_BASE_URL` (mặc định OpenRouter), `LLM_MODEL`: dùng cho phần giải thích.
- `VISION_MODEL`: model đọc ảnh. Nếu ảnh dùng nhà cung cấp khác thì đặt thêm `VISION_BASE_URL` và `VISION_API_KEY`.

Khi không có key, app vẫn chạy và dùng giải thích mẫu, không gọi API nào.

## Chạy

```powershell
streamlit run app.py
```

Lần chạy đầu tiên sẽ tạo cache `.cache/decagon.sqlite3`, mất khoảng 2 phút. Mỗi lần bấm **Phân tích** chỉ gửi đúng 1 request LLM, chứa toàn bộ evidence dưới dạng JSON.

Có thể gọi pipeline trực tiếp từ Python:

```python
from graphrag.pipeline import PolypharmacySystem
system = PolypharmacySystem.load()
result = system.predict(["Panadol", "ibuprofen", "warfarin"], top_k=15)
```

## Kiểm thử và thí nghiệm

```powershell
pytest                                   # dữ liệu tổng hợp + mock, không gọi API
python eval/run_offline.py               # E1, E4, E5, E6 -> eval/results/
python eval/e3_llm_groundedness.py --yes-call-api --regimens 10   # E3, CÓ gọi LLM
```

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
