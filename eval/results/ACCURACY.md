# Kết quả đánh giá độ chính xác

Sinh tự động bởi `python eval/run_offline.py` và `python eval/run_accuracy.py` (seed 0). Không có bước nào gọi LLM. Hình nằm trong `figures/`.

## E1. Nhận dạng tên thuốc

70 câu nhập (generic, biệt dược VN, kèm hàm lượng, sai chính tả, thuốc không có trong dữ liệu). Độ chính xác **98.6%**, nhận sai **0**, hỏi lại 1.

| Loại | n | Đúng | Hỏi lại | Sai |
|---|---|---|---|---|
| Tên hoạt chất | 20 | 100.0% | 0 | 0 |
| Biệt dược VN | 20 | 100.0% | 0 | 0 |
| Kèm hàm lượng | 10 | 100.0% | 0 | 0 |
| Sai chính tả | 10 | 90.0% | 1 | 0 |
| Hoa/thường | 3 | 100.0% | 0 | 0 |
| Không có trong dữ liệu | 7 | 100.0% | 0 | 0 |

![E1](figures/e1_resolver.png)

## E2. Trích thuốc từ câu chat

30 câu chat thực tế (biệt dược, hàm lượng, sai chính tả, tiếng Anh, có từ thừa như "vitamin C", "rượu bia", triệu chứng). Đây là đường đi thật của giao diện chat.

| Đúng cả câu | Precision thuốc | Recall thuốc | Thuốc nhận nhầm | Thuốc bỏ sót | Câu hỏi lại |
|---|---|---|---|---|---|
| 100.0% | 100.0% | 100.0% | 0 | 0 | 2 |

## A1. Độ trung thực của kết quả so với dữ liệu gốc

200 phác đồ ngẫu nhiên (3–6 thuốc). Mọi kết quả được đối chiếu lại với các file CSV gốc, đọc độc lập với cache SQLite.

| Kiểm tra | Đạt | Tổng | Tỷ lệ |
|---|---|---|---|
| Tier 1 – precision: tác dụng phụ trả về có trong combo CSV | 52880 | 52880 | 100.0% |
| Tier 1 – recall: tác dụng phụ trong CSV của các cặp được trả về | 52880 | 52880 | 100.0% |
| Tier 2 – precision: cặp gắn Tier 2 thỏa quy tắc protein chung | 72 | 72 | 100.0% |
| Tier 2 – recall: cặp thỏa quy tắc được gắn Tier 2 | 72 | 72 | 100.0% |
| Tier 1 – đường dẫn bằng chứng đúng từng cạnh | 36145 | 36145 | 100.0% |
| Tier 2 – đường dẫn bằng chứng đúng từng cạnh | 1203 | 1203 | 100.0% |
| Tier 3 – đường dẫn bằng chứng đúng từng cạnh | 3181 | 3181 | 100.0% |

## A2. Dự đoán cặp thuốc có tương tác từ đồ thị protein

Tất cả 40186 cặp giữa các thuốc có protein đích; nhãn dương = cặp có dữ liệu trong `bio-decagon-combo` (14247 cặp). Các đặc trưng chỉ dùng targets + PPI, không dùng dữ liệu combo. Lưu ý: cặp "chưa có dữ liệu" không chắc là không tương tác.

| Đặc trưng | AUROC | AUPRC | Tỷ lệ cặp có | Tỷ lệ tương tác khi có | Tỷ lệ tương tác khi không | Lift |
|---|---|---|---|---|---|---|
| Protein đích chung (Tier 2) | 0.532 | 0.397 | 0.2764 | 0.4024 | 0.3362 | 1.135 |
| Cầu 1 bước (đích A ↔ đích B) | 0.543 | 0.3985 | 0.5912 | 0.3809 | 0.3163 | 1.075 |
| Cầu 2 bước (qua protein trung gian) | 0.5367 | 0.3983 | 0.9566 | 0.3567 | 0.3077 | 1.006 |
| Tổng hợp cầu protein (Tier 3) | 0.5434 | 0.4022 | 0.9567 | 0.3567 | 0.3069 | 1.006 |
| Baseline: độ phổ biến thuốc | 0.9428 | 0.9193 |  |  |  |  |
| Ngẫu nhiên | 0.4952 | 0.3505 |  |  |  |  |

![A2](figures/a2_roc.png)

**Diễn giải.** Cầu protein có ở 95.7% số cặp nên gần như không phân biệt được (AUROC 0.5434, lift 1.006), trong khi chỉ biết thuốc nào được báo cáo nhiều đã đạt AUROC 0.9428. Vì vậy Tier 3 chỉ được trình bày là *giả thuyết cơ chế* cần kiểm chứng, không phải dự báo nguy cơ; hệ thống không dùng sự tồn tại của cầu protein để kết luận một cặp thuốc có tương tác.

## A3. Dự đoán từng loại tác dụng phụ (giao thức Decagon)

Mỗi tác dụng phụ có ≥ 500 cặp: ẩn 10% số cạnh làm tập kiểm tra, mỗi cạnh dương ghép với một cặp âm ngẫu nhiên không có tác dụng phụ đó. Điểm của phương pháp chỉ tính trên phần huấn luyện. Dòng Decagon lấy từ bài báo (GCN, cùng dữ liệu nhưng cách chia khác) — chỉ để tham chiếu.

| Phương pháp | Số TDP | AUROC | AUPRC | AP@50 |
|---|---|---|---|---|
| Đồ thị: bộ chấm nhãn Tier 3 (số lần) | 963 | 0.9509 | 0.94 | 0.9662 |
| Đồ thị: tần suất chuẩn hoá | 963 | 0.8115 | 0.7395 | 0.476 |
| Baseline: độ phổ biến thuốc | 963 | 0.9081 | 0.8894 | 0.8832 |
| Ngẫu nhiên | 963 | 0.5002 | 0.5072 | 0.2708 |
| Decagon (bài báo, GCN) | 964 | 0.872 | 0.832 | 0.803 |

![A3](figures/a3_side_effects.png)

![A3 hist](figures/a3_auroc_hist.png)

**Diễn giải.** Bộ chấm nhãn Tier 3 (đếm số thuốc bạn của A và B có cùng tác dụng phụ) đạt AUROC 0.9509, cao hơn baseline độ phổ biến (0.9081) và con số Decagon công bố (0.872). Không nên kết luận là tốt hơn Decagon: cặp âm lấy ngẫu nhiên phần lớn là cặp ít được báo cáo, nên chỉ riêng độ phổ biến đã đạt 0.9081; giao thức chia dữ liệu cũng khác bài báo. Kết quả cho thấy nhãn mà Tier 3 gợi ý có cơ sở thống kê trên dữ liệu, đồng thời phản ánh thiên lệch báo cáo (reporting bias) của FAERS.

## E4. Nhãn Tier 3 trên cặp thuốc bị ẩn

2000 cặp có dữ liệu (cả hai thuốc có protein đích) được ẩn đi; danh sách ứng viên được dựng lại không dùng chính cặp đó.

| Phương pháp | Hit@10 | Precision@10 | Recall@20 |
|---|---|---|---|
| Đồ thị (ứng viên Tier 3) | 0.9655 | 0.5072 | 0.1553 |
| Baseline: phổ biến toàn cục | 0.9455 | 0.4178 | 0.1302 |
| Ngẫu nhiên | 0.4205 | 0.0649 | 0.0148 |

![E4](figures/e4_tier3_labels.png)

## A4. Độ bền của bộ kiểm tra (validator) đầu ra LLM

Dự đoán hợp lệ và dự đoán bị làm sai có chủ đích (ID bằng chứng bịa, sai tier, sai nhãn, nhãn Tier 3 ngoài danh sách ứng viên) được sinh từ evidence thật.

| Trường hợp | Số dự đoán | Mong đợi | Giữ | Loại | Đúng |
|---|---|---|---|---|---|
| Hợp lệ | 300 | giữ | 300 | 0 | 100.0% |
| ID bằng chứng bịa | 277 | loại | 0 | 277 | 100.0% |
| Nhãn bịa (ngoài từ vựng) | 73 | loại | 0 | 73 | 100.0% |
| Nhãn Tier 3 ngoài ứng viên | 73 | loại | 0 | 73 | 100.0% |
| Sai nhãn tác dụng phụ | 227 | loại | 0 | 227 | 100.0% |
| Sai tier | 300 | loại | 0 | 300 | 100.0% |

## A5. Benchmark tương tác thuốc kinh điển

26 ca tương tác đã được xác lập lâm sàng (`eval/clinical_benchmark.csv`), mỗi ca gồm cặp tương tác và một thuốc thứ ba. Đo vị trí tác dụng phụ mong đợi trong danh sách nghiêm trọng (hệ thống hiển thị 10 mục đầu).

| Cách xếp hạng | Ca có dữ liệu | Hit@3 | Hit@10 | MRR |
|---|---|---|---|---|
| Chỉ từ khóa | 26 | 30.8% | 53.8% | 0.241 |
| IME | 26 | 15.4% | 42.3% | 0.1566 |
| IME + đe dọa tính mạng (mặc định) | 26 | 23.1% | 53.8% | 0.2219 |
| IME + đe dọa tính mạng, bỏ IDF | 26 | 15.4% | 50.0% | 0.1769 |

![A5](figures/a5_clinical_benchmark.png)

| Ca | Phác đồ | Mong đợi | Hạng (mặc định) |
|---|---|---|---|
| 1 | warfarin + ibuprofen + acetaminophen | chảy máu | 2 |
| 2 | warfarin + aspirin + omeprazole | chảy máu | 9 |
| 3 | clopidogrel + aspirin + atorvastatin | chảy máu | 13 |
| 4 | warfarin + ciprofloxacin + metoprolol | chảy máu | 11 |
| 5 | warfarin + fluconazole + amlodipine | chảy máu | 2 |
| 6 | diclofenac + citalopram + amlodipine | chảy máu | 1 |
| 7 | ibuprofen + lisinopril + furosemide | suy thận | 6 |
| 8 | diclofenac + losartan + hydrochlorothiazide | suy thận | 33 |
| 9 | spironolactone + lisinopril + metformin | tăng kali máu | 29 |
| 10 | spironolactone + losartan + amlodipine | tăng kali máu | 4 |
| 11 | verapamil + metoprolol + atorvastatin | nhịp chậm / block tim | 6 |
| 12 | diltiazem + propranolol + omeprazole | nhịp chậm / block tim | 5 |
| 13 | digoxin + amiodarone + furosemide | loạn nhịp | 1 |
| 14 | simvastatin + clarithromycin + omeprazole | tiêu cơ vân | 30 |
| 15 | simvastatin + amiodarone + metformin | tiêu cơ vân | 4 |
| 16 | atorvastatin + erythromycin + acetaminophen | tiêu cơ vân | 6 |
| 17 | glibenclamide + fluconazole + amlodipine | hạ đường huyết | 26 |
| 18 | glipizide + ciprofloxacin + metoprolol | hạ đường huyết | 31 |
| 19 | methotrexate + trimethoprim + omeprazole | ức chế tủy | 28 |
| 20 | methotrexate + ketorolac + prednisone | ức chế tủy | 17 |
| 21 | oxycodone + diazepam + acetaminophen | ức chế thần kinh / hô hấp | 2 |
| 22 | morphine + alprazolam + omeprazole | ức chế thần kinh / hô hấp | 7 |
| 23 | tramadol + sertraline + acetaminophen | co giật / serotonin | 104 |
| 24 | tramadol + fluoxetine + omeprazole | co giật / serotonin | 25 |
| 25 | sildenafil + isosorbide + atorvastatin | tụt huyết áp | không nằm trong nhóm nghiêm trọng |
| 26 | haloperidol + citalopram + omeprazole | loạn nhịp | 3 |

**Diễn giải.** Decagon chỉ cho biết một tác dụng phụ *có* được báo cáo cho cặp thuốc hay không, không có tần suất hay mức độ, nên điểm xếp hạng không phân biệt được tác dụng phụ đặc trưng của tương tác với hàng trăm tác dụng phụ khác của cùng cặp. Khoảng một nửa số ca kinh điển lên được top 10. Thử thêm chuẩn hoá theo độ đặc trưng của thuốc (kiểu PRR) cũng không cải thiện ổn định, nên trọng số không được chỉnh theo bộ 26 ca này để tránh overfit. Hướng cải thiện: dùng số lượng báo cáo FAERS để tính disproportionality thật.

## A6. Phân loại tác dụng phụ nghiêm trọng

Chuẩn tham chiếu: danh sách *Important Medical Event* (IME) của EMA, MedDRA 29.1 (`scripts/build_serious_effects.py`). 244 / 1317 tác dụng phụ phối hợp được gắn nghiêm trọng; ung thư, dị tật bẩm sinh, thai sản, nhiễm trùng cộng đồng và bệnh mạn tính bị loại vì trong FAERS chúng thường là chỉ định hay bệnh nền. Bảng dưới so cách dùng từ khóa với IME trên các tên khớp trực tiếp một thuật ngữ IME.

| Phương pháp | Số tên | IME nghiêm trọng | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Từ khóa cũ (khớp nguyên từ) | 2998 | 1699 | 0.8894 | 0.1042 | 0.1865 |
| Từ khóa mới (số nhiều + loại trừ) | 2998 | 1699 | 0.9164 | 0.1548 | 0.2649 |

**Diễn giải.** Từ khóa chính xác (precision cao) nhưng chỉ bắt được một phần nhỏ (recall thấp) các tác dụng phụ IME, và bỏ sót tên thông dụng như "heart attack", "gastrointestinal bleed", "deep vein thromboses". Vì vậy hệ thống dùng nhãn IME làm chính; từ khóa chỉ dùng cho tên chưa ánh xạ được và để ưu tiên nhóm đe dọa tính mạng khi xếp hạng.

## E3. Dự đoán của LLM qua được validator

Chưa chạy (cần gọi API): `python eval/e3_llm_groundedness.py --yes-call-api --regimens 10`.
