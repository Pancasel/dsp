# Độ phủ trên toàn bộ thuốc trong dữ liệu

Sinh tự động bởi `python eval/run_coverage.py` (seed 0, không gọi LLM). Mọi kiểm tra đối chiếu với file CSV gốc, đọc độc lập với cache SQLite.

C3 chạy 506 phác đồ phủ mọi thuốc; mỗi thuốc xuất hiện ít nhất 3 lần; trung bình 0.032 s/phác đồ.

| Kiểm tra | Đạt | Tổng | Tỷ lệ |
|---|---|---|---|
| C1 – cặp thuốc: tập TDP trong cache = CSV | 63473 | 63473 | 100.0% |
| C1 – (cặp, TDP) trong cache = CSV | 4649441 | 4649441 | 100.0% |
| C1 – thuốc trong dữ liệu được hệ thống nạp | 645 | 645 | 100.0% |
| C1 – TDP đơn lẻ (mono) mỗi thuốc = CSV | 645 | 645 | 100.0% |
| C1 – protein đích mỗi thuốc = CSV | 645 | 645 | 100.0% |
| C2 – thuốc có tên đọc được | 645 | 645 | 100.0% |
| C2 – tên thuốc nhận đúng thuốc (gõ riêng) | 645 | 645 | 100.0% |
| C2 – tên thuốc nhận đúng thuốc (trong câu chat) | 645 | 645 | 100.0% |
| C2 – biệt dược VN nhận đúng thuốc | 123 | 123 | 100.0% |
| C3 – phác đồ chạy hết pipeline không lỗi | 506 | 506 | 100.0% |
| C3 – thuốc được phân tích ít nhất 1 lần | 645 | 645 | 100.0% |
| C3 – thẻ thuốc có tên đọc được | 1937 | 1937 | 100.0% |
| C3 – TDP Tier 1/2 hiển thị: mọi cặp có trong CSV | 4903 | 4903 | 100.0% |
| C3 – TDP Tier 1/2 hiển thị: liệt kê đủ mọi cặp trong CSV | 4903 | 4903 | 100.0% |
| C3 – cầu protein (Tier 3) chỉ cho cặp chưa có dữ liệu | 719 | 719 | 100.0% |
| C3 – cạnh bằng chứng hiển thị (cả cầu Tier 3) đúng với file gốc | 9890 | 9890 | 100.0% |

Số lỗi chi tiết: 0.
