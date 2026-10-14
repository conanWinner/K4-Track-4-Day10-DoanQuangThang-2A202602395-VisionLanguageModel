# Day10 — trạng thái hoàn thiện

Ngày xác minh: 2026-10-14.

- Notebook: https://www.kaggle.com/code/thngonquang/track4-day10-qwen3-vl-road-damage
- Version 7: **COMPLETE**, full E0–E4, Tesla T4; toàn phiên khoảng 374 phút.
- E1/E2/E3 và E4-10% đều có 300 optimizer steps thành công.
- Mỗi evidence có 311 ảnh test, missing_predictions=0; dataset_version=d10-21e4070301.
- E1 parse hợp lệ 99,68%; F1 tăng 0,2310 so với E0, đạt ngưỡng rubric.
- E4 chọn E3 bằng validation F1; chỉ đổi lượng dữ liệu 142 so với 1422 ảnh.
- E5 không chạy, không yêu cầu bonus.

| Setup | F1 | VRAM GB | Phút train |
|---|---:|---:|---:|
| E0 | 0.0678 | — | — |
| E1 | 0.2989 | 5.1029 | 49.1485 |
| E2 | 0.3525 | 6.7613 | 50.0354 |
| E3 | 0.4596 | 6.8974 | 59.3199 |
| E4 10% | 0.2715 | 6.7840 | 58.3040 |
| E4 100% (tái dùng E3) | 0.4596 | 6.8974 | 59.3199 |

## Bằng chứng thực thi

Version 6 preflight: 73 test pass với dataset thật; save/reload adapter/projector thành công.
Version 7 regression cuối: 71 passed, 1 skipped, 1 deselected; checker 0 FAIL, 1 WARN
vì payload không chứa .git. Test template REPORT đòi placeholder nên không áp dụng
cho báo cáo hoàn chỉnh; giữ nguyên tests và deselect đúng một test đó.

Nguồn student/scripts/tests khớp SHA-256 trong report/source-manifest.json của run.
Log đầy đủ: report/logs/kaggle-v7.log. Báo cáo và hình tạo từ artifact thật.
Một nhãn crack train có h=0 chỉ bị bỏ khỏi target SFT; split/GT/scorer không đổi.

## Kiểm tra bản nộp trên GitHub

Repo public. Các commit được tạo theo từng nhóm triển khai khi hoàn thiện bàn giao;
không phải lịch sử thời gian huấn luyện. Kiểm tra GitHub checkout trên Kaggle
được chạy riêng, không huấn luyện hoặc chấm lại test để chọn kết quả.
Kaggle submission-check version 1 **COMPLETE**; kiểm tra commit
`cfc5658ef8c4ffb48347c906558edec6bb0ebc37` (6 commit):
**72 passed, 1 deselected; 0 FAIL, 0 WARN**. Không skip dataset test.
Manifest nguồn, đủ 300 log bước/run, sáu hình và mười câu báo cáo đều được kiểm tra.
Bằng chứng: [submission-check.json](report/submission-check.json),
[log](report/logs/submission-check-v1.log).

Lần cập nhật bàn giao sau kiểm tra chỉ bổ sung giải thích trong báo cáo và lưu log;
không thay đổi nguồn đã train hoặc evidence. Notebook xác minh:
https://www.kaggle.com/code/thngonquang/track4-day10-submission-check

Bài đủ các yêu cầu bắt buộc để nộp qua URL repo public. Chưa thực hiện thao tác
nộp trên cổng học tập; không coi việc push GitHub là biên nhận nộp bài.
