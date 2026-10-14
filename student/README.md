# Mã sinh viên — chỉ thực thi trên Kaggle

`data.py`: chuẩn hoá nhãn và collator có mask prompt/padding.
`model.py`: NF4, exact LoRA targets, trainable projector float32, checkpoint và reload.
`train.py`: AdamW, warmup/cosine, gradient accumulation, GradScaler và log thật.
`infer.py`: greedy generation, lưu output raw không chỉnh sửa.
`pipeline.py`: chạy preflight hoặc E0–E4, scorer gốc và export artifact.
`analyze.py`: báo cáo từ evidence, diagnostic thang sai và ảnh TP/FP/FN.

Ở local chỉ soạn nguồn và đóng gói:

```bash
python kaggle_tools/build_completed.py preflight
kaggle kernels push -p .
kaggle kernels status thngonquang/track4-day10-qwen3-vl-road-damage
```

Sau khi preflight thành công mới tạo mode `full`. Không chạy `student.pipeline`, pytest,
train, inference hoặc scorer trên local.

Metadata private, GPU/Internet và dataset được giữ tại root. Model revision được ghim
trong model.py. Full mode dùng batch 1/accum 8, 300 optimizer steps, seed 42,
generation cap 512 chung. LoRA rank/alpha/dropout = 16/32/0.05.

Gói review xuất tại `/kaggle/working/day10-review.zip`: JSON evidence, predictions raw,
loss/resource logs, split meta, figures và REPORT. Dùng CLI output với regex chính xác
để tải ZIP này, không cần tải lại toàn bộ ảnh/weights. Adapter/projector checkpoints
nằm dưới `day10-source/work/` trong output Kaggle, không đưa vào Git.

Các checkpoint intermediate có adapter/projector. Optimizer resume state ở
`/kaggle/temp/day10-resume/` chỉ tồn tại trong phiên hiện tại, không bảo đảm resume
exact sau khi phiên ngắt. Nếu cần resume qua phiên khác phải xuất state đó thêm trước.

Nguồn đề có một box crack chiều cao bằng 0 ở train. Mã sinh viên bỏ target suy biến
khỏi SFT, thống kê ở degenerate-labels.json; không sửa nguồn/split/GT của scorer.
Box dương bị làm tròn mất chiều được giữ tối thiểu 1 đơn vị trên thang 1000.

Giữ nguyên scripts/tests gốc. Test template REPORT chỉ áp dụng trước khi điền report;
regression cuối deselect đúng test đó và báo lý do. Git commits/publication cần
kiểm chứng riêng bằng repository thật, không tạo .git giả trên Kaggle.
