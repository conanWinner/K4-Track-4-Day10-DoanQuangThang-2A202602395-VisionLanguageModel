# Chiến lược hoàn thiện Lab Day10 trên Kaggle

Bản chiến lược ban đầu (2026-10-13); các trạng thái chưa chạy bên dưới là lịch sử lập kế hoạch.
Kết quả thực tế và trạng thái mới nhất: [KAGGLE_STATUS.md](KAGGLE_STATUS.md).

Ngày rà soát: 2026-10-13. Phạm vi hiện tại: đọc đề và lập chiến lược; chưa thực thi notebook, train hoặc chấm điểm.

## 1. Hiện trạng và mục tiêu

Đã đọc README, GUIDE, RUBRIC, REPORT, notebook, toàn bộ configs, scripts và tests. Repository hiện sạch, có một commit `c2f3997`. Notebook chỉ là khung Colab, các phần xử lý nhãn, train và inference còn `YOUR CODE HERE`; REPORT còn placeholder; chưa có evidence E0–E4 hoặc metadata notebook Kaggle.

Kaggle CLI 2.2.4 đã hoạt động và thao tác `kernels list --mine` xác minh owner `thngonquang`. Chưa thấy notebook Day10 trong trang 20 kết quả đã đọc; cần tìm đúng slug trước khi tạo mới. Chưa kiểm chứng GPU/quota, dataset mount, package compatibility hoặc inference trên Kaggle.

Mục tiêu bắt buộc: E0 zero-shot; E1 LoRA LLM; E2 E1 + projector đầy đủ; E3 E2 + LoRA ViT; E4 đổi một biến với đủ hai giá trị; báo cáo 10 câu, >=6 hình đúng/sai, evidence do scorer gốc sinh ra, và kiểm tra bài nộp không còn FAIL. E5 GRPO chỉ làm sau phần bắt buộc nếu tài nguyên cho phép.

## 2. Ranh giới local và Kaggle

- Local: đọc/sửa mã, notebook và metadata; tải về và đọc artifact/log; viết báo cáo từ kết quả thật. Không chạy Python của lab, pytest, xử lý dữ liệu, train, inference hoặc scorer tại local.
- Kaggle: cài dependency, chạy test, chuẩn bị split, kiểm chứng nhãn, train, inference, scorer, tạo hình và kiểm tra bài nộp.
- Giữ nguyên `scripts/` và `tests/` của giảng viên. Mã riêng đặt ngoài hai thư mục này, ví dụ `student/`. Giữ nguyên template/config gốc khi có thể; cấu hình run riêng ghi vào artifact.
- Không cần tạo dataset riêng: gắn dataset công khai có sẵn bằng `dataset_sources`. Không đưa credential local vào notebook.
- Notebook mới mặc định private, GPU và Internet bật. Giữ nguyên owner/slug/visibility nếu tìm được notebook Day10 hiện có.
- Không tự commit/push Git hoặc nộp bài trong bước lập chiến lược. Yêu cầu >=5 commit sẽ được thực hiện theo các mốc có ý nghĩa khi có phạm vi giao Git phù hợp; không tạo commit rỗng để đủ số.

## 3. Thiết kế mã và vận chuyển lên Kaggle

Mã local đề xuất: `student/data.py` (nhãn/chat/collator), `student/model.py` (load/setup/save/reload), `student/train.py`, `student/infer.py`, `student/analyze.py`; notebook là điểm vào từng giai đoạn. Có kiểm tra riêng cho mã sinh viên ngoài `tests/` gốc.

Kaggle CLI push notebook/script không tự chuyển toàn bộ repository bên cạnh code lên môi trường chạy. Vì vậy cần một bước đóng gói có thể kiểm tra: notebook chứa payload nguồn gồm scripts/tests/configs/docs và mã sinh viên, kèm manifest SHA-256; cell đầu giải mã vào một thư mục mới dưới `/kaggle/working`. Chỉ đóng gói các file trong allowlist, không có `.git`, `.env`, credential, dữ liệu hay weights. Nguồn local vẫn là bản chuẩn; payload được tạo lại từ đó trước mỗi push. Không lệ thuộc bản GitHub chưa được push.

Giữ `kernel-metadata.json` cạnh `lab_day10.ipynb` trong dự án. Trước mỗi push kiểm tra `code_file`, nguồn dữ liệu, mode, owner, slug và manifest. Không dùng lại thư mục output local có sẵn để tránh ghi đè.

## 4. Các giai đoạn và điều kiện chuyển bước

| Giai đoạn | Thực hiện trên Kaggle | Điều kiện hoàn thành |
|---|---|---|
| P0: môi trường + dữ liệu | GPU/runtime/package audit; mount dữ liệu; test gốc; prepare split | GPU khả dụng, dependency import được, test dữ liệu thật không skip, split không trùng session |
| P1: kiểm chứng pipeline | YOLO -> JSON -> pixel; mask/collator; module audit; forward/backward ngắn; save/reload | Nhãn đúng, prompt bị mask, loss/gradient hữu hạn, target và trạng thái trainable đúng, checkpoint nạp lại đúng |
| P2: E0 | Gỡ lỗi trên val; sau khi khoá pipeline chạy đủ test | Một output raw cho mỗi ảnh test, scorer gốc sinh e0.json |
| P3: E1 | 300 optimizer steps; val và checkpoint; reload rồi inference test | Adapter thật, log train, đủ predictions, e1.json + meta tài nguyên |
| P4: E2 và E3 | Mỗi setup khởi tạo độc lập từ base; train và reload đúng | e2/e3.json, projector riêng có checksum, cấu hình so sánh nhất quán |
| P5: E4 | Chọn setup bằng val; chạy 10% so với 100% train | Hai evidence E4, chỉ train_fraction khác nhau, kết luận có CI |
| P6: báo cáo + kiểm tra | Bảng số liệu, >=6 ảnh, kiểm tra evidence và submission | Số báo cáo khớp artifact, không missing predictions/placeholder, kiểm tra rubric đầy đủ |
| P7: E5 tuỳ chọn | Reward tests, GRPO 60 bước, phân tích | Chỉ làm sau P6; báo rõ có hoặc chưa có bonus |

Ưu tiên GPU T4 nếu có để sát điều kiện đề. Nếu dùng P100/GPU khác, phải ghi rõ khác biệt phần cứng so với yêu cầu Colab T4 và dùng cùng GPU cho các setup so sánh. Hai T4 không tự cộng thành một GPU 32 GB; ban đầu dùng một GPU với `device_map={"": 0}`. Không giả định đủ VRAM trước khi đo.

P0 dùng torch/torchvision có sẵn nếu tương thích; kiểm chứng các bản ghim trong requirements bằng cài/import thật. Không coi ghi chú Colab là bằng chứng chạy được trên Kaggle. Nếu phải thay dependency, dùng requirements Kaggle riêng, ghi lý do và version thực tế; không âm thầm sửa requirements của đề.

## 5. Dữ liệu và ngăn rò rỉ

Gắn `lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes`, tìm thư mục thật chứa `images/` và `labels-YOLO/` trong `/kaggle/input`, gọi `scripts/prepare_data.py --source ...` vào thư mục run mới. Không cần download lại bằng credential hoặc dùng `--unzip`.

Đọc số liệu thực tế từ `meta.json`: đề dự kiến 2.009 ảnh, train/val/test 1.422/276/311. Test của đề ghi tham chiếu dataset_version `d10-21e4070301`; phải đo lại chứ không hardcode. Nếu dữ liệu hiện tại khác, dừng đối chiếu phiên bản trước khi train. Lưu checksum nguồn/split và thống kê lớp/session.

Val dùng để gỡ lỗi, chọn setup E4 và kiểm tra checkpoint. Không dùng F1 test để chỉnh prompt, lr, checkpoint hoặc chọn setup. Sau khi khoá mỗi setup, inference/chấm test một lần chính thức. Các run kỹ thuật thất bại được ghi rõ, không chọn kết quả test tốt nhất qua nhiều lần thử.

E4 đề xuất `data_size`: chọn subset train 10% một lần bằng seed 42, lưu danh sách image_id và phân bố lớp; không động đến val/test. Giữ setup, initialization, seed, lr, rank, batch hiệu dụng, 300 bước và inference giống run 100%. Bản 100% có thể tái sử dụng run gốc đã khoá; tạo evidence mang tên E4 bằng scorer gốc từ predictions đó và ghi nguồn rõ ràng. Đây là so sánh dưới cùng ngân sách optimizer steps; 10% dữ liệu sẽ được lặp nhiều lần hơn, phải nêu trong kết luận.

## 6. Các kiểm chứng kỹ thuật bắt buộc

1. JSON: chỉ ba lớp, `bbox_2d`, số nguyên thang 0–1000, thứ tự ổn định theo y1/x1; kiểm tra chiều box và round-trip pixel bằng hình.
2. Processor: ảnh RGB; đo `image_grid_thw` và số token ảnh thực tế, đối chiếu 220 ở 640x360. Không tự resize khác để giảm VRAM mà vẫn gọi đó là thí nghiệm gốc.
3. Loss: kiểm tra prompt ids là prefix full ids; labels prompt/padding = -100, chỉ JSON và token kết thúc tham gia loss. Collator pad chuỗi nhưng cat pixel/grid; giữ `mm_token_type_ids` nếu processor trả về.
4. Quantization: NF4 + double quant; visual và lm_head không bị lượng tử hoá. In loại module thực tế để xác nhận, không chỉ tin cấu hình skip.
5. LoRA: đối chiếu cây module thật; dự kiến 252 target LLM và E3 thêm 96 target ViT = 348. Kiểm tra rõ các tham số ngoài phạm vi không trainable; E2/E3 chỉ mở merger/deepstack merger và LoRA tương ứng.
6. Precision: projector E2/E3 ở float32, autocast fp16 và GradScaler; kiểm tra dtype của tham số trainable thực tế và gradient. LoRA trainable cần precision phù hợp với GradScaler, không mặc định mọi adapter đều fp16.
7. Train: AdamW, lr 2e-4, warmup 5%, cosine, gradient clipping 1.0, checkpointing non-reentrant, cache tắt. 300 bước là optimizer steps, không phải micro-batches.
8. Checkpoint: adapter và projector đã train lưu riêng; thêm optimizer/scheduler/scaler/RNG/data order nếu muốn resume chính xác. Checkpoint mỗi 100 bước, không xóa bản cũ. Base và mỗi setup bắt đầu độc lập; không train nối E1 -> E2 -> E3.
9. Reload: mở base mới, nạp adapter + projector, kiểm tra tensor/checksum và inference val. Lưu cả processor/config/revision; không đưa weights vào Git.
10. Inference: greedy `do_sample=False`, padding left, cắt output bằng chiều dài input đã padding; lưu raw nguyên bản và image_id. Không sửa raw thành JSON hợp lệ hoặc dùng ground truth thay dự đoán.

Nếu OOM, thử batch 1/accum 8 để giữ batch hiệu dụng 8. Sau smoke test nên khoá chung thiết lập khả dụng cho E1–E4. Mọi thay đổi bắt buộc theo từng setup phải được ghi rõ như yếu tố ảnh hưởng so sánh. Nếu token 256 bị cắt trên val, quyết định giới hạn chung trước test và ghi khác biệt so với config gốc.

## 7. Quản lý run và tài nguyên

Tách run môi trường/smoke, E0, E1, E2, E3 và E4 để giảm thiệt hại khi lỗi. Có thể dùng nhiều private kernels riêng, mỗi kernel có nguồn/config cụ thể; chạy train tuần tự để quản lý GPU quota. Trước khi tạo kiểm tra slug không trùng; reuse owner và đích hiện có nếu phù hợp.

Kaggle session mới không bảo đảm còn `/kaggle/working` của run trước. Resume hoặc E5 cần attach output kernel đã COMPLETE qua `kernel_sources`, rồi copy checkpoint từ input read-only sang thư mục run mới. Không suy ra checkpoint local tự xuất hiện trên Kaggle.

Mỗi run lưu GPU model, driver/CUDA, torch/dependency versions, model revision, nguồn/config hash, số bước, số ảnh, trainable_params theo khối, peak allocated VRAM và thời gian train đo thật. E0 cũng ghi tài nguyên inference nhưng không cần meta train dương.

Không lấy 20–30 phút/run trong đề làm cam kết. Đo thời gian smoke gồm forward/backward và inference nhỏ; ước lượng 300 optimizer steps và 311 ảnh từ tốc độ thật. Có ít nhất 4 run SFT đầy đủ (E1/E2/E3/E4-10%) và 5 lượt inference test nếu tái sử dụng run 100% cho E4. Chỉ kết luận khả thi sau kiểm tra quota/runtime.

Quy trình CLI: `kernels push` khởi chạy notebook; ghi URL/version; `kernels status` đến COMPLETE hoặc ERROR; `kernels output` tải vào thư mục local mới theo run/version. Nếu push timeout, kiểm tra đích trước retry. RUNNING/upload thành công chưa phải hoàn thành lab.

## 8. Evidence và báo cáo

Artifact mỗi run: `predictions.jsonl`, evidence JSON do `scripts/score.py` sinh, `run.json`, loss/log, config đã resolve, manifest, adapter, projector nếu có. Artifact tổng hợp: split meta, bảng số liệu, >=6 hình có GT/pred và chú thích, ví dụ raw đúng/sai, kết quả kiểm tra Kaggle.

Evidence chính: `report/evidence/e0.json` đến `e3.json`, `e4_data_size_0.1.json`, `e4_data_size_1.0.json`. Dùng scorer nguyên bản, score_version=1, test đủ ảnh, missing_predictions=0, dataset_version nhất quán. Kiểm tra thêm E4 vì check_submission gốc chỉ kiểm tra E4 khá lỏng.

Chỉ báo đạt điểm khi số thật đáp ứng: E1 parse_valid_rate >=0.95 và F1_E1 - F1_E0 >=0.10. Nếu chưa đạt, báo đúng kết quả; không bịa số hoặc tiếp tục tối ưu dựa trên test. Bảng REPORT phải khớp F1, CI95, FP/ảnh, per-class F1, VRAM và thời gian.

Câu 3 có thể kiểm chứng bằng chấm lại cùng raw predictions với thang sai như một phân tích sau khi kết quả chính đã khoá, lưu tên evidence chẩn đoán riêng ngoài e0/e1/e2/e3/e4. Không train một model khác chỉ để minh hoạ lỗi thang.

CI scorer bootstrap theo ảnh, trong khi frame trong cùng session có tương quan. Báo rõ giới hạn này; CI hai setup chồng lấn không tự chứng minh không khác biệt, và CI riêng lẻ không phải kiểm định chênh lệch. Có thể thêm paired/session bootstrap dưới mã sinh viên như phân tích bổ sung, giữ nguyên scorer và số chính của đề.

### Xung đột bộ test với báo cáo cuối

`tests/test_configs_and_docs.py::test_report_template_has_student_markers_and_ten_questions` yêu cầu REPORT chứa `<điền`, còn `scripts/check_submission.py` yêu cầu không còn `<điền`. Hai điều kiện không thể cùng đạt trên cùng báo cáo hoàn chỉnh.

Chạy đủ test gốc trên Kaggle khi REPORT còn template ở P0. Cuối bài giữ nguyên tests, chạy submission checker và các test còn áp dụng; deselect rõ một test template này khi chạy regression cuối, ghi lý do và kết quả đầy đủ. Không để lại `<điền` hoặc sửa test chỉ để xanh. Checker về số commit/forbidden files cần bản sao Git để đọc lịch sử thật trên Kaggle; không được tạo `.git` giả hoặc coi WARN không-Git là đã đạt phần Git. Khi phạm vi Git đã được giao, có thể đưa một git bundle chỉ gồm lịch sử tracked lên Kaggle để kiểm tra checkout ở thư mục mới, không chứa credential/untracked data.

## 9. Thứ tự ưu tiên

Hoàn thiện P0–P1 trước mọi train dài; E0/E1 trước để xác nhận pipeline học được; E2/E3 rồi E4; báo cáo và audit cuối; E5 sau cùng. Bước tiếp theo khi chuyển sang triển khai là viết nguồn/mã đóng gói local và chạy preflight thật trên Kaggle, chưa chạy toàn bộ train trong bước lập chiến lược này.
