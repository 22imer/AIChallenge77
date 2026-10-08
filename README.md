# PTIT AI Challenge 77 — Trung → Việt

Chạy [`finetune_colab.ipynb`](finetune_colab.ipynb) trên [Colab](https://colab.research.google.com/github/22imer/AIChallenge77/blob/main/finetune_colab.ipynb).

## Cấu hình đã chốt

- Model: [`Helsinki-NLP/opus-mt-zh-vi`](https://huggingface.co/Helsinki-NLP/opus-mt-zh-vi), revision cố định, Apache-2.0.
- Batch **64** cho train và inference; không auto chọn/hạ batch.
- Input và target tối đa **512 token**, padding theo câu dài nhất trong batch.
- BLEU **`tokenize="none"`**, giữ nguyên `_` và khoảng trắng; reference validation không bị cắt.
- Chỉ dùng dữ liệu cuộc thi để fine-tune. Validation cố định theo nhóm câu nguồn, loại cặp trùng.

## Chạy

1. Chọn GPU Colab. Đặt ZIP tại `MyDrive/AIChallenge77/datasets/dataset.zip`.
2. Nếu runtime mới thiếu dependency, bỏ `#` ở cell `%pip install` và chạy cell đó. Không reinstall PyTorch GPU.
3. Chỉnh `ROOT` nếu cần, rồi Run All.

Run mặc định: **`marian-zh-vi-b64-t512`**. Không trộn với checkpoint cấu hình cũ.

Mỗi lần chạy cell training có ngân sách mềm **3 giờ**, dành 10 phút cuối để lưu/dừng. Lưu định kỳ và validation cuối epoch được tính trong ngân sách; chuẩn bị dữ liệu, đánh giá cuối và xuất submission nằm ngoài. Thao tác GPU/Drive đang chạy có thể vượt mốc dừng.

Checkpoint chứa weights/tokenizer, optimizer, scaler, RNG và vị trí batch. Phiên mới: giữ cấu hình và Run All để resume; có thể mất phần từ lần lưu gần nhất (~10 phút). Không chạy đồng thời hai phiên cùng run. Cần khoảng 4–5 GiB Drive trống.

## Kết quả

- Checkpoint tốt nhất theo validation BLEU và `metrics.json`: `MyDrive/AIChallenge77/checkpoints/<RUN_NAME>/`.
- Bài nộp: `MyDrive/AIChallenge77/submissions/<RUN_NAME>/nlp_submission.csv` và `nlp_submission.zip`.
- `TEST_SPLIT` chọn `public_test` hoặc `private_test`; tên bài nộp giữ nguyên.

Metric nội bộ chưa được đối chiếu với chuẩn chấm chính thức. Không commit dữ liệu, weights hoặc notebook outputs. `baseline.ipynb` được giữ nguyên để tham khảo.

Kiểm tra dữ liệu/checkpoint ở local:

```bash
python -B -m unittest discover -s tests -v
```
