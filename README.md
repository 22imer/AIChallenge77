# PTIT AI Challenge 77

Hai notebook dịch máy Trung → Việt: baseline GRU gốc và notebook fine-tune Marian pretrained riêng. Đối chiếu metric và định dạng submission với quy định chính thức trước khi nộp.

## Fine-tune pretrained trên Colab

Mở [`finetune_colab.ipynb`](finetune_colab.ipynb) bằng Colab, hoặc dùng [Open in Colab](https://colab.research.google.com/github/22imer/AIChallenge77/blob/main/finetune_colab.ipynb) sau khi notebook được đưa lên GitHub. File mới ở local chưa tự xuất hiện trên GitHub.

1. Chọn runtime GPU. Đặt ZIP tại `MyDrive/AIChallenge77/datasets/dataset.zip`.
2. Sửa `ROOT` nếu dùng đường dẫn Drive khác; giữ một `RUN_NAME` cho một cấu hình/dataset.
3. Cell `%pip install` đang được comment theo bản Colab. Với runtime mới chưa có dependency, bỏ dấu `#` và chạy cell đó trước; sau đó Run All để mount Drive, staging dữ liệu về `/content`, fine-tune và xuất bài nộp.

Checkpoint là [`Helsinki-NLP/opus-mt-zh-vi`](https://huggingface.co/Helsinki-NLP/opus-mt-zh-vi), chuyên dịch Trung–Việt, Apache-2.0; revision được cố định trong notebook. Chỉ dùng dữ liệu cuộc thi để fine-tune; không bổ sung corpus ngoài. Giữ nguyên `baseline.ipynb`.

- Validation cố định theo nhóm câu nguồn, loại cặp trùng trước khi chia; chấm với reference nguyên văn. SacreBLEU `tokenize="none"` chấm token phân cách bằng khoảng trắng, không detokenize; đây là metric nội bộ, chưa xác nhận chuẩn chấm chính thức.
- Pilot thử batch size từ 128 xuống 1, đo bộ nhớ/tốc độ trên GPU thực tế và chọn mức đầu tiên dưới 80% VRAM. Training mỗi lần tối đa **3 giờ theo giới hạn mềm**, dành 10 phút cuối để lưu/dừng. Thao tác GPU/Drive đang chạy có thể vượt mốc này; download, đánh giá pretrained ban đầu, đánh giá cuối và inference nằm ngoài ngân sách training.
- Lưu weights, tokenizer, optimizer, AMP scaler và RNG lên Drive mỗi khoảng 10 phút và ở ranh giới epoch. Phiên bị ngắt có thể mất công việc từ lần lưu gần nhất; Run All lại với cấu hình cũ để resume. Không chạy đồng thời hai phiên cùng `RUN_NAME`.
- `checkpoint-index.json` cập nhật nguyên tử cả lựa chọn `latest` để tiếp tục training và `best` theo validation BLEU, kể cả pretrained ban đầu nếu fine-tune chưa cải thiện. Chỉ snapshot đã có marker hoàn tất được chọn.
- Dành khoảng **4–5 GiB Drive trống** cho checkpoint. Đổi cấu hình hoặc dữ liệu thì đổi `RUN_NAME`; không trộn checkpoint/tokenizer cũ.
- Run mặc định cho metric mới là `marian-zh-vi-none-v1`; cấu hình checkpoint ghi `bleu_tokenize: none`. Checkpoint `13a` cũ được giữ nguyên, không tự resume hoặc so `best_bleu` cũ với điểm `none`. Run mới bắt đầu từ pretrained.
- `metrics.json` và checkpoints: `MyDrive/AIChallenge77/checkpoints/<RUN_NAME>/`.
- CSV/ZIP: `MyDrive/AIChallenge77/submissions/<RUN_NAME>/nlp_submission.csv` và `nlp_submission.zip`. Đổi `TEST_SPLIT` sang `private_test` để dịch private; tên bài nộp vẫn là `nlp_submission.*`.

Input/target training giới hạn 256 token; notebook báo tỷ lệ bị cắt. Validation references không bị cắt, nhưng input inference/output generation vẫn có giới hạn. BLEU của baseline gốc dùng split/reference khác, **không so trực tiếp** với BLEU notebook mới.

Kiểm tra hồi quy dữ liệu ở local, không cần GPU:

```bash
python -m unittest discover -s tests -v
```

Kiểm chứng local: 7 kiểm tra hồi quy dữ liệu/checkpoint; smoke CPU dùng checkpoint pretrained thật (77.943.296 tham số), 4 cặp train và 2 cặp validation. Đã chạy optimizer, save/reload/resume (toàn bộ weights khớp chạy liền), nhánh hết ngân sách, full-reference evaluation và xuất CSV/ZIP từ checkpoint được chọn. Không dùng BLEU của smoke nhỏ này để kết luận chất lượng.

Chưa chạy trên GPU Colab hoặc mount Google Drive thực tế; pilot CUDA, thời gian train 3 giờ và BLEU trên toàn bộ validation cần đo trong phiên Colab. Tokenizer audit trên ZIP hiện tại: source train dài nhất 72 token, target train 89 token, public/private source 50 token; không câu nào vượt giới hạn 256.

## Baseline GRU gốc

## Mở notebook trên Colab

[Open in Colab](https://colab.research.google.com/github/22imer/AIChallenge77/blob/main/baseline.ipynb)

Chọn GPU trong cài đặt runtime. Notebook baseline có cell mount Drive với đường dẫn dataset cũ; cập nhật đường dẫn trước khi chạy. Đây **chưa phải runner `.py`**. Phần dưới hướng dẫn chuẩn bị dữ liệu cho baseline, không cần thêm các cell này vào notebook fine-tune mới.

## Chuẩn bị dữ liệu

Dataset, checkpoint, tokenizer và submission không được commit. Giữ dataset trên Google Drive:

```text
MyDrive/AIChallenge77/
├── datasets/dataset.zip
├── checkpoints/
└── submissions/
```

Thêm cell này trước các cell baseline:

```python
from google.colab import drive
from pathlib import Path
import shutil
import subprocess
import os

drive.mount('/content/drive')
root = Path('/content/drive/MyDrive/AIChallenge77')
for name in ('checkpoints', 'submissions'):
    (root / name).mkdir(parents=True, exist_ok=True)
Path('/content/data').mkdir(parents=True, exist_ok=True)
shutil.copy2(root / 'datasets/dataset.zip', '/content/dataset.zip')
subprocess.run(['unzip', '-oq', '/content/dataset.zip', '-d', '/content/data'], check=True)
os.chdir('/content/data')
```

Trong cell configuration của baseline, thay `SAVE_DIR` bằng:

```python
SAVE_DIR = '/content/drive/MyDrive/AIChallenge77/checkpoints'
```

Giữ đường dẫn `dataset/train/train.zh`, `dataset/train/train.vi` và `dataset/public_test/public_test.zh`: ZIP chứa thư mục `dataset/`, được giải nén vào `/content/data`.

Trong cell xuất CSV và cell tạo ZIP, thay các đường dẫn output:

```python
submission_path = '/content/drive/MyDrive/AIChallenge77/submissions/public_submission.csv'
zip_path = '/content/drive/MyDrive/AIChallenge77/submissions/public_submission.zip'
```

Baseline lưu `best_model.pt`, `spm_zh.model` và `spm_vi.model` vào `SAVE_DIR`. Model weights phải dùng đúng tokenizer đi kèm. Không train trực tiếp từ hàng nghìn file trên Drive khi có thể staging về `/content`.

Cài dependency trong một cell trước phần imports:

```python
%pip install -q pandas sentencepiece sacrebleu tqdm
```

Không tự reinstall PyTorch GPU của Colab.

## GitHub → Colab

Repo public, clone không cần token:

```bash
git clone https://github.com/22imer/AIChallenge77.git /content/AIChallenge77
```

Sau đó cập nhật source bằng `git -C /content/AIChallenge77 pull --ff-only`. Mở notebook từ GitHub tạo một bản notebook trong Colab; nó không tự đồng bộ khi GitHub thay đổi.

## Local / VS Code

Mở thư mục project hiện tại bằng VS Code. Sau khi sửa source:

```bash
git add baseline.ipynb README.md .gitignore
git commit -m "experiment: update baseline"
git push
```

Xóa output notebook trước khi commit để tránh public các câu dữ liệu và prediction nằm trong output. `.gitignore` chỉ chặn file ngoài notebook, không chặn nội dung được nhúng trong notebook.

Repo chứa hai notebook; chưa có `src/train.py`, `src/inference.py` hoặc `notebooks/colab_runner.ipynb`. Vì vậy command `python -m src.train` chưa dùng được.
