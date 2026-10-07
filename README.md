# PTIT AI Challenge 77

Baseline dịch máy Trung → Việt: PyTorch GRU Seq2Seq, SentencePiece BPE và đánh giá validation bằng SacreBLEU. Đối chiếu metric và định dạng submission với quy định chính thức trước khi nộp.

## Mở notebook trên Colab

[Open in Colab](https://colab.research.google.com/github/22imer/AIChallenge77/blob/main/baseline.ipynb)

Chọn GPU trong cài đặt runtime. Notebook hiện là baseline gốc, **chưa phải runner `.py`**: chưa tự mount Drive hoặc restore dataset. Không Run All trước khi chuẩn bị dữ liệu và chỉnh đường dẫn như dưới đây.

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

Hiện repo chứa baseline notebook; chưa có `src/train.py`, `src/inference.py` hoặc `notebooks/colab_runner.ipynb`. Vì vậy command `python -m src.train` chưa dùng được.
