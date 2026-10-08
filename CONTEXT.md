# PTIT AI Challenge 77

Bài toán dịch câu tiếng Trung sang tiếng Việt. Các thuật ngữ dưới đây phân biệt dữ liệu có bản dịch tham chiếu với dữ liệu cần dự đoán để nộp bài.

## Language

**Cặp dịch Trung–Việt**:
Một câu tiếng Trung và bản dịch tiếng Việt tham chiếu tương ứng trong dữ liệu huấn luyện của cuộc thi.
_Avoid_: Hai danh sách câu độc lập

**Validation nội bộ**:
Phần dữ liệu có bản dịch tham chiếu được tách khỏi dữ liệu dùng để học nhằm so sánh chất lượng các phương án dịch; không phải public test.
_Avoid_: Public test, điểm leaderboard

**Submission**:
Bài nộp chứa bản dịch tiếng Việt dự đoán cho các câu tiếng Trung của bộ test tương ứng, theo định dạng cuộc thi yêu cầu.
_Avoid_: Bản dịch tham chiếu
