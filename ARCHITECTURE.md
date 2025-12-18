# Phân tích Kiến trúc Kỹ thuật - Mô hình Profiling Giọng nói Đa nhiệm

Tài liệu này phân tích chi tiết kiến trúc của mô hình `MultiTaskSpeakerModel` trong `src/models.py`, được thiết kế cho bài toán phân loại giới tính và giọng địa phương từ âm thanh.

## 1. Tổng quan Kiến trúc

Mô hình được xây dựng theo kiến trúc pipeline linh hoạt, có thể thay đổi "xương sống" (backbone) là các encoder xử lý giọng nói nổi tiếng.

Luồng xử lý dữ liệu tổng thể như sau:

**Audio -> Encoder Backbone -> Attentive Pooling -> LayerNorm -> Classification Heads -> Multi-task Loss**

Kiến trúc này cho phép:
-   **Học đa nhiệm (Multi-task Learning):** Cùng lúc học hai tác vụ (giới tính và giọng địa phương) trên một mô hình duy nhất, giúp các tác vụ bổ trợ thông tin cho nhau.
-   **Encoder linh hoạt:** Dễ dàng thay thế các encoder khác nhau (WavLM, HuBERT, Whisper, ECAPA-TDNN) để thử nghiệm và tìm ra mô hình phù hợp nhất.
-   **Tập trung vào đặc trưng quan trọng:** Sử dụng cơ chế `AttentivePooling` để mô hình tự học cách "chú ý" vào những phần quan trọng của tín hiệu âm thanh.

## 2. Các Thành phần Chính

### 2.1. Encoder Backbone

Đây là thành phần đầu tiên, chịu trách nhiệm chuyển đổi tín hiệu âm thanh thô thành chuỗi các vector đặc trưng cấp cao (`last_hidden_state`). Mã nguồn hỗ trợ nhiều loại encoder thông qua `ENCODER_REGISTRY`. Việc xử lý các encoder khác nhau được thể hiện trong phương thức `_encode`.

#### Mã nguồn tham khảo (`_encode` method):
```python
def _encode(
    self, 
    input_values: torch.Tensor, 
    attention_mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Trích xuất các trạng thái ẩn (hidden states) từ encoder.
    """
    # Trường hợp 1: ECAPA-TDNN đã tự pooling, trả về vector [B, 1, H]
    if self.is_ecapa:
        outputs = self.encoder(input_values, attention_mask)
        hidden_states = outputs.last_hidden_state
        
    # Trường hợp 2: Whisper có kiến trúc Encoder-Decoder,
    # ta chỉ sử dụng phần encoder của nó.
    elif self.is_whisper:
        outputs = self.encoder.encoder(input_values)
        hidden_states = outputs.last_hidden_state
        
    # Trường hợp 3: Các mô hình còn lại (WavLM, HuBERT, Wav2Vec2)
    # là các encoder tiêu chuẩn.
    else:
        outputs = self.encoder(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
    
    return hidden_states
```

### 2.2. Lớp `AttentivePooling`

Đây là trái tim của việc tổng hợp đặc trưng, giúp chuyển chuỗi đặc trưng `[B, T, H]` từ encoder thành một vector đại diện duy nhất `[B, H]`.

#### Mã nguồn tham khảo (`AttentivePooling` class):
```python
class AttentivePooling(nn.Module):
    """
    Lớp pooling dựa trên cơ chế chú ý để tổng hợp chuỗi theo chiều thời gian.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        # Một mạng neural nhỏ để học cách tính "điểm quan trọng"
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), # Lớp biến đổi
            nn.Tanh(),                         # Hàm kích hoạt phi tuyến
            nn.Linear(hidden_size, 1, bias=False) # Lớp tạo ra điểm số cuối cùng
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # x có kích thước [Batch, Time, Hidden_dim]
        
        # 1. Tính điểm chú ý thô cho mỗi bước thời gian
        attn_weights = self.attention(x)  # -> [B, T, 1]
        
        # 2. (Tùy chọn) Che đi các phần đệm (padding)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            # Gán điểm số của các phần padding bằng một số âm rất lớn
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        
        # 3. Dùng softmax để chuẩn hóa điểm số thành trọng số (tổng bằng 1)
        attn_weights = F.softmax(attn_weights, dim=1) # -> [B, T, 1]
        
        # 4. Tính tổng có trọng số
        # Nhân mỗi vector trong chuỗi với trọng số tương ứng
        # và tính tổng theo chiều thời gian
        pooled = torch.sum(x * attn_weights, dim=1) # -> [B, H]
        
        return pooled, attn_weights.squeeze(-1)
```

### 2.3. Lớp `LayerNorm` (Chuẩn hóa)

Lớp này được áp dụng ngay sau bước pooling để chuẩn hóa vector đặc trưng, giúp ổn định quá trình huấn luyện.

#### Mã nguồn tham khảo (từ `MultiTaskSpeakerModel.forward`):
```python
# ...
# Sau khi pooling, ta có vector `pooled`
pooled, attn_weights = self.attentive_pooling(hidden_states, pooled_mask)

# Chuẩn hóa vector sau khi pooling và áp dụng dropout
pooled = self.layer_norm(pooled)
pooled = self.dropout(pooled)
# ...
```

### 2.4. Các Đầu phân loại (Classification Heads)

Sau khi có vector đặc trưng cuối cùng, nó được đưa vào hai nhánh mạng riêng biệt. Thiết kế này cho phép mỗi tác vụ có bộ phân loại riêng, phù hợp với độ phức tạp của nó.

#### Mã nguồn tham khảo (từ `MultiTaskSpeakerModel.__init__`):
```python
# self.hidden_size là kích thước của vector đặc trưng sau khi pooling
hidden_size = self.encoder.config.hidden_size

# ... (Các lớp pooling, norm, dropout)

# Đầu phân loại giới tính: mạng 2 lớp, nhiệm vụ đơn giản hơn
self.gender_head = nn.Sequential(
    nn.Linear(hidden_size, head_hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(head_hidden_dim, num_genders)
)

# Đầu phân loại giọng địa phương: mạng 3 lớp, sâu hơn cho nhiệm vụ phức tạp hơn
self.dialect_head = nn.Sequential(
    nn.Linear(hidden_size, head_hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(head_hidden_dim, head_hidden_dim // 2),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(head_hidden_dim // 2, num_dialects)
)
```

### 2.5. Hàm Mất mát Đa nhiệm (Multi-task Loss)

Mô hình được huấn luyện bằng cách tối ưu một hàm mất mát kết hợp, với trọng số để cân bằng giữa hai tác vụ.

#### Mã nguồn tham khảo (từ `MultiTaskSpeakerModel.forward`):
```python
# ... (lấy ra gender_logits và dialect_logits từ các head)

loss = None
# Chỉ tính loss nếu có nhãn được cung cấp (trong lúc huấn luyện)
if gender_labels is not None and dialect_labels is not None:
    loss_fct = nn.CrossEntropyLoss()
    
    # Tính loss cho từng tác vụ
    gender_loss = loss_fct(gender_logits, gender_labels)
    dialect_loss = loss_fct(dialect_logits, dialect_labels)
    
    # Kết hợp loss: cộng loss giới tính với loss giọng địa phương đã nhân trọng số
    # Trọng số self.dialect_loss_weight (mặc định=3.0) giúp mô hình
    # tập trung hơn vào tác vụ khó hơn là phân loại giọng.
    loss = gender_loss + self.dialect_loss_weight * dialect_loss
```

## 3. Các Chế độ Hoạt động

Mô hình được thiết kế rất linh hoạt, hỗ trợ hai chế độ hoạt động chính.

### Chế độ 1: Huấn luyện End-to-End (sử dụng `MultiTaskSpeakerModel`)
-   **Đầu vào:** Sóng âm thô (`input_values`).
-   **Quá trình:** Toàn bộ pipeline từ encoder đến tính loss được thực thi.
-   **Ưu điểm:** Cho phép tinh chỉnh (fine-tune) cả encoder, có thể dẫn đến kết quả tốt nhất.
-   **Nhược điểm:** Tốn nhiều tài nguyên (bộ nhớ GPU) và thời gian huấn luyện.

### Chế độ 2: Huấn luyện với Đặc trưng đã trích xuất (sử dụng `ClassificationHeadModel`)
-   **Quá trình:**
    1.  **Bước tiền xử lý:** Sử dụng `model.get_embeddings()` để trích xuất và lưu lại các vector đặc trưng từ tất cả audio trong bộ dữ liệu.
    2.  **Huấn luyện:** Chỉ huấn luyện lớp `ClassificationHeadModel`, nhận đầu vào là các đặc trưng đã được lưu (`input_features`).
-   **Ưu điểm:**
    -   **Tiết kiệm bộ nhớ:** Không cần tải encoder nặng nề vào GPU trong lúc huấn luyện.
    -   **Tốc độ huấn luyện cực nhanh:** Bỏ qua được bước encoding tốn kém nhất.
-   **Nhược điểm:** Không thể tinh chỉnh được encoder.
