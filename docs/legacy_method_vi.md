### TÓM TẮT PHƯƠNG PHÁP THU THẬP DỮ LIỆU VÀ THỰC HIỆN NGHIÊN CỨU

#### 1. Phương pháp Thu thập và Tiền xử lý Dữ liệu (Data Collection & Preprocessing)
*   **Nguồn dữ liệu:** Khai thác sản phẩm vệ tinh Landsat-8 Level-2 (Collection-2) thông qua nền tảng Google Earth Engine.
*   **Khu vực nghiên cứu (ROIs):** Ba thành phố đại diện cho ba hệ sinh thái đặc trưng tại Đồng bằng Sông Cửu Long: Cần Thơ (đô thị lõi ven sông), Rạch Giá (vùng ven biển / đất ngập nước) và Long Xuyên (vùng canh tác nông nghiệp lúa nước).
*   **Thời gian và Bộ lọc:** Thu thập từ Tháng 1/2015 đến Tháng 12/2024. Loại bỏ các cảnh ảnh (scenes) có tỷ lệ mây che phủ tự nhiên >80%.
*   **Chiến lược phân chia (Data Split):** Áp dụng chia tách theo thời gian nghiêm ngặt (Strict temporal hold-out):
    *   **Train/Val Set (80/20):** Sử dụng dữ liệu từ năm 2015 - 2022.
    *   **Test Set (Tập kiểm thử):** Độc lập 100%, sử dụng toàn bộ dữ liệu 2023 - 2024 (chứa 37,101,299 pixels bị mây che) để đánh giá năng lực khái quát hóa (generalization) thực tế.
*   **Tổ chức đầu vào (10-Channel Input):** Ảnh được cắt thành các bản vá (patches) 128x128 pixel (stride 64). Khối đầu vào gồm 10 kênh:
    *   *1 kênh mục tiêu:* LST (chuẩn hóa Min-Max về [0,1]).
    *   *3 kênh chỉ số quang học:* NDVI, NDBI, MNDWI (được tái tỷ lệ từ [-1, 1] về[0, 1]).
    *   *2 kênh thông số bề mặt:* Fractional Vegetation Cover (FVC) và Land Surface Emissivity (LSE).
    *   *3 kênh không gian:* Distance-to-Water, Elevation, Population Density.
    *   *1 kênh mặt nạ:* Binary Cloud Mask.

#### 2. Kỹ thuật Mô phỏng Mặt nạ Mây (Cloud Mask Simulation)
Do Landsat-8 hiếm khi có ảnh Ground Truth (ảnh hoàn toàn sạch mây) trùng khớp ngày với ảnh có mây, nghiên cứu tạo ra mây nhân tạo (synthetic clouds) trực tiếp trong quá trình huấn luyện bằng cách gán giá trị 0 cho 6 kênh quang học/nhiệt, giữ lại 4 kênh bổ trợ không gian (Elevation, Distance-to-water, Population, Mask). 
Mặt nạ mây được sinh ra ngẫu nhiên bao phủ từ 10% đến 70% diện tích bản vá, kết hợp 3 thuật toán:
*   **Random Blocks:** Mô phỏng mây tích (cumulus).
*   **Freeform Brush:** Mô phỏng các dải mây ti (cirrus) mỏng và dài.
*   **Perlin Noise:** Mô phỏng các mảng mây tầng (stratus) diện rộng.

#### 3. Phương pháp Triển khai Kiến trúc và Tối ưu hóa (Research Implementation)
Dựa trên phát hiện về "Nghịch lý phân phối đặc trưng", nghiên cứu triển khai theo 2 trục cốt lõi:
*   **Kiến trúc Mạng đề xuất:** Khung Inpainting được xây dựng dựa trên kiến trúc **Standard U-Net (CNN)** thay vì Transformer. Cấu trúc nội suy mạng cho phép bảo toàn định kiến quy nạp cục bộ (local inductive bias), ưu việt trong việc khôi phục gradient nhiệt độ không gian mượt mà trên các bản vá 128x128. Lớp đầu vào được sửa đổi để tiếp nhận khối tensor 10 kênh vật lý.
*   **Chiến lược Tối ưu hóa (Cốt lõi của nghiên cứu):** Đề xuất cơ chế **Automatic Weighted Loss (Auto Loss)**. Sử dụng lý thuyết học độ bất định đồng phương sai (Homoscedastic Uncertainty) để tự động học và cân bằng trọng số giữa 2 hàm suy hao:
    *   *Charbonnier Loss:* Tối ưu hóa độ trung thực của từng điểm ảnh (pixel-level fidelity), bền vững với nhiễu.
    *   *Laplacian Edge Loss:* Trừng phạt sự đứt gãy cấu trúc không gian (áp dụng bộ lọc Laplacian lên phần dư).
    *   *Lý do:* Cơ chế Auto Loss này là "chìa khóa" giải quyết hiện tượng nhiễu gradient khi nhồi nhét 10 kênh dữ liệu có phân phối vật lý khác nhau vào cùng một mạng.

#### 4. Kịch bản Đánh giá và So sánh (Evaluation Scenarios)
Đánh giá so sánh được thiết kế khắt khe với 2 Baseline mạnh:
*   **Baseline Học máy (Random Forest):** Huấn luyện trên 3% mẫu điểm ảnh ngẫu nhiên để tránh tràn RAM, đánh giá trên 100% tập Test. Mục đích: Chứng minh Inpainting là bài toán bắt buộc phải dùng ngữ cảnh không gian (spatial context) của Deep Learning.
*   **Baseline Transformer (Swin-UNet):** Dùng trọng số pre-trained ImageNet-21k, chuyển giao học thuật để so sánh trực tiếp cơ chế Global Attention (ViT) với Local Inductive Bias (CNN).
*   **Ablation Study (Nghiên cứu cắt bỏ):** Chạy kiểm chứng trên U-Net với 4 cấu hình: (1) 2 kênh + L1, (2) 2 kênh + Auto Loss, (3) 10 kênh + L1, và (4) 10 kênh + Auto Loss (Khung đề xuất). 

#### 5. Thiết lập Cấu hình Huấn luyện (Training Setup)
*   **Optimizer:** AdamW. Khởi tạo Learning Rate (LR) ở mức $10^{-4}$, riêng các trọng số của hàm Auto Loss được gán LR là $10^{-3}$ để hội tụ nhanh.
*   **Scheduler:** CosineAnnealingWarmRestarts để tránh sa lầy vào cực tiểu cục bộ (local minima).
*   **Tối ưu tài nguyên:** Huấn luyện độ chính xác hỗn hợp (Mixed-precision training - `torch.cuda.amp`) giúp giảm 40% VRAM sử dụng.
*   **Kiểm soát học vẹt:** Cơ chế Dừng sớm (Early Stopping) với `patience=20` theo dõi L1 loss trên tập Validation chưa bị che mây.
