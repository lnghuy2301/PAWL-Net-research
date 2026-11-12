# Ghi chú tái cấu trúc dự án

## Phạm vi

Dự án được làm sạch và sắp xếp để chuẩn bị chia sẻ mã nguồn phục vụ hồ sơ học thuật. Ở lần dọn dẹp ngày 09/11/2026, chưa tạo commit hay đẩy lên GitHub. Thông tin tác giả, cơ quan và ORCID lấy từ PDF theo yêu cầu của chủ dự án. Chưa lựa chọn giấy phép.

Ngày 12/11/2026, chủ dự án yêu cầu đưa mã lên repository đã đổi tên thành [PAWL-Net-research](https://github.com/lnghuy2301/PAWL-Net-research). Việc chuẩn bị sử dụng checkout sạch từ lịch sử trên GitHub; Git cục bộ bị lỗi được sao lưu trước khi thay thế. Các ghi chú bên dưới mô tả tình trạng tại lần dọn dẹp ban đầu.

Chủ dự án xác nhận bộ mã và quy trình hiện có là cơ sở tạo ra bài báo. Việc tái cấu trúc giữ các kiến trúc, tên khóa checkpoint, hằng số chuẩn hóa và siêu tham số mặc định của mã gốc. Không hiệu chỉnh dữ liệu, không huấn luyện lại các mô hình nghiên cứu và không thay số liệu đã báo cáo bằng kết quả mới.

## Các file đã được gom

| Vị trí cũ | Vị trí mới |
| --- | --- |
| `dataset.py` | `src/pawl_net/dataset.py`, `normalization.py`, `masks.py` |
| `model_unet.py` | `src/pawl_net/models/unet.py` |
| `model.py` | `src/pawl_net/models/swin.py`, `losses.py` |
| `train.py`, `train_unet.py`, ba script ablation | `scripts/train.py` và `src/pawl_net/train.py`, chọn bằng `--experiment` |
| `test.py`, `test_unet.py`, ba script ablation | `scripts/evaluate.py` và `src/pawl_net/evaluate.py` |
| `train_ml.py` | `scripts/random_forest.py` và `src/pawl_net/baselines.py` |
| `Cloud_Inpainting_Data_Preprocessing.ipynb` | `notebooks/01_data_preprocessing.ipynb`, `scripts/export_earth_engine.py`, `scripts/prepare_patches.py` |
| `UNet_Inpainting.ipynb` | `notebooks/02_analysis_and_inference.ipynb`, `src/pawl_net/inference.py` |
| `method.txt` | `docs/legacy_method_vi.md`, giữ nguyên nội dung lịch sử |
| `result.txt` | `reports/legacy_results.txt`, giữ nguyên các lần chạy và số liệu |
| `Data_Patches/S128_Test.rar` | `data/archives/S128_Test.rar` |
| `Experiments/checkpoints/*` | `outputs/checkpoints/*`, giữ nguyên byte và tên file |
| Môi trường Linux `venv/` cũ | `.local/venv/` |
| Cache Python ở thư mục gốc | `.local/__pycache__/` |

Bản nguyên gốc của toàn bộ script, notebook có output, các ghi chú và cấu hình được sao lưu trong `.local/original_files/`. `manifest.json` trong thư mục đó ghi kích thước và SHA-256. `.local/artifact_manifest.json` ghi SHA-256 của archive và tám file mô hình. Các thư mục `.local/`, `data/`, `outputs/` được bỏ qua bởi Git, ngoại trừ các README mô tả thư mục dữ liệu/đầu ra.

Archive được giải nén thành `data/patches/test/` để thuận tiện kiểm tra và sử dụng tại máy; file RAR gốc vẫn giữ nguyên. Xem [hồ sơ dữ liệu](data.md) về 324 file rỗng trong archive đang có. Không tự thay thế chúng bằng dữ liệu giả hay loại bỏ khỏi tập test.

## Những thay đổi về cách chạy

- Cài package bằng `python -m pip install -e ".[dev,notebooks]"` trong môi trường mới; môi trường `venv` cũ là của Linux.
- Tất cả đường dẫn dữ liệu/checkpoint trong YAML được tính từ `project_root`, không phụ thuộc thư mục chạy lệnh.
- Chọn thí nghiệm qua `--experiment`: `pawl`, `swin`, `unet-2ch-l1`, `unet-2ch-auto`, `unet-10ch-l1`.
- Các lần train mới ghi vào `outputs/checkpoints/runs/<run-name>/`, tránh ghi đè mô hình nghiên cứu.
- Import module không khởi chạy huấn luyện, nạp dữ liệu hay tạo thư mục đầu ra.
- CUDA và AMP có điều kiện; mặc định chạy được CPU. `num_workers=0` phù hợp với Windows; có thể chỉnh khi train thật.
- Validation/test dùng mặt nạ cố định theo seed và thứ tự file; số liệu lịch sử dùng mặt nạ ngẫu nhiên vẫn được giữ nguyên, không gán lại thành kết quả của mã mới.
- File hỏng/rỗng được báo lỗi cụ thể, thay vì âm thầm thay bằng mảng 0 như loader cũ.

Các khác biệt giữa mô tả PDF, ghi chú cũ và cách chạy trong mã được ghi riêng ở [reproducibility.md](reproducibility.md). Chế độ loss toàn ảnh và các hằng số gốc vẫn là mặc định. Các tùy chọn mới cần được xem là thí nghiệm mới khi sử dụng.

## Tình trạng Git có sẵn

Trước khi sửa mã, Git đã báo ownership khác tài khoản và nhiều object rỗng/hỏng; `git fsck --full` cũng phát hiện HEAD/ref không hợp lệ. Vì vậy không thể xem thư mục `.git` hiện tại là một lịch sử khỏe mạnh chỉ nhờ việc làm sạch mã.

Thư mục `.git`, cấu hình remote, index và lịch sử được giữ nguyên. Không chạy `reset`, `clean`, `gc`, xóa `.git` hay tự khởi tạo lịch sử mới. Lệnh đọc Git trong phiên kiểm tra chỉ dùng `-c safe.directory=...` cho từng lệnh, không thay đổi cấu hình toàn máy.

Trước lần đưa lên GitHub sau này, cần quyết định khôi phục lịch sử cũ hay tạo repository sạch từ cây mã nguồn đã chuẩn bị. Đây là việc riêng với tái cấu trúc; không có thao tác push nào trong lần làm việc này.

## Gói mã nguồn cục bộ

`dist/pawl-net-source.zip` chứa cây mã nguồn và tài liệu cùng manifest SHA-256. Gói này không chứa dữ liệu, checkpoint, môi trường, bản sao lưu hay thư mục `.git` đang lỗi. Có thể giải nén để xem hoặc chuẩn bị một repository mới vào thời điểm bạn quyết định xuất bản. Lệnh tạo lại là `python scripts/package_source.py`; công cụ không tải gì lên mạng.
