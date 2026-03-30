import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import UHIDataset
from model import PhysicsSwinUNet
import numpy as np

with open("configs/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

DEVICE = torch.device(config['training']['device'])
BEST_PATH = os.path.join(config['paths']['ckpt_dir'], 'best_model.pth')
TEST_DIR = config['paths']['test_patch_dir']

# 1. Load Data - Sử dụng mode='test' để lấy 100% dữ liệu
print(f"🔄 Đang tải tập Test 2023-2024 từ: {TEST_DIR}")
test_set = UHIDataset(TEST_DIR, mode='test')

if len(test_set) == 0:
    print("❌ Không tìm thấy dữ liệu Test. Hãy đảm bảo bạn đã chạy prepare_data.py cho dữ liệu 2023-2024.")
    exit()

test_loader = DataLoader(test_set, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)

# 2. Init Model & Load Best Checkpoint
model = PhysicsSwinUNet(in_channels=config['model']['in_channels'], img_size=config['model']['img_size']).to(DEVICE)

if os.path.exists(BEST_PATH):
    model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE, weights_only=True))
    print("✅ Đã load mô hình tốt nhất thành công!")
else:
    print("❌ Không tìm thấy best_model.pth. Vui lòng train mô hình trước.")
    exit()

# 3. Đánh giá (Evaluation)
model.eval()
total_mae = 0.0
total_rmse = 0.0

print("🚀 Đang chạy inference trên tập Test...")
with torch.no_grad():
    for inputs, targets, masks in test_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        preds = model(inputs)
        
        # Chỉ tính sai số ở những vùng BỊ MÂY CHE (Mask == 0)
        # mask_tensor có giá trị 0 ở vùng bị mây, 1 ở vùng thấy rõ
        # (targets - preds) ** 2 tính sai số từng pixel
        cloud_pixels = (masks == 0).float().to(DEVICE)
        
        # Nếu patch đó không có mây, bỏ qua tính toán để không bị lỗi chia 0
        if cloud_pixels.sum() > 0:
            mae = (torch.abs(preds - targets) * cloud_pixels).sum() / cloud_pixels.sum()
            mse = (((preds - targets)**2) * cloud_pixels).sum() / cloud_pixels.sum()
            
            total_mae += mae.item()
            total_rmse += torch.sqrt(mse).item()

avg_mae = total_mae / len(test_loader)
avg_rmse = total_rmse / len(test_loader)

# Denormalize về nhiệt độ thực tế (Độ C)
# Công thức: value * (MAX - MIN) + MIN
LST_MIN, LST_MAX = 293.0, 323.0 # 20C - 50C
temp_range = LST_MAX - LST_MIN

mae_celsius = avg_mae * temp_range
rmse_celsius = avg_rmse * temp_range

print("\n🏁 KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (2023 - 2024):")
print(f"🌡️  Sai số tuyệt đối trung bình (MAE): {mae_celsius:.2f} °C")
print(f"🌡️  Sai số bình phương trung bình (RMSE): {rmse_celsius:.2f} °C")