import os
import yaml
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from dataset import UHIDataset
from model import PhysicsSwinUNet, CharbonnierLoss, EdgeLoss, AutomaticWeightedLoss

with open("configs/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

os.makedirs(config['paths']['ckpt_dir'], exist_ok=True)
BEST_PATH = os.path.join(config['paths']['ckpt_dir'], 'best_model.pth')
LAST_PATH = os.path.join(config['paths']['ckpt_dir'], 'last_model.pth')
DEVICE = torch.device(config['training']['device'])

# 1. Load Data
print("🔄 Loading Train/Val Data...")
train_set = UHIDataset(config['paths']['train_patch_dir'], mode='train')
val_set = UHIDataset(config['paths']['train_patch_dir'], mode='val')

train_loader = DataLoader(train_set, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

# 2. Init Model & Loss
model = PhysicsSwinUNet(in_channels=config['model']['in_channels'], img_size=config['model']['img_size']).to(DEVICE)
criterion_pixel = CharbonnierLoss().to(DEVICE)
criterion_edge = EdgeLoss().to(DEVICE)
criterion_wrapper = AutomaticWeightedLoss().to(DEVICE)

optimizer = optim.AdamW([
    {'params': model.parameters()},
    {'params': criterion_wrapper.parameters(), 'lr': 1e-3}
], lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
scaler = GradScaler()

# 3. Resume & Early Stop
start_epoch, best_val_loss, early_stop_counter = 0, float('inf'), 0

if os.path.exists(LAST_PATH):
    print("🔄 Resuming from checkpoint...")
    checkpoint = torch.load(LAST_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    criterion_wrapper.load_state_dict(checkpoint['loss_weights'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

# 4. Training Loop
print("🚀 Bắt đầu huấn luyện...")
for epoch in range(start_epoch, config['training']['epochs']):
    model.train()
    criterion_wrapper.train()
    train_loss = 0.0
    
    loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{config['training']['epochs']}")
    for inputs, targets, masks in loop:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion_wrapper(criterion_pixel(outputs, targets), criterion_edge(outputs, targets))
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets, masks in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            val_loss += F.l1_loss(model(inputs), targets).item()
            
    avg_val_loss = val_loss / len(val_loader)
    scheduler.step()
    
    print(f"   => Val (L1): {avg_val_loss:.4f} | Weights: {criterion_wrapper.params.data.tolist()}")

    # Save
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), BEST_PATH)
        print(f"   🌟 New Best Saved!")
    else:
        early_stop_counter += 1

    torch.save({
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'loss_weights': criterion_wrapper.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(), 'best_val_loss': best_val_loss
    }, LAST_PATH)

    if early_stop_counter >= config['training']['patience']:
        print("🛑 EARLY STOPPING!")
        break