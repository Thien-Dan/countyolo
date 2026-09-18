import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import cv2

from models.density_yolo import DensityYOLOWorld

def generate_density_map(image_shape, points, sigma=4):
    """
    Sinh ra Heatmap Mật độ (Density Map) từ tập hợp các điểm Ground Truth.
    """
    H, W = image_shape
    density_map = np.zeros((H, W), dtype=np.float32)
    
    if len(points) == 0:
        return density_map

    for point in points:
        x, y = min(int(point[0]), W-1), min(int(point[1]), H-1)
        density_map[y, x] = 1.0
        
    # Gaussian blur để tạo heatmap lan tỏa
    density_map = cv2.GaussianBlur(density_map, (15, 15), sigma)
    # Scale sao cho tổng giá trị map xấp xỉ số điểm (count)
    if density_map.sum() > 0:
        density_map = density_map / density_map.sum() * len(points)
        
    return density_map

class FSC147Dataset(Dataset):
    def __init__(self, data_dir="data/FSC147", split="train"):
        self.img_dir = os.path.join(data_dir, "images_384_VarV2")
        split_path = os.path.join(data_dir, "Train_Test_Val_FSC_147.json")
        anno_path = os.path.join(data_dir, "annotation_FSC147_384.json")
        
        with open(split_path, 'r') as f:
            self.images = json.load(f)[split]
            
        with open(anno_path, 'r') as f:
            self.annos = json.load(f)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Đọc ảnh
        image = Image.open(img_path).convert("RGB")
        W, H = image.size
        
        # Resize/Pad (để đưa vào YOLO) -> chuẩn hóa 640x640 cho dễ train demo
        image = image.resize((640, 640))
        img_tensor = TF.to_tensor(image) # (3, 640, 640)
        
        # Đọc points và tính scale
        gt_points = self.annos.get(img_name, {}).get("points", [])
        scaled_points = []
        for p in gt_points:
            scaled_points.append([p[0] * 640 / W, p[1] * 640 / H])
            
        # Target Density Map: P4 stride là 16, nên map size = 640/16 = 40
        density_map = generate_density_map((40, 40), [[p[0]/16, p[1]/16] for p in scaled_points])
        density_tensor = torch.from_numpy(density_map).unsqueeze(0) # (1, 40, 40)
        
        return img_tensor, density_tensor

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Bắt đầu huấn luyện trên: {device}")
    
    # 1. Khởi tạo Mô hình
    model = DensityYOLOWorld("yolov8s-world.pt").to(device)
    model.train_mode() # Đóng băng YOLO, chỉ train Density Head
    
    # 2. Dataset & DataLoader (Demo dùng subset nhỏ để test code)
    print("Đang nạp dữ liệu (FSC-147 Train)...")
    dataset = FSC147Dataset(split="train")
    
    # Chỉ lấy 100 ảnh demo cho chạy thử
    subset_indices = list(range(100))
    subset = torch.utils.data.Subset(dataset, subset_indices)
    dataloader = DataLoader(subset, batch_size=4, shuffle=True)
    
    # 3. Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.density_head.parameters(), lr=1e-4)
    
    epochs = 5
    print(f"Bắt đầu Training trong {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.density_head.train()
        
        for images, gt_density in dataloader:
            images = images.to(device)
            gt_density = gt_density.to(device)
            
            optimizer.zero_grad()
            
            # Forward: lấy Density Map dự đoán
            _, pred_density = model(images)
            
            # MSE Loss giữa Heatmap dự đoán và Heatmap chuẩn
            loss = criterion(pred_density, gt_density)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss / len(dataloader):.6f}")

    print("Huấn luyện hoàn tất. Lưu trọng số Density Head...")
    torch.save(model.density_head.state_dict(), "density_head_best.pth")

if __name__ == "__main__":
    train()
