from ultralytics import YOLO
import multiprocessing
import os
import torch

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def train_model():
    # 確保 CUDA 可用
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，請確認 GPU 驅動與 PyTorch 安裝正確！")
    
    # torch.cuda.synchronize()  # 確保 CUDA 錯誤即時報告
    
    # 顯示 CUDA 設備資訊
    print(f"是否使用 CUDA: {torch.cuda.is_available()}")
    print(f"當前使用的 CUDA 設備數量: {torch.cuda.device_count()}")
    print(f"當前 CUDA 設備: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # 加載模型並轉移到 GPU
    model = YOLO("yolo11m-seg.pt")
    
    results = model.train(
        data="yolo/data.yaml",
        epochs=100,
        imgsz=640,
        device="cuda",
        batch=32,
        optimizer="SGD",
        cache='cuda',
        workers=4,
        pretrained=False,
        plots=False, 
        save=False, 
        val=False,
        )
    print("Training complete.")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train_model()
