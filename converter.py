from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("runs/detect/train/weights/best.pt")

# Export the model
model.export(format="imx", data="data.yaml")  # exports with PTQ quantization by default
