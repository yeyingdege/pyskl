# pip install ultralytics
from ultralytics import YOLO

model = YOLO(".cache/yolo11m-pose.pt") # load a pretrained model
# Train the model with 2 GPUs
results = model.train(
    data="data/gen-yolo-pose3d/data.yaml",
    epochs=10,
    imgsz=640,
    batch=16,
    lr0=0.001,
    device=[1]
)

