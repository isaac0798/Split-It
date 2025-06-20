from ultralytics import YOLO
import os

def train_model():
    # Load pre-trained model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    results = model.train(
        data='./data/Split-It.v2i.yolov8/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='glass_detector',
        patience=20,
        save=True,
        cache=True
    )
    
    return model

if __name__ == "__main__":
    model = train_model()
    print("Training completed!")