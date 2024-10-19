from ultralytics import YOLO

# Load a model

model = YOLO('models/FOMDet.yaml')


# Train the model
model.train(data='data/SUODAC.yaml', epochs=300, imgsz=640, batch=8, resume=True)
