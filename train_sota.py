from ultralytics import YOLO

model = YOLO("/root/ultralytics/cfg/SL-YOLO.yaml")  # build a YOLOv8n model from scratch

model.info()  # display model information

model.train(data="/root/ultralytics/cfg/VisDrone.yaml",
            epochs=100,
            name="SL-YOLO",
           imgsz=640,
           batch=16,
            device=0)  # train the model

