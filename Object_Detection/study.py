from ultralytics import YOLO
from config import Train_Dataset_Path, Test_Dataset_Path

train_dataset_path = Train_Dataset_Path

test_dataset_path = Test_Dataset_Path


# 모델 불러오기
model = YOLO("yolo11n.yaml")

# 모델 학습, 검증, 예측
model = YOLO("yolo11n.pt")

# 모델 학습
results = model.train(data=train_dataset_path, epochs=3)

# 모델 검증
results = model.val()

# 모델 예측
results = model(test_dataset_path)

#모델 추출
success = model.export(format="onnx")