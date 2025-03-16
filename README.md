### Metrics:

- Speed: 0.8ms preprocess, 2.7ms inference, 0.0ms loss, 4.2ms postprocess per image
- 11-12 hours for transfer learning on two datasets, finetuning a pretrained COCO YOLO11n:
    - Charles Tang (CT) [Cyclist Detection Dataset](https://universe.roboflow.com/bicycle-detection/bike-detect-ct/dataset/5)
    - Cyclist Orientation [CIMAT Dataset](https://gitlab.com/MaryChelo/cimat-cyclist)

<br>

- mAP50-95  0.846183
- mAP50     0.985844
- mAP75     0.945719

<br>

Pretrained models: 
- [TensorRT](model/yolo/TrainedCTCIMATModels/CTCIMAT.engine)
- [ONNX](model/yolo/TrainedCTCIMATModels/CTCIMAT.onnx)
- [PyTorch](model/yolo/TrainedCTCIMATModels/CTCIMAT.pt)
