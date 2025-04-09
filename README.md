## Read the full article in my website: [kadenseto.vercel.app](https://kadenseto.vercel.app)

And navigate to the Aegis article through: 
- Projects --> Aegis

### Metrics:

- Speed: 0.8ms preprocess, 2.7ms inference, 0.0ms loss, 4.2ms postprocess per image
- 11-12 hours for transfer learning on two datasets, finetuning a pretrained COCO YOLO11n:
    - Charles Tang (CT) [Cyclist Detection Dataset](https://universe.roboflow.com/bicycle-detection/bike-detect-ct/dataset/5)
    - Cyclist Orientation [CIMAT Dataset](https://gitlab.com/MaryChelo/cimat-cyclist)
- mAP50-95  0.846183
- mAP50     0.985844
- mAP75     0.945719

Pretrained models: 
- [TensorRT](https://raw.githubusercontent.com/kseto06/Aegis/main/model/yolo/TrainedCTCIMATModels/CTCIMAT.engine)
- [ONNX](https://raw.githubusercontent.com/kseto06/Aegis/main/model/yolo/TrainedCTCIMATModels/CTCIMAT.onnx)
- [PyTorch](https://raw.githubusercontent.com/kseto06/Aegis/main/model/yolo/TrainedCTCIMATModels/CTCIMAT.pt)

---

### Tests & Demos:

<div style="display: flex; justify-content: space-around;">
  <img src="docs/aegis-compressed.gif" width="600">
  <img src="docs/test1.gif" width="200">
  <img src="docs/test2.gif" width=200>
  <img src="docs/test3.gif" width=200>
  <img src="docs/test4.gif" width="200">
  <img src="docs/test5.gif" width="200">
</div>

Watch the prototype demo here: [Aegis Demo](https://www.youtube.com/watch?v=klIUhFy4po0)
