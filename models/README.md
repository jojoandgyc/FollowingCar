# RKNN model files

The default follow runtime uses these included RK3588 models:

- `yolo11n_int8_person_val2017.rknn`: person detector.
- `osnet_x0_25_msmt17_b1.rknn`: person ReID embedding model.

The main follow runtime opens the camera and runs YOLO11 + ReID + tracking.
The vision pipeline can also process frames supplied by another caller.

Source ONNX/PT files are downloaded to `models/source/` and ignored by Git:

```powershell
.\scripts\download_yolo11_onnx.ps1
.\scripts\download_osnet_onnx.ps1
.\scripts\download_osnet_torchreid_weights.ps1 -Variant x0_50
.\scripts\download_fastreid_weights.ps1 -Model market_bot_R50
```

YOLO11n uses the URL from Rockchip `rknn_model_zoo/examples/yolo11`; the same
example is present in the local `rknn_model_zoo-v2.3.2-2025-04-09.tgz` package.
To experiment with a YOLO11s model, download/export an ONNX from Ultralytics
weights and convert it with RKNN-Toolkit2. This does not replace the default
YOLO11n model.

```powershell
.\scripts\download_yolo11_pt.ps1 -Variant s
.\scripts\export_yolo11_onnx.ps1 -Variant s
.\scripts\convert_yolo11_rknn.ps1 -Variant s -DType fp
```

The OSNet download and conversion scripts below are optional experiments; the
checked-in OSNet x0_25 RKNN file is the default ReID model for this branch.

To test the public OSNet x0_25 MSMT17 ONNX fallback:

```powershell
.\scripts\download_osnet_onnx.ps1
.\scripts\convert_osnet_rknn.ps1
```

That ONNX export is fixed to batch 16, so the conversion script first patches a
batch-1 copy under `models/source/`.  Run it with RGB 128x256 ReID crops:

```bash
python3 tools/rknn_video_smoke.py test.mp4 \
  --reid-model models/osnet_x0_25_msmt17_b1.rknn \
  --reid-input-width 128 \
  --reid-input-height 256 \
  --reid-input-format RGB
```

To test the larger Torchreid OSNet x0_5 MSMT17-combineall model (`x0_50` is an
alias for the same architecture):

```powershell
.\scripts\download_osnet_torchreid_weights.ps1 -Variant x0_50
.\scripts\export_osnet_onnx.ps1 -Variant x0_50
.\scripts\convert_reid_rknn.ps1 `
  -SourceOnnx models\source\osnet_x0_5_msmt17_combineall_b1.onnx `
  -Output models\osnet_x0_5_msmt17_combineall_b1.rknn
```

Run it with RGB ImageNet-normalized 128x256 crops:

```bash
python3 tools/rknn_video_smoke.py test.mp4 \
  --reid-model models/osnet_x0_5_msmt17_combineall_b1.rknn \
  --reid-input-width 128 \
  --reid-input-height 256 \
  --reid-input-format RGB
```

To test FastReID Market1501 BoT R50:

```powershell
.\scripts\download_fastreid_weights.ps1 -Model market_bot_R50
.\scripts\export_fastreid_onnx.ps1 -Model market_bot_R50
.\scripts\convert_reid_rknn.ps1 `
  -SourceOnnx models\source\fastreid_market_bot_R50_b1.onnx `
  -Output models\fastreid_market_bot_R50_b1.rknn
```

FastReID exports the RGB mean/std preprocessing into the ONNX graph, so pass raw
float32 RGB crops at runtime:

```bash
python3 tools/rknn_video_smoke.py test.mp4 \
  --reid-model models/fastreid_market_bot_R50_b1.rknn \
  --reid-input-width 128 \
  --reid-input-height 256 \
  --reid-input-format RGB \
  --reid-normalize none
```

`tools/rknn_video_smoke.py` writes `timing_ms` per frame and reports
`avg_timing_ms`/`max_timing_ms` in the summary, including decode, YOLO,
ReID, tracker, and total frame time.
