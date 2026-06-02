# Diabetic Retinopathy Detection

## Dataset

- DDR Dataset: https://www.kaggle.com/datasets/mariaherrerot/ddrdataset
- DDR Preprocess Augmented: https://www.kaggle.com/datasets/xuanductran/ddr-preprocess-augmented

## Model Structure

This project uses the model structure from `structure_model_v5_for_ddr.ipynb`. The system is a hierarchical ensemble for diabetic retinopathy grading with five classes:

| Grade | Label |
|---|---|
| 0 | No DR |
| 1 | Mild DR |
| 2 | Moderate DR |
| 3 | Severe DR |
| 4 | Proliferative DR |

## Input Data

The notebook expects preprocessed DDR images arranged by class folder:

```text
train_preprocess/
  0/
  1/
  2/
  3/
  4/
val_preprocess/
  0/
  1/
  2/
  3/
  4/
test_preprocess/
  0/
  1/
  2/
  3/
  4/
```

Images are loaded as RGB, resized to `300 x 300`, batched with `BATCH_SIZE = 8` for training, and passed through TensorFlow EfficientNet `preprocess_input`.

## Backbone And Head

All three core models use the same base architecture:

```text
Input image: 300 x 300 x 3
  -> EfficientNetB3 backbone, ImageNet weights, include_top=False
  -> Simple Channel Attention block
  -> GlobalAveragePooling2D
  -> BatchNormalization
  -> Dropout(0.4)
  -> Dense(1024, relu)
  -> BatchNormalization
  -> Dropout(0.3)
  -> Task-specific sigmoid output
```

The Simple Channel Attention block applies global average pooling, a `1 x 1` reduction convolution, a sigmoid gate convolution, and channel-wise multiplication with the backbone feature map.

## Three-Model Hierarchy

### Model 1: Low vs High

`EffB3_M1_LowHigh` separates lower grades from higher grades.

```text
Classes: 0, 1, 2, 3, 4
Target: 0 for grades {0, 1}; 1 for grades {2, 3, 4}
Output: Dense(1, sigmoid)
Saved prefix: effb3_low_high
```

### Model 2: Grade 0 vs Grade 1

`EffB3_M2_0vs1` is used for images routed to the low-severity branch.

```text
Classes: 0, 1
Target: 0 or 1
Output: Dense(1, sigmoid)
Saved prefix: effb3_0_vs_1
```

### Model 3: Ordinal Grades 2, 3, 4

`EffB3_M3_234Ordinal` is used for images routed to the high-severity branch.

```text
Classes: 2, 3, 4
Ordinal target: [y >= 3, y >= 4]
Grade 2 -> [0, 0]
Grade 3 -> [1, 0]
Grade 4 -> [1, 1]
Output: Dense(2, sigmoid)
Saved prefix: effb3_234_ordinal2bit
```

## Prediction Flow

The main inference flow is:

```text
Input image
  -> Model 1: p_high
      if p_high < 0.5:
          -> Model 2 predicts grade 0 or 1
      else:
          -> Model 3 predicts grade 2, 3, or 4
```

For Model 3, the two ordinal sigmoid outputs are decoded as probabilities:

```text
p2 = 1 - p(y >= 3)
p3 = p(y >= 3) - p(y >= 4)
p4 = p(y >= 4)
prediction = argmax([p2, p3, p4]) + 2
```

The notebook also contains experimental unified evaluation code that compares the three-model hierarchy with an additional ordinal model, but the core DDR structure is the three EfficientNetB3 models above.

## Training Pipeline

Each model is trained with the same staged fine-tuning pipeline:

| Stage | Trainable Layers | Learning Rate |
|---|---|---|
| 1 | Classification head only, EfficientNetB3 frozen | `3e-4` |
| 2 | EfficientNet blocks 6-7 | `1e-4` |
| 3 | EfficientNet blocks 4-7 | `5e-5` |
| 4 | EfficientNet blocks 1-7 | `3e-5` |

Training settings:

- `HEAD_EPOCHS = 10`
- `FT_EPOCHS = 10` per fine-tuning stage
- `WEIGHT_DECAY = 1e-4`
- Binary cross-entropy for Model 1 and Model 2
- Multi-label binary cross-entropy for Model 3
- Metrics: binary accuracy and AUC for binary models, multi-label AUC for the ordinal model
- Callbacks: early stopping, reduce learning rate on plateau, and best checkpoint saving

## Output Models

The trained checkpoints are saved in `output_three_models_ddr/`:

```text
effb3_low_high_stage1.keras
effb3_low_high_stage2.keras
effb3_low_high_stage3.keras
effb3_low_high_stage4.keras

effb3_0_vs_1_stage1.keras
effb3_0_vs_1_stage2.keras
effb3_0_vs_1_stage3.keras
effb3_0_vs_1_stage4.keras

effb3_234_ordinal2bit_stage1.keras
effb3_234_ordinal2bit_stage2.keras
effb3_234_ordinal2bit_stage3.keras
effb3_234_ordinal2bit_stage4.keras
```

The stage 4 checkpoints are the final models used for evaluation and inference.
