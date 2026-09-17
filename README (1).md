# Model assets

## Disease classifier
The API defaults to a pretrained Hugging Face MobileNetV2 classifier:
`linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`

It is downloaded automatically by `transformers` on first use. The model card reports 38 PlantVillage classes and 0.9541 evaluation accuracy on its evaluation set. Treat that as benchmark information, not field accuracy.

## Severity model
Do NOT claim the included color-proxy estimator is trained. It is a safe development fallback only.

For the final project, train a segmentation model using disease masks (for example PlantSeg) and place its exported weights here. Then replace `app/severity.py` with the trained inference adapter.

## Progression model
The current prognosis is a transparent heuristic scenario. A validated progression model requires repeated observations from the same plant/plot plus environment and crop-stage metadata.
