# Result tables and figures

One folder per experiment group. `verify/verify_claims.py` reads these folders.

| Folder | Content |
|---|---|
| `benchmark_figures/` | cross-model comparison figures and tables |
| `1_end_to_end_models/` | fine-tuned ResNet-50, ResNet-18 and EfficientNet-B3 at 224 and 448 px |
| `2_pretrained_vision_models/` | frozen-backbone linear probes: DINOv2, DINOv3, Cell-DINO, CLIP, BiomedCLIP |
| `3_lora_finetune/` | DINOv2-Small, frozen and LoRA-adapted |
| `4_explainability/` | `random_65_targets/` (targets and tables), `xai_vit/`, `xai_cnn/`, `pilot_archive/` |
| `5_brightfield_vs_fluorescence/` | brightfield-only and fluorescence models on the same 102 plates and related checks |
