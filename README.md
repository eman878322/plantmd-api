# PlantMD AI — Professional MVP Backend

PlantMD is a production-oriented starter backend for a plant-health decision-support product inspired by the clinical workflow: **observe → classify → quantify severity → stage → retrieve evidence → generate a structured report → monitor progression**.

## What is included
- Pretrained disease-image classifier adapter (Hugging Face MobileNetV2 / PlantVillage)
- Image diagnosis endpoint: `POST /diagnose-image`
- Structured analysis endpoint: `POST /analyze`
- Embedding-based RAG with Sentence Transformers + FAISS
- Wheat, maize, and onion starter knowledge base
- Severity/staging engine
- Transparent prognosis scaffold
- Docker + docker-compose
- Tests, environment config, API schemas

## Important scientific boundary
This package is a **software MVP**, not a validated agricultural diagnostic device. The pretrained classifier is benchmarked on PlantVillage-style data, not necessarily on field images from Qena/Egypt. Severity currently uses a clearly labeled visual proxy. The progression forecast is a heuristic scenario. For the final graduation/project deployment, collect local field images and longitudinal observations, then fine-tune the classifier and train/validate a segmentation + progression model.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env  # Linux/macOS
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Image endpoint
`POST /diagnose-image` with multipart form field `file` and optional `crop`.

The first request downloads the configured Hugging Face model, so internet access is required on first startup unless the model cache is preloaded.

## Next production steps
1. Replace `data/plant_knowledge.json` placeholders with verified Egyptian agricultural references.
2. Fine-tune the classifier on Egyptian field images for wheat/maize/onion.
3. Train a disease segmentation model on masks (PlantSeg is a useful research starting point).
4. Collect repeated images per plant/plot and train a validated progression model.
5. Add authentication, rate limiting, object storage, database persistence, monitoring, and model versioning.
