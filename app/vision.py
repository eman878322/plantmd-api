from typing import Any, Dict, List
from PIL import Image

class VisionClassifier:
    """Lazy-loading adapter for a pretrained Hugging Face image classifier."""
    def __init__(self, model_id: str, threshold: float = 0.55):
        self.model_id = model_id
        self.threshold = threshold
        self._pipe = None
        self.error = None

    def _load(self):
        if self._pipe is not None or self.error is not None:
            return
        try:
            from transformers import pipeline
            self._pipe = pipeline('image-classification', model=self.model_id)
        except Exception as e:
            self.error = str(e)

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        self._load()
        if self._pipe is None:
            return {'available': False, 'error': self.error, 'predictions': []}
        preds: List[Dict[str, Any]] = self._pipe(image)
        best = preds[0] if preds else {'label':'unknown','score':0.0}
        return {
            'available': True,
            'predictions': preds[:5],
            'best': {'label': best['label'], 'confidence': float(best['score'])},
            'accepted': float(best['score']) >= self.threshold
        }
