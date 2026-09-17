import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv('APP_NAME', 'PlantMD AI')
    environment: str = os.getenv('ENVIRONMENT', 'development')
    model_id: str = os.getenv('MODEL_ID', 'linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification')
    embedding_model: str = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    knowledge_path: str = os.getenv('KNOWLEDGE_PATH', 'data/plant_knowledge.json')
    top_k: int = int(os.getenv('TOP_K', '5'))
    confidence_threshold: float = float(os.getenv('CONFIDENCE_THRESHOLD', '0.55'))

settings = Settings()
