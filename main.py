import uuid
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from .config import settings
from .schemas import AnalyzeRequest
from .rag import KnowledgeRAG, severity_stage, build_prognosis
from .vision import VisionClassifier
from .severity import SeverityEstimator

app = FastAPI(title=settings.app_name, version='2.0.0', description='PlantMD: AI plant health decision-support API with vision + severity + RAG.')
rag = KnowledgeRAG(settings.knowledge_path, settings.embedding_model)
vision = VisionClassifier(settings.model_id, settings.confidence_threshold)
severity = SeverityEstimator()

@app.get('/')
def root(): return {'name': settings.app_name, 'version':'2.0.0','status':'running'}

@app.get('/health')
def health(): return {'status':'ok','vision_model':settings.model_id,'rag':'configured'}

@app.post('/analyze')
def analyze(req: AnalyzeRequest):
    request_id=str(uuid.uuid4())
    sev=req.severity_percent if req.severity_percent is not None else 0.0
    stage=req.stage or severity_stage(sev)
    hits=rag.retrieve(req.crop, req.disease, settings.top_k)
    if not hits: raise HTTPException(404,'No knowledge found for this crop/disease')
    best=hits[0]['document']
    return {
      'request_id':request_id,
      'report':{
        'crop':req.crop,'disease':best.get('disease',req.disease),'disease_type':best.get('type'),
        'confidence':None,'severity_percent':sev,'stage':stage,
        'symptoms':best.get('symptoms'),'risk':best.get('risk'),
        'prognosis':build_prognosis(sev, req.environment.model_dump() if req.environment else None),
        'management':best.get('management'),'prevention':best.get('prevention')
      },
      'retrieved_sources':[{'source':h['document'].get('source'),'score':round(h['score'],4)} for h in hits],
      'warnings':['Treatment decisions must be checked against current local agricultural guidance and product labels.','Prognosis is a heuristic scenario until longitudinal field data are used to train and validate a progression model.']
    }

@app.post('/diagnose-image')
async def diagnose_image(crop: Optional[str]=None, file: UploadFile=File(...)):
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(415,'Please upload an image file.')
    raw=await file.read()
    try: img=Image.open(__import__('io').BytesIO(raw)).convert('RGB')
    except Exception: raise HTTPException(400,'Invalid image.')
    pred=vision.predict(img)
    sev=severity.estimate(img)
    label=pred.get('best',{}).get('label')
    hits=rag.retrieve(crop or 'unknown', label, settings.top_k)
    best=hits[0]['document'] if hits else {}
    severity_pct=sev['severity_percent_proxy']
    return {'request_id':str(uuid.uuid4()),'vision':pred,'severity':sev,'report':{
      'crop':crop,'disease':best.get('disease') or label,'severity_percent':severity_pct,
      'stage':severity_stage(severity_pct),'management':best.get('management'),
      'prognosis':build_prognosis(severity_pct,None)
    },'sources':[{'source':h['document'].get('source'),'score':round(h['score'],4)} for h in hits],
    'warnings':['Severity is currently a non-validated visual proxy. Replace with a trained segmentation model before using it for agronomic decisions.']}
