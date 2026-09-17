from app.config import settings
from app.rag import KnowledgeRAG

if __name__ == '__main__':
    r=KnowledgeRAG(settings.knowledge_path, settings.embedding_model)
    r._ensure_index()
    print(f'Indexed {len(r.docs)} knowledge records with {settings.embedding_model}')
