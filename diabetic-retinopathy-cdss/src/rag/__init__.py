"""
Retrieval-Augmented Generation (RAG) module for Diabetic Retinopathy.
Extracts guidelines, generates embeddings via SentenceTransformers,
builds FAISS vector index, and provides semantic clinical evidence retrieval.
"""

from .knowledge_retriever import DRKnowledgeRetriever

__all__ = ["DRKnowledgeRetriever"]
