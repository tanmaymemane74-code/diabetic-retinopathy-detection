"""
Medical Knowledge Retriever using SentenceTransformers and FAISS.
Indexes clinical guideline documents (AAO, ADA, WHO) and performs
semantic similarity search to retrieve relevant evidence for DR severity grades.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.config import KNOWLEDGE_BASE_PATH, VECTOR_DB_PATH

logger = logging.getLogger("dr_cdss.rag")


class DRKnowledgeRetriever:
    """Manages document chunking, embedding generation, FAISS indexing, and retrieval."""

    def __init__(
        self,
        knowledge_base_dir: Optional[Path] = None,
        vector_db_dir: Optional[Path] = None,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 400,
        chunk_overlap: int = 50,
    ):
        self.kb_dir = Path(knowledge_base_dir or KNOWLEDGE_BASE_PATH)
        self.vector_dir = Path(vector_db_dir or VECTOR_DB_PATH)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embedder = None
        self.index = None
        self.chunks: List[Dict[str, Any]] = []

    def _get_embedder(self):
        """Lazy load SentenceTransformer embedder with offline caching optimization."""
        if self.embedder is None:
            import os
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SentenceTransformer model: %s", self.model_name)
            try:
                self.embedder = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                self.embedder = SentenceTransformer(self.model_name)
        return self.embedder

    def chunk_text(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Split text into overlapping semantic passages."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            words = para.split()
            if current_len + len(words) > self.chunk_size and current_chunk:
                chunk_str = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_str,
                    "source": source,
                    "word_count": len(current_chunk),
                })
                # Overlap
                current_chunk = current_chunk[-self.chunk_overlap :] if len(current_chunk) > self.chunk_overlap else []
                current_len = len(current_chunk)

            current_chunk.extend(words)
            current_len += len(words)

        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "source": source,
                "word_count": len(current_chunk),
            })
        return chunks

    def build_index(self, force_rebuild: bool = False):
        """Index all markdown, txt, and pdf documents in knowledge_base/."""
        import faiss

        index_file = self.vector_dir / "faiss_index.bin"
        metadata_file = self.vector_dir / "chunks_metadata.json"

        if not force_rebuild and index_file.exists() and metadata_file.exists():
            logger.info("Loading existing FAISS index from %s", index_file)
            self.index = faiss.read_index(str(index_file))
            with open(metadata_file, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            return len(self.chunks)

        logger.info("Building FAISS index from knowledge base at: %s", self.kb_dir)
        all_chunks = []

        # Scan directory
        for file_path in self.kb_dir.glob("*.*"):
            if file_path.suffix.lower() in [".md", ".txt"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                file_chunks = self.chunk_text(content, source=file_path.name)
                all_chunks.extend(file_chunks)

        if not all_chunks:
            logger.warning("No knowledge base documents found in %s.", self.kb_dir)
            return 0

        self.chunks = all_chunks
        texts = [c["text"] for c in all_chunks]

        # Generate Embeddings
        embedder = self._get_embedder()
        embeddings = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Create FAISS L2/Cosine index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner Product on normalized vectors = Cosine similarity
        self.index.add(embeddings)

        # Persist
        faiss.write_index(self.index, str(index_file))
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)

        logger.info("Successfully indexed %d chunks into FAISS vector database.", len(self.chunks))
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-K most semantically relevant clinical evidence passages."""
        if self.index is None or not self.chunks:
            self.build_index()

        if self.index is None or len(self.chunks) == 0:
            return []

        embedder = self._get_embedder()
        query_vec = embedder.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype=np.float32)

        k = min(top_k, len(self.chunks))
        distances, indices = self.index.search(query_vec, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx].copy()
            chunk["similarity_score"] = round(float(distances[0][i]), 4)
            results.append(chunk)

        return results

    def retrieve_for_grade(self, predicted_grade: int, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve guidelines specific to a given Diabetic Retinopathy grade."""
        grade_queries = {
            0: "No apparent diabetic retinopathy annual dilated screening examination guidelines glycemic control",
            1: "Mild nonproliferative diabetic retinopathy microaneurysms follow up examination schedule",
            2: "Moderate nonproliferative diabetic retinopathy referral specialist macular edema hemorrhages",
            3: "Severe nonproliferative diabetic retinopathy 4-2-1 rule urgent retina specialist referral",
            4: "Proliferative diabetic retinopathy neovascularization vitreous hemorrhage immediate referral vitrectomy laser",
        }
        query = grade_queries.get(predicted_grade, f"Diabetic retinopathy Grade {predicted_grade} clinical guidelines")
        return self.retrieve(query, top_k=top_k)
