"""
Unit tests for RAG Medical Knowledge Retriever.
"""

from pathlib import Path
import pytest

from src.rag.knowledge_retriever import DRKnowledgeRetriever


class TestKnowledgeRetriever:
    """Tests for document chunking, indexing, and semantic retrieval."""

    def test_chunk_text(self):
        retriever = DRKnowledgeRetriever(chunk_size=50, chunk_overlap=10)
        sample_doc = "This is a paragraph about diabetic retinopathy.\n\nIt describes mild nonproliferative changes with microaneurysms."
        chunks = retriever.chunk_text(sample_doc, source="test.md")
        assert len(chunks) >= 1
        assert "source" in chunks[0]
        assert "text" in chunks[0]

    def test_indexing_and_retrieval(self, tmp_path: Path):
        # Create small test guideline file
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir(parents=True, exist_ok=True)
        vec_dir = tmp_path / "vec"
        vec_dir.mkdir(parents=True, exist_ok=True)

        guide_file = kb_dir / "test_guideline.md"
        guide_file.write_text(
            "American Academy of Ophthalmology recommends annual dilated eye examinations for patients with no diabetic retinopathy.\n\n"
            "Patients with severe nonproliferative diabetic retinopathy must be referred urgently within 2 to 4 weeks.",
            encoding="utf-8",
        )

        retriever = DRKnowledgeRetriever(knowledge_base_dir=kb_dir, vector_db_dir=vec_dir)
        count = retriever.build_index(force_rebuild=True)
        assert count > 0

        # Retrieve
        results = retriever.retrieve("urgent referral severe nonproliferative", top_k=1)
        assert len(results) > 0
        assert "severe" in results[0]["text"].lower() or "referral" in results[0]["text"].lower()
