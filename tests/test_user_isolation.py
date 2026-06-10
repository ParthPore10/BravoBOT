import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma

import app.document_ingestion as document_ingestion
import app.job_store as job_store
import app.retriever as retriever


class FakeVectorstore:
    def __init__(self):
        self.filters = []

    def similarity_search_with_score(self, query, k, filter):
        self.filters.append(filter)
        user_id = filter["user_id"]
        document = Document(
            page_content=f"content for {user_id}",
            metadata={
                "chunk_id": f"{user_id}-chunk",
                "user_id": user_id,
            },
        )
        return [(document, 0.1)]


class TestEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    @staticmethod
    def _embed(text):
        normalized = text.lower()
        return [
            1.0 if "alpha" in normalized else 0.0,
            1.0 if "beta" in normalized else 0.0,
        ]


class FakeResult:
    def __init__(self, row):
        self.row = row

    def mappings(self):
        return self

    def first(self):
        return self.row


class FakeSession:
    def __init__(self, expected_user_id):
        self.expected_user_id = expected_user_id

    def execute(self, statement, parameters):
        if parameters["user_id"] != self.expected_user_id:
            return FakeResult(None)

        return FakeResult({
            "job_id": parameters["job_id"],
            "user_id": parameters["user_id"],
            "status": "completed",
            "result_json": None,
            "error": None,
            "created_at": FakeTimestamp(),
            "updated_at": FakeTimestamp(),
        })

    def close(self):
        pass


class FakeTimestamp:
    def isoformat(self):
        return "2026-06-10T00:00:00"


class UserIsolationTests(unittest.TestCase):
    def setUp(self):
        retriever.BM25_CACHE.clear()

    def test_same_filename_is_stored_separately_per_user(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                document_ingestion,
                "RAW_UPLOAD_DIR",
                Path(temp_dir),
            ):
                first = document_ingestion.save_upload(
                    "../paper.txt",
                    b"user-a content",
                    "user-a",
                )
                second = document_ingestion.save_upload(
                    "../paper.txt",
                    b"user-b content",
                    "user-b",
                )

                self.assertEqual(first.name, "paper.txt")
                self.assertEqual(second.name, "paper.txt")
                self.assertNotEqual(first.parent, second.parent)
                self.assertEqual(first.read_bytes(), b"user-a content")
                self.assertEqual(second.read_bytes(), b"user-b content")

    def test_bm25_returns_only_the_requesting_users_chunks(self):
        chunks = [
            {
                "chunk_id": "a-1",
                "user_id": "user-a",
                "text": "alpha private document",
            },
            {
                "chunk_id": "b-1",
                "user_id": "user-b",
                "text": "beta private document",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            chunks_path = Path(temp_dir) / "chunks.json"
            chunks_path.write_text(json.dumps(chunks), encoding="utf-8")

            with patch.object(retriever, "CHUNKS_PATH", chunks_path):
                user_a_results = retriever.search_bm25(
                    "private document",
                    "user-a",
                )
                user_b_results = retriever.search_bm25(
                    "private document",
                    "user-b",
                )

        self.assertEqual(
            [item["chunk_id"] for item in user_a_results],
            ["a-1"],
        )
        self.assertEqual(
            [item["chunk_id"] for item in user_b_results],
            ["b-1"],
        )

    def test_legacy_chunks_receive_an_isolated_owner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            chunks_path = Path(temp_dir) / "chunks.json"
            chunks_path.write_text(
                json.dumps([{
                    "chunk_id": "legacy-1",
                    "text": "legacy content",
                }]),
                encoding="utf-8",
            )

            with patch.object(
                document_ingestion,
                "CHUNKS_PATH",
                chunks_path,
            ):
                chunks = document_ingestion.load_existing_chunks()

        self.assertEqual(chunks[0]["user_id"], "legacy-user")

    def test_dense_search_passes_user_filter_to_chroma(self):
        vectorstore = FakeVectorstore()

        with patch.object(
            retriever,
            "load_vectorstore",
            return_value=vectorstore,
        ):
            results = retriever.dense_search(
                "question",
                "user-a",
                top_k=3,
            )

        self.assertEqual(vectorstore.filters, [{"user_id": "user-a"}])
        self.assertEqual(results[0]["chunk_id"], "user-a-chunk")

    def test_chroma_filter_prevents_cross_user_retrieval(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            vectorstore = Chroma.from_documents(
                documents=[
                    Document(
                        page_content="alpha private material",
                        metadata={
                            "chunk_id": "a-1",
                            "user_id": "user-a",
                        },
                    ),
                    Document(
                        page_content="beta private material",
                        metadata={
                            "chunk_id": "b-1",
                            "user_id": "user-b",
                        },
                    ),
                ],
                embedding=TestEmbeddings(),
                persist_directory=temp_dir,
                collection_name="isolation_test",
            )

            with patch.object(
                retriever,
                "load_vectorstore",
                return_value=vectorstore,
            ):
                user_a_results = retriever.dense_search(
                    "alpha",
                    "user-a",
                    top_k=5,
                )
                user_b_results = retriever.dense_search(
                    "alpha",
                    "user-b",
                    top_k=5,
                )

        self.assertEqual(
            [item["chunk_id"] for item in user_a_results],
            ["a-1"],
        )
        self.assertEqual(
            [item["chunk_id"] for item in user_b_results],
            ["b-1"],
        )

    def test_job_lookup_requires_matching_user(self):
        with patch.object(
            job_store,
            "SessionLocal",
            return_value=FakeSession("user-a"),
        ):
            owned_job = job_store.get_job_record("job-1", "user-a")
            foreign_job = job_store.get_job_record("job-1", "user-b")

        self.assertEqual(owned_job["user_id"], "user-a")
        self.assertIsNone(foreign_job)


if __name__ == "__main__":
    unittest.main()
