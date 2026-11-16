import numpy as np


def test_dimension_is_consistent_for_documents_and_queries(embedder):
    docs = embedder.embed_documents(["Banks shall verify customer identity.", "Interest rates on deposits."])
    query = embedder.embed_query("How do banks verify identity?")
    assert docs.shape == (2, embedder.dimension) == (2, 384)
    assert query.shape == (embedder.dimension,)
    assert docs.dtype == np.float32 and query.dtype == np.float32


def test_vectors_are_l2_normalised(embedder):
    vecs = embedder.embed_documents(["one", "two words", "a much longer sentence about compliance"])
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(embedder.embed_query("q")), 1.0, atol=1e-5)


def test_semantically_related_text_scores_higher(embedder):
    q = embedder.embed_query("Within how many days must complaints be resolved?")
    docs = embedder.embed_documents([
        "A bank shall resolve every complaint within 30 days of receipt.",
        "The IT Strategy Committee shall meet every quarter.",
    ])
    sims = docs @ q
    assert sims[0] > sims[1] + 0.1


def test_embedding_is_deterministic(embedder):
    a = embedder.embed_documents(["same text"])
    b = embedder.embed_documents(["same text"])
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_empty_input_returns_empty_matrix(embedder):
    assert embedder.embed_documents([]).shape == (0, embedder.dimension)
