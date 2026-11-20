"""Tests for Tier 2: Word2Vec trainer, encoding, forward pass, SGD training, save/load.

Skipped automatically if torch / gensim are not installed.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src import config as C  # noqa: E402
from src.neural_net import (AvgEmbeddingClassifier, NeuralModel, NNConfig,  # noqa: E402
                            W2VConfig, collate_bags, embedding_matrix, encode,
                            standardize_embeddings,
                            tokenize_corpus, train_and_evaluate, train_word2vec)
from tests.test_baseline import _toy  # noqa: E402

TINY_W2V = W2VConfig(vector_size=16, window=3, min_count=1, max_final_vocab=None,
                     epochs=5, workers=1)
TINY_NN = NNConfig(hidden=(16,), dropout=0.0, batch_size=32, eval_batch_size=64,
                   lr=0.1, max_epochs=25, patience=5, num_workers=0)


# ------------------------------------------------------------------ word2vec
def test_word2vec_trains_and_exports_matrix():
    pytest.importorskip("gensim")
    toks = tokenize_corpus(["you stupid idiot", "good game well played", "Y0u 1d10t!!"] * 20)
    kv = train_word2vec(toks, TINY_W2V)
    vocab, emb = embedding_matrix(kv)
    assert emb.shape == (len(vocab), 16)
    assert emb.dtype == np.float32
    assert "idiot" in vocab            # leetspeak "1d10t" normalised before training
    assert "1d10t" not in vocab
    assert np.isfinite(emb).all()


def test_standardize_embeddings():
    emb = np.random.RandomState(0).randn(50, 8).astype(np.float32) + 5.0
    z = standardize_embeddings(emb)
    assert np.allclose(z.mean(0), 0, atol=1e-5) and np.allclose(z.std(0), 1, atol=1e-3)
    assert z.dtype == np.float32


# ------------------------------------------------------------------ encoding
def test_encode_drops_oov_and_truncates():
    stoi = {"a": 0, "b": 1, "c": 2}
    assert encode(["a", "zzz", "b", "c"], stoi, max_len=10) == [0, 1, 2]
    assert encode(["a", "b", "c"], stoi, max_len=2) == [0, 1]
    assert encode(["zzz"], stoi, max_len=5) == []


def test_collate_offsets():
    flat, offs, y = collate_bags([([1, 2, 3], 1.0), ([], 0.0), ([4], 1.0)])
    assert flat.tolist() == [1, 2, 3, 4]
    assert offs.tolist() == [0, 3, 3]
    assert y.tolist() == [1.0, 0.0, 1.0]


# ------------------------------------------------------------------ forward pass
def test_forward_shape_and_mean_pooling():
    emb = np.arange(12, dtype=np.float32).reshape(4, 3)
    net = AvgEmbeddingClassifier(emb, hidden=(8,), dropout=0.0).eval()
    flat, offs, _ = collate_bags([([0, 1], 0.0), ([], 0.0), ([2, 3, 3], 0.0)])
    pooled = net.embedding(flat, offs)
    assert torch.allclose(pooled[0], torch.tensor(emb[[0, 1]].mean(0)))
    assert torch.allclose(pooled[1], torch.zeros(3))          # empty message -> zeros
    out = net(flat, offs)
    assert out.shape == (3,)
    assert torch.isfinite(out).all()


def test_embeddings_frozen_by_default():
    net = AvgEmbeddingClassifier(np.ones((5, 4), np.float32))
    assert not net.embedding.weight.requires_grad
    net_ft = AvgEmbeddingClassifier(np.ones((5, 4), np.float32), finetune_emb=True)
    assert net_ft.embedding.weight.requires_grad


def test_backprop_updates_mlp_not_frozen_embeddings():
    torch.manual_seed(0)
    emb = np.random.RandomState(0).randn(6, 4).astype(np.float32)
    net = AvgEmbeddingClassifier(emb, hidden=(8,), dropout=0.0)
    before_emb = net.embedding.weight.clone()
    before_w = net.mlp[1].weight.clone()
    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=0.5)
    flat, offs, y = collate_bags([([0, 1], 1.0), ([2, 3], 0.0)])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(net(flat, offs), y)
    loss.backward()
    opt.step()
    assert torch.equal(before_emb, net.embedding.weight)
    assert not torch.equal(before_w, net.mlp[1].weight)


# ------------------------------------------------------------------ end-to-end
def test_neural_net_end_to_end(tmp_path):
    pytest.importorskip("gensim")
    train, val, test = _toy(400, 0), _toy(100, 1), _toy(100, 2)
    path = tmp_path / "model.pt"
    res = train_and_evaluate(train, val, test, TINY_W2V, TINY_NN, save_path=path,
                             w2v_path=None, latency_runs=100, verbose=False)
    assert res["macro_f1"] > 0.8
    assert res["size_mb"] > 0
    assert res["latency_runs"] >= 100 and res["latency_mean_ms"] > 0
    assert res["latency_device"] == "cpu"
    assert 1 <= res["epochs_trained"] <= TINY_NN.max_epochs

    model = NeuralModel.load(path)
    label, p = model.predict_one("you stupid idiot")
    assert label == 1 and 0.0 <= p <= 1.0
    # single-message and batched paths agree
    assert abs(model.predict_proba(["you stupid idiot"])[0] - p) < 1e-5
    # all-OOV / empty input does not crash
    assert model.predict_one("")[1] >= 0.0
    assert set(np.unique(model.predict(test[C.TEXT_COL]))) <= {0, 1}
