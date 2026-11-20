"""Transformer + quantization tests on a TINY randomly initialised BERT (no download).

Skipped automatically if torch / transformers are not installed.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import BertConfig, BertForSequenceClassification, BertTokenizer  # noqa: E402

from src.quantize import load_quantized, quantize_model, save_quantized  # noqa: E402
from src.transformer_model import (BertClassifier, class_weights,  # noqa: E402
                                   write_moderation_config)

WORDS = ["you", "are", "an", "idiot", "good", "game", "well", "played", "stupid", "hello"]


@pytest.fixture
def tiny(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]) + "\n")
    tok = BertTokenizer(vocab_file=str(vocab), do_lower_case=True)
    cfg = BertConfig(vocab_size=len(WORDS) + 5, hidden_size=32, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=64, max_position_embeddings=64,
                     num_labels=2)
    torch.manual_seed(0)
    model = BertForSequenceClassification(cfg).eval()
    return model, tok


def test_class_weights_balanced():
    w = class_weights([0] * 90 + [1] * 10)
    assert w.shape == (2,)
    assert torch.allclose(w, torch.tensor([100 / 180, 100 / 20]))


def test_wrapper_single_and_batch_agree(tiny):
    model, tok = tiny
    clf = BertClassifier(model, tok, threshold=0.5, max_length=32)
    texts = ["you are an idiot", "good game well played", "", "hello"]
    probs = clf.predict_proba(texts, batch_size=3)     # padding + length sorting
    assert probs.shape == (4,) and np.all((probs >= 0) & (probs <= 1))
    for t, p in zip(texts, probs):
        label, p1 = clf.predict_one(t)
        assert abs(p - p1) < 1e-4
        assert label == int(p1 >= 0.5)


def test_fp32_save_load_roundtrip(tiny, tmp_path):
    model, tok = tiny
    d = tmp_path / "fp32"
    model.save_pretrained(d)
    tok.save_pretrained(d)
    write_moderation_config(d, 0.42, 32, True)
    clf = BertClassifier.load(d)
    assert clf.threshold == pytest.approx(0.42)
    ref = BertClassifier(model, tok, 0.42, 32).predict_proba(["you idiot"])
    assert np.allclose(clf.predict_proba(["you idiot"]), ref, atol=1e-5)


def test_quantize_replaces_linear_layers(tiny):
    model, _ = tiny
    q = quantize_model(model)
    qlinear = torch.ao.nn.quantized.dynamic.Linear
    assert isinstance(q.classifier, qlinear)
    assert isinstance(q.bert.encoder.layer[0].attention.self.query, qlinear)
    assert not any(type(m) is torch.nn.Linear for m in q.modules())


def test_quantized_save_load_roundtrip(tiny, tmp_path):
    model, tok = tiny
    q = quantize_model(model)
    d = save_quantized(q, tok, model.config, tmp_path / "int8", 0.3, 32, True)
    assert (d / "quantized_state_dict.pt").exists()
    clf = load_quantized(d)
    assert clf.threshold == pytest.approx(0.3)
    texts = ["you are an idiot", "good game"]
    ref = BertClassifier(q, tok, 0.3, 32).predict_proba(texts)
    assert np.allclose(clf.predict_proba(texts), ref, atol=1e-5)
    label, p = clf.predict_one("stupid")
    assert label in (0, 1) and 0 <= p <= 1
