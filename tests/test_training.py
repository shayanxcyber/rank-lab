from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as functional

from rank_lab.data import RetrievalData
from rank_lab.training import TrainingConfig, contrastive_loss, train, weights_fingerprint


class ToyEncoder(nn.Module):
    def __init__(self, texts):
        super().__init__()
        self.lookup = {text: index for index, text in enumerate(sorted(texts))}
        self.embedding = nn.Embedding(len(texts), 8)
        self.dropout = nn.Dropout(0.2)

    def forward(self, texts):
        indices = torch.tensor([self.lookup[text] for text in texts])
        return functional.normalize(self.dropout(self.embedding(indices)), dim=1)


@pytest.fixture
def experiment():
    documents = {f"d{index}": f"document {index}" for index in range(5)}
    queries = {f"q{index}": f"query {index}" for index in range(5)} | {"test": "new query"}
    data = RetrievalData(
        documents, queries, {f"q{i}": {f"d{i}": 1} for i in range(5)}, {"test": {"d0": 1}}
    )
    return data, data.split()


def test_resumed_training_matches_uninterrupted_weights_and_losses(tmp_path, experiment):
    data, split = experiment
    texts = set(data.documents.values()) | set(data.queries.values())
    config = TrainingConfig(
        epochs=3, batch_size=2, device="cpu", mixed_precision=False, learning_rate=0.01
    )
    torch.manual_seed(7)
    uninterrupted = ToyEncoder(texts)
    full_result = train(uninterrupted, data, split, config, tmp_path / "full")

    torch.manual_seed(7)
    interrupted = ToyEncoder(texts)
    partial = train(interrupted, data, split, config, tmp_path / "resumed", stop_after_steps=2)
    assert partial["status"] == "paused"

    torch.manual_seed(999)
    resumed = ToyEncoder(texts)
    resumed_result = train(resumed, data, split, config, tmp_path / "resumed", resume=True)

    assert full_result["status"] == resumed_result["status"] == "completed"
    assert full_result["history"] == resumed_result["history"]
    assert weights_fingerprint(uninterrupted) == weights_fingerprint(resumed)
    assert resumed_result["weights"] != resumed_result["initial_weights"]


def test_resume_rejects_changed_training_contract(tmp_path, experiment):
    data, split = experiment
    texts = set(data.documents.values()) | set(data.queries.values())
    config = TrainingConfig(epochs=3, batch_size=2, device="cpu", mixed_precision=False)
    train(ToyEncoder(texts), data, split, config, tmp_path, stop_after_steps=1)
    previous = (tmp_path / "training-state.pt").read_bytes()

    with pytest.raises(ValueError, match="contract does not match"):
        train(
            ToyEncoder(texts), data, split, replace(config, temperature=0.1), tmp_path, resume=True
        )

    assert (tmp_path / "training-state.pt").read_bytes() == previous


def test_other_known_positives_are_not_used_as_negatives():
    queries = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    documents = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    pairs = [("q1", "d1"), ("q2", "d2")]
    judgments = {"q1": {"d1": 1, "d2": 1}, "q2": {"d2": 1}}

    loss = contrastive_loss(queries, documents, pairs, judgments, temperature=0.1)

    assert float(loss) == pytest.approx(float(functional.softplus(torch.tensor(10.0))) / 2)
