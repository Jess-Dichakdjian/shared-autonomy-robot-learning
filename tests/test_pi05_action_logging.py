import numpy as np

from thesis_vla.pi05 import Pi05PolicyAdapter


class ChunkPolicy:
    def predict_action_chunk(self, batch):
        result = np.zeros((1, 3, 7), dtype=np.float32)
        result[:, 1, 0] = 0.75
        return result
    def reset(self):
        pass


def test_proposal_preserves_raw_prediction_and_postprocessed_execution():
    policy = Pi05PolicyAdapter(
        ChunkPolicy(), lambda observation, task: {},
        postprocess_action=lambda chunk: np.asarray(chunk) * 2,
        task="test", execute_steps=2,
    )
    first = policy.act({})
    second = policy.act({})
    assert first.predicted_action[0] == 0
    assert first.executed_action[0] == 0
    assert second.predicted_action[0] == 0.75
    assert second.executed_action[0] == 1.0

def test_pretrained_loader_refuses_missing_local_checkpoint_without_downloading():
    import pytest

    with pytest.raises(FileNotFoundError, match="local checkpoint does not exist"):
        Pi05PolicyAdapter.from_pretrained(
            "/tmp/no-such-pi05-checkpoint", task="fixture", local_files_only=True
        )
