"""Opt-in authenticated π0.5 processor test; uses mock inference, never loads weights."""
import os
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_PI05_PROCESSOR_INTEGRATION") != "1",
    reason="set RUN_PI05_PROCESSOR_INTEGRATION=1 after hf auth login",
)


def test_real_pi05_tokenizer_and_processors_preserve_libero_contract():
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from thesis_vla.libero import LiberoTaskConfig
    from thesis_vla.pi05 import Pi05PolicyAdapter

    config = LiberoTaskConfig()
    policy_config = PI05Config(device="cpu")
    policy_pre, _ = make_pre_post_processors(policy_cfg=policy_config)
    tokenizers = [step.input_tokenizer for step in policy_pre.steps if hasattr(step, "input_tokenizer")]
    assert tokenizers and type(tokenizers[0]).__name__ == "GemmaTokenizer"

    raw = {
        "pixels": {
            "image": np.zeros((256, 256, 3), dtype=np.uint8),
            "image2": np.zeros((256, 256, 3), dtype=np.uint8),
        },
        "robot_state": {
            "eef": {
                "pos": np.zeros(3, dtype=np.float32),
                "quat": np.array([0, 0, 0, 1], dtype=np.float32),
            },
            "gripper": {
                "qpos": np.zeros(2, dtype=np.float32),
                "qvel": np.zeros(2, dtype=np.float32),
            },
        },
    }

    class MockInference:
        config = policy_config
        def predict_action_chunk(self, batch):
            assert tuple(batch["observation.state"].shape) == (1, 8)
            assert tuple(batch["observation.images.image"].shape) == (1, 3, 256, 256)
            assert tuple(batch["observation.images.image2"].shape) == (1, 3, 256, 256)
            assert "observation.language.tokens" in batch
            return torch.zeros((1, 2, 7), dtype=torch.float32)
        def reset(self):
            pass

    adapter = Pi05PolicyAdapter.from_lerobot_policy(
        MockInference(), task="pick up the block", env_config=config, execute_steps=1
    )
    prepared = adapter.prepare_batch(raw, "pick up the block")
    assert tuple(prepared["observation.state"].shape) == (1, 8)
    assert "observation.language.tokens" in prepared
    proposal = adapter.act(raw)
    assert proposal.predicted_action.shape == (7,)
    assert proposal.executed_action.shape == (7,)
    assert np.isfinite(proposal.executed_action).all()
