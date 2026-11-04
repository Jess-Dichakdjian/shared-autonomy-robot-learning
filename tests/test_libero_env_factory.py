import os


def test_create_vector_env_selects_configured_task_id(monkeypatch):
    from thesis_vla import libero

    sentinel = object()

    class FakeEnvConfig:
        def create_envs(self, *, n_envs, use_async_envs):
            assert n_envs == 1
            assert use_async_envs is False
            return {"libero_spatial": {3: sentinel}}

    monkeypatch.setattr(libero, "make_lerobot_env_config", lambda config: FakeEnvConfig())
    result = libero.create_vector_env(libero.LiberoTaskConfig(suite="libero_spatial", task_id=3), gl="osmesa")
    assert result is sentinel
    assert os.environ["MUJOCO_GL"] == "osmesa"
