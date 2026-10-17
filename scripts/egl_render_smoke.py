"""Minimal EGL renderer test without LIBERO assets or a policy checkpoint."""
import os
import mujoco

xml = '<mujoco><worldbody><light pos="0 0 2"/><geom type="plane" size="1 1 0.1"/><body pos="0 0 0.2"><freejoint/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, width=128, height=128)
try:
    renderer.update_scene(data)
    image = renderer.render()
    assert image.shape == (128, 128, 3)
    print("MUJOCO_EGL_OK backend={} image={} dtype={}".format(
        os.environ.get("MUJOCO_GL"), image.shape, image.dtype))
finally:
    renderer.close()
