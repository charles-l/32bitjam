import pyray as rl

import raylib as rll
import functools
from dataclasses import dataclass, field
from argparse import Namespace
import math
import glm

SCREEN_WIDTH = 320 * 3
SCREEN_HEIGHT = 240 * 3

if __name__ == "__main__":
    rl.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, "game")
    rl.set_target_fps(60)
    rl.set_config_flags(rl.FLAG_MSAA_4X_HINT | rl.FLAG_VSYNC_HINT)

cam = rl.Camera3D(
    rl.Vector3(-10, 15, -10),
    rl.Vector3(0, 0, 0),
    rl.Vector3(0, 1, 0),
    45,
    rl.CAMERA_PERSPECTIVE,
)

rl.set_camera_mode(cam, rl.CAMERA_FREE)
psx_shader = rl.load_shader(0, "psx.frag")
render_target = rl.load_render_texture(320, 240)

tex = rl.load_texture("texture_07.png")
mesh = rl.gen_mesh_sphere(2, 32, 16)
mat = rl.load_material_default()
rl.set_material_texture(mat, rll.MATERIAL_MAP_ALBEDO, tex)

state = Namespace(
    pos=glm.vec3(0)
    )

def update(state):
    rl.update_camera(cam)

    state.pos.x -= 0.1

    rl.begin_texture_mode(render_target)
    rl.clear_background(rl.GRAY)
    rl.begin_mode_3d(cam)
    rl.draw_mesh(mesh, mat, sum(glm.transpose(glm.translate(state.pos)).to_tuple(), ()))
    rl.draw_grid(10, 1)
    rl.end_mode_3d()
    rl.end_texture_mode()

    rl.clear_background(rl.BLACK)
    rl.begin_shader_mode(psx_shader)
    rl.draw_texture_pro(render_target.texture,
                        rl.Rectangle(0, 0, render_target.texture.width, -render_target.texture.height),
                        rl.Rectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT),
                        rl.Vector2(0, 0),
                        0,
                        rl.WHITE)
    rl.end_shader_mode()

if __name__ == "__main__":
    while not rl.window_should_close():
        rl.begin_drawing()
        rl.clear_background(rl.BLACK)

        update(state)
        rl.end_drawing()

    rl.close_window()
