import pyray as rl

import raylib as rll
import functools
from dataclasses import dataclass, field
from argparse import Namespace
import math
import random
import glm

SCREEN_WIDTH = 320 * 3
SCREEN_HEIGHT = 240 * 3

if __name__ == "__main__":
    rl.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, "game")
    rl.set_target_fps(60)
    rl.set_config_flags(rl.FLAG_VSYNC_HINT)
    rl.init_audio_device()

cam = rl.Camera3D(
    rl.Vector3(-10, 15, -10),
    rl.Vector3(0, 0, 0),
    rl.Vector3(0, 1, 0),
    45,
    rl.CAMERA_PERSPECTIVE,
)

psx_shader = rl.load_shader(0, "psx.frag")
fog_shader = rl.load_shader("base.vert", "fog.frag")
fog_shader.locs[rl.SHADER_LOC_MATRIX_MODEL] = rl.get_shader_location(
    fog_shader, "matModel"
)
fog_shader.locs[rl.SHADER_LOC_VECTOR_VIEW] = rl.get_shader_location(
    fog_shader, "viewPos"
)
FOG_DENSITY_LOC = rl.get_shader_location(fog_shader, "fogDensity")
FOG_COLOR_LOC = rl.get_shader_location(fog_shader, "fogColor")

render_target = rl.load_render_texture(320, 240)

tex = rl.load_texture("texture_06.png")
explosion_sheet = rl.load_texture("explosion_sheet.png")
mat = rl.load_material_default()
rl.set_material_texture(mat, rll.MATERIAL_MAP_ALBEDO, tex)
mat.shader = fog_shader

redmat = rl.load_material_default()
rl.set_material_texture(redmat, rll.MATERIAL_MAP_ALBEDO, tex)
redmat.maps[0].color = (255, 0, 0)
redmat.shader = fog_shader

ship = rl.load_model("ship.glb").meshes[0]
enemy = rl.load_model("enemy.glb").meshes[0]
gun_sound = rl.load_sound("gun.wav")
rl.set_sound_volume(gun_sound, 0.3)

WATER_LEVEL = -2


class Spring:
    """Damped spring. Based on https://www.youtube.com/watch?v=KPoeNZZ6H4s"""

    def __init__(self, f, z, r, x0):
        self.k1 = z / (math.pi * f)
        self.k2 = 1 / ((2 * math.pi * f) ** 2)
        self.k3 = r * z / (2 * math.pi * f)
        self.xp = x0
        self.y = x0
        self.yd = type(x0)(0)

    def update(self, x, xd=None, k3=None):
        dt = rl.get_frame_time() or 0.01
        if xd is None:
            xd = (x - self.xp) / dt
            self.xp = x
        k2_stable = max(self.k2, 1.1 * ((dt**2) / 4 + dt * self.k1 / 2))
        self.y += dt * self.yd
        self.yd += (
            dt * (x + (k3 or self.k3) * xd - self.y - self.k1 * self.yd) / k2_stable
        )
        return self.y


state = Namespace(
    vel=glm.vec3(0),
    pos=glm.vec3(0),
    died_pos=glm.vec3(),
    pstate="alive",
    water=0,
    bullet_cooldown=0,
    yspring=Spring(2, 0.5, 0, 0),
    rotspring=Spring(1, 0.5, 2, 0),
    camspring=Spring(0.1, 10, 2, glm.vec3(0, 8, 4)),
    bullets_pos=glm.array.zeros(100, glm.vec3),
    bullets_vel=glm.array.zeros(100, glm.vec3),
    bullet_i=0,
    enemy_pos=glm.array(
        [
            glm.vec3(random.randrange(-20, 20), 0, random.randrange(-1000, -10))
            for _ in range(100)
        ]
    ),
    explosions=[],
    water_particles=[],
    obstacles=[-100, -200, -400],
)

OBSTACLE_DEPTH = -401
OBSTACLE_DIMENSIONS = (40, 800, 4)
PLAYER_RADIUS = 0.5


def obstacle_bounding_box(z):
    return rl.BoundingBox(
        rl.Vector3(
            -OBSTACLE_DIMENSIONS[0] / 2,
            -OBSTACLE_DIMENSIONS[1] / 2 + OBSTACLE_DEPTH,
            -OBSTACLE_DIMENSIONS[2] / 2 + z,
        ),
        rl.Vector3(
            OBSTACLE_DIMENSIONS[0] / 2,
            OBSTACLE_DIMENSIONS[1] / 2 + OBSTACLE_DEPTH,
            OBSTACLE_DIMENSIONS[2] / 2 + z,
        ),
    )

GRID_SIZE = 8


class SpatialHash:
    def __init__(self, grid_size):
        self.hash = {}

    @classmethod
    def get_bucket(p: glm.vec3):
        return glm.round(p.xz / GRID_SIZE).to_tuple()

    def __getitem__(self, k):
        return self.hash.get(k, [])

    def add(self, pos, item):
        k = self.get_bucket(pos)
        self.hash[k] = self.hash.get(k, []).append(item)

    def around(self, pos):
        r = []
        for o in (
            glm.vec3(GRID_SIZE, 0, 0),
            glm.vec3(-GRID_SIZE, 0, 0),
            glm.vec3(0, 0, GRID_SIZE),
            glm.vec3(0, 0, -GRID_SIZE),
            glm.vec3(GRID_SIZE, 0, GRID_SIZE),
            glm.vec3(GRID_SIZE, 0, -GRID_SIZE),
            glm.vec3(-GRID_SIZE, 0, GRID_SIZE),
            glm.vec3(-GRID_SIZE, 0, -GRID_SIZE),
        ):
            r.extend(self[self.get_bucket(pos + o)])
        return r


spatial_hash = {}


def reinit(nstate):
    global state
    state = nstate

    for p in state.bullets_pos:
        if glm.length2(p) != 0:
            spatial_hash[glm.round(p / 10)]


def collide_with_obstacle(player_pos):
    for e in state.enemy_pos:
        if glm.distance(e, player_pos) < 2 + PLAYER_RADIUS:
            return player_pos

    for z in state.obstacles:
        if rl.check_collision_box_sphere(
            obstacle_bounding_box(z),
            player_pos.to_tuple(),
            PLAYER_RADIUS,
        ):
            return glm.vec3(player_pos.xy, z + OBSTACLE_DIMENSIONS[2] / 2 + 0.1)
    return None


def update(state):
    # input
    inputv = glm.vec2()
    dive = 0
    rotation = 0

    if rl.is_key_down(rl.KEY_RIGHT):
        inputv.x += 1
        rotation -= 0.3
    if rl.is_key_down(rl.KEY_LEFT):
        inputv.x -= 1
        rotation += 0.3
    if rl.is_key_down(rl.KEY_UP):
        inputv.y -= 1
    if rl.is_key_down(rl.KEY_DOWN):
        inputv.y += 1
    if rl.is_key_down(rl.KEY_LEFT_SHIFT):
        dive = -6  # -2.4
    elif rl.is_key_down(rl.KEY_LEFT_CONTROL):
        if state.water > 0:
            state.water -= 0.2 * rl.get_frame_time()
            dive = 3
            state.water_particles.append(
                (state.pos + glm.vec3(-0.3, -0.1, -0.1), state.vel + glm.vec3(0, -1, 0))
            )
            state.water_particles.append(
                (state.pos + glm.vec3(0.3, -0.1, -0.1), state.vel + glm.vec3(0, -1, 0))
            )
    else:
        dive = 0

    interp_pos = state.pos

    if state.pstate == "alive":
        state.pos.y = dive
        interp_pos = glm.vec3(state.pos.x, state.yspring.update(dive), state.pos.z)
        if rl.is_key_down(rl.KEY_SPACE):
            if state.bullet_cooldown <= 0:
                state.bullets_pos[state.bullet_i] = glm.vec3(interp_pos)
                state.bullets_vel[state.bullet_i] = glm.vec3(state.vel) + glm.vec3(
                    0, 0, -0.4
                )
                state.bullet_i = (state.bullet_i + 1) % len(state.bullets_pos)
                state.bullet_cooldown = 0.1
                rl.set_sound_pitch(gun_sound, 1.0 + random.random() * 0.1 - 0.2)
                rl.play_sound(gun_sound)

        if glm.length(inputv) != 0:
            inputv = glm.normalize(inputv) * 0.1
            state.vel.x += inputv.x

            if inputv.y < 0 and state.water > 0:
                state.water_particles.append(
                    (interp_pos + glm.vec3(0, -0.1, -0.1), state.vel + glm.vec3(0, 0, 1))
                )
                state.water -= 0.1 * rl.get_frame_time()
                state.vel.z += inputv.y
            elif inputv.y > 0:
                state.vel.z += inputv.y / 2
        else:
            state.vel *= 0.7
            if state.vel.z > -0.8:
                state.vel.z = -0.8

        # clamp max vel
        if glm.length(state.vel) > 2:
            state.vel = glm.normalize(state.vel) * 2

        state.pos += state.vel

        if glm.abs(state.pos.x) > 20:
            state.pos.x = glm.sign(state.pos.x) * 20
            state.vel.x = 0

        state.rotspring.update(rotation)
        state.bullet_cooldown -= rl.get_frame_time()
        if interp_pos.y < WATER_LEVEL:
            state.water += 0.5 * rl.get_frame_time()

        if died_pos := collide_with_obstacle(interp_pos):
            state.pstate = "dead"
            state.died_pos = died_pos
            state.explosions.append((died_pos, rl.get_time()))

        if state.water > 2:
            state.pstate = "dead"
            state.died_pos = interp_pos
            state.explosions.append((interp_pos, rl.get_time()))

    elif state.pstate == "dead":
        if rl.is_key_pressed(rl.KEY_DOWN):
            state.pos = glm.vec3()
            state.pstate = "alive"
            cam.position = (0, 4, 4)
            state.water = 0

    # cam update
    if state.pstate == "alive":
        camera_goal_pos = interp_pos + glm.vec3(0, 5 + state.vel.z * 2, 8)
    elif state.pstate == "dead":
        camera_goal_pos = state.pos + glm.vec3(0, -0.3 + state.camspring.y.y, 12)
    else:
        assert False
    state.camspring.update(camera_goal_pos)
    cam.position = state.camspring.y.to_tuple()
    cam.target = (interp_pos + glm.vec3(0, 0, -4)).to_tuple()
    is_under_water = cam.position.y < WATER_LEVEL

    rl.set_shader_value(
        fog_shader,
        fog_shader.locs[rl.SHADER_LOC_VECTOR_VIEW],
        rl.ffi.cast("void *", rl.ffi.addressof(cam.position)),
        rl.SHADER_UNIFORM_VEC3,
    )

    if is_under_water:
        rl.set_shader_value(
            fog_shader,
            FOG_DENSITY_LOC,
            rl.ffi.new("float *", 0.2),
            rl.SHADER_UNIFORM_FLOAT,
        )
        rl.set_shader_value(
            fog_shader,
            FOG_COLOR_LOC,
            rl.ffi.new("float[3]", [0, 0, 1]),
            rl.SHADER_UNIFORM_VEC3,
        )
    else:
        rl.set_shader_value(
            fog_shader,
            FOG_DENSITY_LOC,
            rl.ffi.new("float *", 0.03),
            rl.SHADER_UNIFORM_FLOAT,
        )
        rl.set_shader_value(
            fog_shader,
            FOG_COLOR_LOC,
            rl.ffi.new("float[3]", [0.1, 0.1, 0.4]),
            rl.SHADER_UNIFORM_VEC3,
        )

    cam.fovy = (
        45
        + (
            glm.distance(interp_pos, (cam.position.x, cam.position.y, cam.position.z))
            / 16
        )
        * 20
    )

    # update bullet vels
    state.bullets_pos += state.bullets_vel

    # update water particles
    for i, (pos, vel) in enumerate(state.water_particles):
        pos += vel
        pos.y -= 0.098
        if pos.y < WATER_LEVEL - 8:
            del state.water_particles[i]

    rl.begin_texture_mode(render_target)
    rl.clear_background(rl.GRAY)
    rl.begin_mode_3d(cam)

    if state.pstate == "alive":
        rl.draw_mesh(
            ship,
            mat,
            sum(
                glm.transpose(
                    glm.translate(interp_pos)
                    @ glm.rotate(state.rotspring.y, glm.vec3(0, 0, 1))
                ).to_tuple(),
                (),
            ),
        )

    for e in state.enemy_pos:
        rl.draw_mesh(enemy, redmat, sum(glm.transpose(glm.translate(e)).to_tuple(), ()))

    rl.draw_sphere_wires(state.died_pos.to_tuple(), PLAYER_RADIUS, 4, 4, rl.WHITE)

    # rl.draw_grid(50, 2)

    for p in state.bullets_pos:
        if glm.length2(p) != 0:
            rl.draw_sphere(p.to_tuple(), 0.1, rl.GREEN)

    for o in state.obstacles:
        rl.draw_cube(
            (0, OBSTACLE_DEPTH, o),
            OBSTACLE_DIMENSIONS[0],
            OBSTACLE_DIMENSIONS[1],
            OBSTACLE_DIMENSIONS[2],
            rl.DARKGRAY,
        )

    rl.draw_plane(
        glm.vec3(0, WATER_LEVEL, state.pos.z - 170).to_tuple(),
        (40, 400),
        rl.color_alpha(rl.BLUE, 0.3),
    )

    for i, (pos, init_time) in enumerate(state.explosions):
        frame = int((rl.get_time() - init_time) / 0.04)
        if frame > 12:
            del state.explosions[i]
        else:
            rl.draw_billboard_rec(
                cam,
                explosion_sheet,
                rl.Rectangle(256 * frame, 0, 256, 256),
                pos.to_tuple(),
                (4, 4),
                rl.WHITE,
            )

    for pos, _ in state.water_particles:
        rl.draw_sphere(pos.to_tuple(), 0.3, rl.BLUE)

    rl.end_mode_3d()
    rl.end_texture_mode()

    rl.clear_background(rl.BLACK)
    rl.begin_shader_mode(psx_shader)
    rl.draw_texture_pro(
        render_target.texture,
        rl.Rectangle(0, 0, render_target.texture.width, -render_target.texture.height),
        rl.Rectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT),
        rl.Vector2(0, 0),
        0,
        rl.SKYBLUE if is_under_water else rl.WHITE,
    )
    rl.end_shader_mode()
    if state.pstate == "alive":
        water, waterlogged = min(state.water, 1), max(state.water - 1, 0)
        rl.draw_ring(
            (SCREEN_WIDTH // 2 + 40, SCREEN_HEIGHT // 2 - 40),
            10,
            13,
            0,
            360 / (1 / water) if water > 0 else 0,
            16,
            rl.SKYBLUE,
        )
        rl.draw_ring(
            (SCREEN_WIDTH // 2 + 40, SCREEN_HEIGHT // 2 - 40),
            10,
            13,
            0,
            360 / (1 / waterlogged) if waterlogged > 0 else 0,
            16,
            rl.RED,
        )
    if state.pstate == "dead":
        rl.draw_text(
            "Press down arrow to restart", 10, int(SCREEN_HEIGHT / 2), 50, rl.WHITE
        )

    if state.water - 1 > 0.1:
        if (rl.get_time() % 0.4) < 0.2:
            rl.draw_text("WARNING: waterlog", SCREEN_WIDTH // 2 + 40, SCREEN_HEIGHT // 2 - 90, 30, rl.RED)

if __name__ == "__main__":
    while not rl.window_should_close():
        rl.begin_drawing()
        rl.clear_background(rl.BLACK)

        update(state)
        rl.end_drawing()

    rl.close_window()
