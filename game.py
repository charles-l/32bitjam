import pyray as rl

import raylib as rll
import bisect
import functools
from dataclasses import dataclass, field
from argparse import Namespace
import math
import random
import glm

SCREEN_WIDTH = 320 * 3
SCREEN_HEIGHT = 240 * 3

def load_render_texture_with_depth(width, height):
    target = rl.RenderTexture()
    target.id = rll.rlLoadFramebuffer(width, height)
    assert target.id > 0

    rll.rlEnableFramebuffer(target.id)

    target.texture.id = rll.rlLoadTexture(rl.ffi.NULL, width, height, rll.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8, 1)
    target.texture.width = width
    target.texture.height = height
    target.texture.format = rll.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    target.texture.mipmaps = 1

    target.depth.id = rll.rlLoadTextureDepth(width, height, False)
    target.depth.width = width
    target.depth.height = height
    target.depth.format = 19
    target.depth.mipmaps = 1

    rll.rlFramebufferAttach(target.id, target.texture.id, rll.RL_ATTACHMENT_COLOR_CHANNEL0, rll.RL_ATTACHMENT_TEXTURE2D, 0)
    rll.rlFramebufferAttach(target.id, target.depth.id, rll.RL_ATTACHMENT_DEPTH, rll.RL_ATTACHMENT_TEXTURE2D, 0)

    assert rll.rlFramebufferComplete(target.id)

    rll.rlDisableFramebuffer()

    return target



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

render_target = load_render_texture_with_depth(320, 240)
reflect_target = rl.load_render_texture(320, 240)

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

ssr_shader = rl.load_shader(0, "ssr.frag")
ssr_shader.locs[rl.SHADER_LOC_MATRIX_VIEW] = rl.get_shader_location(
    ssr_shader, "viewMat"
)

skybox = rl.load_model_from_mesh(rl.gen_mesh_cube(1, 1, 1))

water_plane = rl.load_model_from_mesh(rl.gen_mesh_plane(
    400, 400,
    1, 1,
    ))
water_plane.materials[0].maps[0].texture = reflect_target.texture
water_plane.materials[0].shader = rl.load_shader("water.vert", "water.frag")
water_plane.materials[0].shader.locs[rl.SHADER_LOC_MATRIX_MODEL] = rl.get_shader_location(water_plane.materials[0].shader, "matModel")
water_plane.materials[0].shader.locs[rl.SHADER_LOC_VECTOR_VIEW] = rl.get_shader_location(water_plane.materials[0].shader, "viewPos")
water_time_loc = rl.get_shader_location(water_plane.materials[0].shader, "time")

skybox.materials[0].shader = rl.load_shader("skybox.vert", "skybox.frag")
rl.set_shader_value(skybox.materials[0].shader, rl.get_shader_location(skybox.materials[0].shader, "environmentMap"), rl.ffi.new("int*", rl.MATERIAL_MAP_CUBEMAP), rl.SHADER_UNIFORM_INT)

skybox_tex = rl.load_image("skybox.png")
skybox.materials[0].maps[rl.MATERIAL_MAP_CUBEMAP].texture = rl.load_texture_cubemap(skybox_tex, rl.CUBEMAP_LAYOUT_AUTO_DETECT)

tex = rl.load_texture("texture_06.png")
explosion_sheet = rl.load_texture("explosion_sheet.png")
mat = rl.load_material_default()
rl.set_material_texture(mat, rll.MATERIAL_MAP_ALBEDO, tex)
mat.shader = fog_shader

redmat = rl.load_material_default()
rl.set_material_texture(redmat, rll.MATERIAL_MAP_ALBEDO, tex)
redmat.maps[0].color = (255, 0, 0)
redmat.shader = fog_shader

enemymat = rl.load_material_default()
rl.set_material_texture(enemymat, rll.MATERIAL_MAP_ALBEDO, tex)
enemymat.maps[0].color = (255, 255, 0)

ship = rl.load_model("ship.glb").meshes[0]
spikeball = rl.load_model("enemy.glb").meshes[0]
_enemy_model = rl.load_model("enemyship.glb")
enemyball, enemyspikes = _enemy_model.meshes[1], _enemy_model.meshes[0]
gun_sound = rl.load_sound("gun.wav")
rl.set_sound_volume(gun_sound, 0.3)
bg = rl.load_music_stream("32bitjam.mp3")
rl.play_music_stream(bg)

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


@dataclass
class Enemy:
    pos: glm.vec3
    vel: glm.vec3
    goal_pos: glm.vec3
    rate: float
    lane_i: int

state = Namespace(
    vel=glm.vec3(0),
    pos=glm.vec3(0),
    died_pos=glm.vec3(),
    pstate="alive",
    water=0.1,
    bullet_cooldown=0,
    yspring=Spring(2, 0.5, 0, 0),
    rotspring=Spring(1, 0.5, 2, 0),
    camspring=Spring(0.1, 10, 2, glm.vec3(0, 8, 4)),
    bullets_pos=glm.array.zeros(100, glm.vec3),
    bullets_vel=glm.array.zeros(100, glm.vec3),
    bullet_i=0,
    spike_obstacles=glm.array(
        sorted([
            glm.vec3(random.randrange(-20, 20), 0, random.randrange(-10000, -10))
            for _ in range(100)
            ], key=lambda v: v.z)
    ),
    enemies=[
        Enemy(glm.vec3(0, 10, 0), glm.vec3(), glm.vec3(), (random.random() - 0.5) * 5, i) for i in range(5)
        ],
    explosions=[],
    water_particles=[],
    obstacles=sorted([glm.vec3(0, -1, -100),
                      glm.vec3(0, -1, -200),
                      glm.vec3(0, -1, -400)], key=lambda v: v.z),
)

OBSTACLE_DEPTH = -401
OBSTACLE_DIMENSIONS = (40, 800, 4)
PLAYER_RADIUS = 0.5
ENEMY_RADIUS = 1
BULLET_RADIUS = 0.1

obstacle_model = rl.load_model_from_mesh(rl.gen_mesh_cube(*OBSTACLE_DIMENSIONS))
obstacle_model.materials[0].shader = rl.load_shader("obstacle.vert", "obstacle.frag")
obstacle_model.materials[0].shader.locs[rl.SHADER_LOC_MATRIX_MODEL] = rl.get_shader_location(
    obstacle_model.materials[0].shader, "matModel"
)
is_reflection_loc = rl.get_shader_location(
    obstacle_model.materials[0].shader, "isReflection"
    )


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

def z_near(arr, z, depth, key=lambda v: v.z):
    """Assumes arr is sorted in ascending order. Searches for elements in
    range `z - depth` to `z + depth`"""
    i = bisect.bisect_left(arr, z - depth, key=key)
    j = bisect.bisect_right(arr, z + depth, key=key)
    return arr[i:j]


def collide_with_obstacle(pos, radius, kill=True):
    for e in z_near(state.spike_obstacles, pos.z, radius + 2):
        if glm.distance(e, pos) < 2 + radius:
            return pos

    for p in z_near(state.obstacles, pos.z, radius + 4):
        if rl.check_collision_box_sphere(
            obstacle_bounding_box(pos.z),
            pos.to_tuple(),
            PLAYER_RADIUS,
        ):
            return glm.vec3(pos.xy, p.z + OBSTACLE_DIMENSIONS[2] / 2 + 0.1)

    return None


def render_scene(state, camera, interp_pos, reflected=False):
    rl.begin_mode_3d(camera)

    rll.rlDisableBackfaceCulling()
    rll.rlDisableDepthMask()
    rl.draw_model(skybox, (0, 0, 0), 1, rl.WHITE)
    rll.rlEnableBackfaceCulling()
    rll.rlEnableDepthMask()

    if reflected:
        rl.set_shader_value(
            obstacle_model.materials[0].shader,
            is_reflection_loc,
            rl.ffi.new("int *", 1),
            rl.SHADER_UNIFORM_INT,
        )
    else:
        rl.set_shader_value(
            obstacle_model.materials[0].shader,
            is_reflection_loc,
            rl.ffi.new("int *", 0),
            rl.SHADER_UNIFORM_INT,
        )


    if state.pstate == "alive":
        if not reflected or state.pos.y > WATER_LEVEL:
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

    for p in state.spike_obstacles:
        rl.draw_mesh(spikeball, redmat, sum(glm.transpose(glm.translate(p)).to_tuple(), ()))

    for enemy in state.enemies:
        rl.draw_mesh(enemyball, enemymat, sum(glm.transpose(glm.translate(enemy.pos)).to_tuple(), ()))
        rl.draw_mesh(enemyspikes, enemymat, sum(glm.transpose(glm.translate(enemy.pos)
                                                              @ glm.rotate(rl.get_time() * enemy.rate, glm.vec3(0, 0, 1))).to_tuple(), ()))

    rl.draw_sphere_wires(state.died_pos.to_tuple(), PLAYER_RADIUS, 4, 4, rl.BLACK)

    # rl.draw_grid(50, 2)

    for p in state.bullets_pos:
        if glm.length2(p) != 0:
            rl.draw_sphere(p.to_tuple(), BULLET_RADIUS, rl.GREEN)

    for o in state.obstacles:
        rl.draw_model(
            obstacle_model,
            (0, OBSTACLE_DEPTH, o.z),
            1,
            rl.WHITE,
        )

    # DEBUG
    # for o in lanes:
    #   rl.draw_line_3d((o, -3, state.pos.z), (o, -3, state.pos.z - 40), rl.WHITE)

    if not reflected:
        rl.set_shader_value(water_plane.materials[0].shader, water_time_loc, rl.ffi.new("float *", rl.get_time()), rl.SHADER_UNIFORM_FLOAT)
        rl.draw_model(water_plane, (0, WATER_LEVEL, state.pos.z - 180), 1, rl.color_alpha(rl.BLUE, 0.4))

    #rl.draw_plane(
    #    glm.vec3(0, WATER_LEVEL, state.pos.z - 170).to_tuple(),
    #    (40, 400),
    #    rl.color_alpha(rl.BLUE, 0.3),
    #)

    for i, (pos, init_time) in enumerate(state.explosions):
        frame = int((rl.get_time() - init_time) / 0.04)
        if frame > 12:
            del state.explosions[i]
        else:
            rl.draw_billboard_rec(
                camera,
                explosion_sheet,
                rl.Rectangle(256 * frame, 0, 256, 256),
                pos.to_tuple(),
                (4, 4),
                rl.WHITE,
            )

    for pos, _ in state.water_particles:
        if not reflected or pos.y > WATER_LEVEL:
            rl.draw_sphere(pos.to_tuple(), 0.3, rl.BLUE)

    rl.end_mode_3d()


def update(state):

    rl.update_music_stream(bg)

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

    waterlog_factor = 1 - (max(state.water - 1.5, 0) * 0.5)

    if state.pstate == "alive":
        state.pos.y = dive
        interp_pos = glm.vec3(state.pos.x, state.yspring.update(dive), state.pos.z)
        if rl.is_key_down(rl.KEY_SPACE):
            if state.bullet_cooldown <= 0:
                state.bullets_pos[state.bullet_i] = glm.vec3(interp_pos)
                state.bullets_vel[state.bullet_i] = glm.vec3(0, 0, state.vel.z) + glm.vec3(
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
            state.vel.z += inputv.y * (1 if glm.length(state.vel.z) < 2 else 0.03)
        elif inputv.y > 0:
            state.vel.z += inputv.y / 2
        else:
            state.vel.z *= 0.99
            state.vel.x *= 0.9

        # add waterlog factor
        state.vel *= waterlog_factor

        # underwater logic
        if interp_pos.y < WATER_LEVEL:
            state.water += 0.5 * rl.get_frame_time()

            # cap underwater speed
            if glm.length(state.vel) > 2.5:
                state.vel = glm.normalize(state.vel) * 2.5

        # clamp max vel
        if glm.length(state.vel) > 3:
            state.vel = glm.normalize(state.vel) * 3

        state.pos += state.vel

        if glm.abs(state.pos.x) > 20:
            state.pos.x = glm.sign(state.pos.x) * 20
            state.vel.x = 0

        state.rotspring.update(rotation)
        state.bullet_cooldown -= rl.get_frame_time()

        if died_pos := collide_with_obstacle(interp_pos, PLAYER_RADIUS):
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
        camera_goal_pos = interp_pos + glm.vec3(0, 5 + state.vel.z * 1.5, 8)
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

    rl.set_shader_value(
        water_plane.materials[0].shader,
        water_plane.materials[0].shader.locs[rl.SHADER_LOC_VECTOR_VIEW],
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

    if state.pstate == 'alive':
        cam.fovy = (
            45
            + (
                glm.distance(interp_pos, (cam.position.x, cam.position.y, cam.position.z))
                / 22
            )
            * 20
        )

    LANE_WIDTH = 4
    lanes = []
    # update enemies
    if state.enemies:
        lanes = [x for x in range(-20, 20, LANE_WIDTH)]
        for obstacle in z_near(state.spike_obstacles, max(state.enemies, key=lambda e: e.pos.z).pos.z - 10, 20):
            for off in (-2, 0, 2):
                try:
                    lanes.remove(math.floor((obstacle.x + off) / LANE_WIDTH) * LANE_WIDTH)
                except ValueError:
                    pass

        available_lanes = list(lanes)
        start_i = len(available_lanes) // 2 - (len(state.enemies) // 2) if len(available_lanes) > len(state.enemies) else 0
        for i, enemy in enumerate(state.enemies):
            if available_lanes:
                lane_i = (start_i + enemy.lane_i) % len(available_lanes)
                x = available_lanes[lane_i]
                del available_lanes[lane_i]
            else:
                x = random.random() * 40 - 20
            enemy.goal_pos = glm.vec3(x, 0, state.pos.z) + glm.vec3(0, 0, -16)

            enemy.vel = (enemy.goal_pos - enemy.pos) * 0.3
            if glm.length(enemy.vel) > 3:
                enemy.vel = glm.normalize(enemy.vel) * 3
            enemy.pos += enemy.vel
            for collider in z_near(state.spike_obstacles, enemy.pos.z, 4):
                if glm.distance(collider, enemy.pos) < 3:
                    state.explosions.append((enemy.pos, rl.get_time()))
                    del state.enemies[i]


    # update bullets
    state.bullets_pos += state.bullets_vel
    for i, bullet in enumerate(state.bullets_pos):
        collide = collide_with_obstacle(bullet, BULLET_RADIUS)
        for i, e in enumerate(state.enemies):
            if glm.distance(bullet, e.pos) < BULLET_RADIUS + ENEMY_RADIUS:
                state.explosions.append((bullet, rl.get_time()))
                del state.enemies[i]
                collide = True
                break

        if collide:
            state.bullets_pos[i] = glm.vec3()
            state.bullets_vel[i] = glm.vec3()


    # update water particles
    for i, (pos, vel) in enumerate(state.water_particles):
        pos += vel
        pos.y -= 0.098
        if pos.y < WATER_LEVEL - 8:
            del state.water_particles[i]

    rl.begin_texture_mode(reflect_target)
    rl.clear_background(rl.BLACK)

    reflect_cam = rl.Camera3D(cam.position, cam.target, cam.up, cam.fovy)
    reflect_cam.position.y = reflect_cam.position.y * -1 + WATER_LEVEL
    reflect_cam.target.y = reflect_cam.target.y * -1 + WATER_LEVEL
    render_scene(state, reflect_cam, interp_pos, reflected=True)
    rl.end_texture_mode()

    rl.begin_texture_mode(render_target)
    rl.clear_background(rl.GRAY)

    render_scene(state, cam, interp_pos)

    rl.end_texture_mode()

    rl.clear_background(rl.BLACK)

    rl.begin_shader_mode(psx_shader)
    #rl.set_shader_value_texture(psx_shader, rl.get_shader_location(psx_shader, "texture1"), render_target.depth)
    rl.draw_texture_pro(
        render_target.texture,
        rl.Rectangle(0, 0, render_target.texture.width, -render_target.texture.height),
        rl.Rectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT),
        rl.Vector2(0, 0),
        0,
        rl.SKYBLUE if is_under_water else rl.WHITE,
    )
    rl.end_shader_mode()

    # debug reflect buffer
    # rl.draw_texture_pro(reflect_target.texture,
    #                     rl.Rectangle(0, 0, render_target.texture.width, -render_target.texture.height),
    #                     rl.Rectangle(0, 0, 200, 150),
    #                     rl.Vector2(0, 0),
    #                     0,
    #                     rl.WHITE)

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
        rl.draw_text(
            f"{round(abs(state.vel.z)*10):} m/s",
            SCREEN_WIDTH // 2 + 40,
            SCREEN_HEIGHT // 2 - 20,
            20,
            rl.GREEN)

    if state.pstate == "dead":
        rl.draw_text(
            "Press down arrow to restart", 10, int(SCREEN_HEIGHT / 2), 50, rl.WHITE
        )

    if state.water - 1 > 0.1:
        if (rl.get_time() % 0.4) < 0.2:
            x = SCREEN_WIDTH // 2 + 40
            rl.draw_text("WARNING: WATERLOGGING", x, SCREEN_HEIGHT // 2 - 120, 30, rl.RED)
            if waterlog_factor < 1:
                rl.draw_text(f"{int((2-state.water)*100)}% capacity", x, SCREEN_HEIGHT // 2 - 90, 30, rl.RED)

if __name__ == "__main__":
    while not rl.window_should_close():
        rl.begin_drawing()
        rl.clear_background(rl.BLACK)

        update(state)
        rl.end_drawing()

    rl.close_window()
