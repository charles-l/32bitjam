import pyray as rl
import raylib as rll
import bisect
import functools
from dataclasses import dataclass, field
from argparse import Namespace
from perlin_noise import PerlinNoise
import math
import random
import glm

SCREEN_WIDTH = 320 * 3
SCREEN_HEIGHT = 240 * 3

pnoise = PerlinNoise()

def draw_text_centered(text, font_size, color, x=None, y=None):
    if x is None:
        x = SCREEN_WIDTH // 2 - rl.measure_text(text, font_size) // 2
    if y is None:
        y = SCREEN_HEIGHT // 2 - font_size // 2
    rl.draw_text(text, x, y, font_size, color)


def load_render_texture_with_depth(width, height):
    target = rl.RenderTexture()
    target.id = rll.rlLoadFramebuffer(width, height)
    assert target.id > 0

    rll.rlEnableFramebuffer(target.id)

    target.texture.id = rll.rlLoadTexture(
        rl.ffi.NULL, width, height, rll.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8, 1
    )
    target.texture.width = width
    target.texture.height = height
    target.texture.format = rll.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    target.texture.mipmaps = 1

    target.depth.id = rll.rlLoadTextureDepth(width, height, False)
    target.depth.width = width
    target.depth.height = height
    target.depth.format = 19
    target.depth.mipmaps = 1

    rll.rlFramebufferAttach(
        target.id,
        target.texture.id,
        rll.RL_ATTACHMENT_COLOR_CHANNEL0,
        rll.RL_ATTACHMENT_TEXTURE2D,
        0,
    )
    rll.rlFramebufferAttach(
        target.id,
        target.depth.id,
        rll.RL_ATTACHMENT_DEPTH,
        rll.RL_ATTACHMENT_TEXTURE2D,
        0,
    )

    assert rll.rlFramebufferComplete(target.id)

    rll.rlDisableFramebuffer()

    return target


if __name__ == "__main__":
    rl.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, "game")
    rl.set_target_fps(60)
    rl.set_config_flags(rl.FLAG_VSYNC_HINT)
    rl.init_audio_device()

cam = rl.Camera3D(
    rl.Vector3(0, 0, 0),
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

skydome = rl.load_model("sky.glb")
checkpoint_model = rl.load_model("checkpoint.glb")
shark = rl.load_model("shark.glb")
jaw = shark.meshes[0]
eyes = shark.meshes[1]
body = shark.meshes[2]
SHARK_WEAK_POINTS = [
    (glm.vec3(-6, 3, 2), 1, None),
    (glm.vec3(+6, 3, 2), 1, None),
    (glm.vec3(0, 0, 2), 2, lambda state: state.shark.jawspring.y > 0),
    ]

water_plane = rl.load_model_from_mesh(
    rl.gen_mesh_plane(
        400,
        400,
        1,
        1,
    )
)
water_plane.materials[0].maps[0].texture = reflect_target.texture
water_plane.materials[0].shader = rl.load_shader("water.vert", "water.frag")
water_plane.materials[0].shader.locs[
    rl.SHADER_LOC_MATRIX_MODEL
] = rl.get_shader_location(water_plane.materials[0].shader, "matModel")
water_plane.materials[0].shader.locs[
    rl.SHADER_LOC_VECTOR_VIEW
] = rl.get_shader_location(water_plane.materials[0].shader, "viewPos")
water_time_loc = rl.get_shader_location(water_plane.materials[0].shader, "time")

skydome.materials[0].shader = rl.load_shader("skybox.vert", "skybox.frag")
skydome.materials[0].maps[0].texture = rl.load_texture("sky.png")
invert_sky_loc = rl.get_shader_location(skydome.materials[0].shader, "flipColor")

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

ocean_container = rl.load_model("oceancontainer.glb")
ocean_container.materials[0] = redmat

gun_sound = rl.load_sound("gun.wav")
explosion_sound = rl.load_sound("explosion.wav")
rl.set_sound_volume(explosion_sound, 0.3)
click_sound = rl.load_sound("click.wav")
fanfare_sound = rl.load_sound("fanfare.ogg")
impact = rl.load_sound("impact.wav")
enemy_shot = rl.load_sound("enemyshot.wav")
rl.set_sound_volume(gun_sound, 0.3)
bg = rl.load_music_stream("32bitjam.ogg")
engine = rl.load_music_stream("engine.wav")
rl.set_music_volume(engine, 1.3)

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
    state_time: float
    state = "idle"


state = Namespace(
    cam_shake_amount=0,
    level=0,
    armor=1,
    invincible_time=0,
    vel=glm.vec3(0),
    pos=glm.vec3(0),
    died_pos=glm.vec3(),
    pstate="alive",
    water=0.3,
    bullet_cooldown=0,
    yspring=Spring(2, 0.5, 0, 0),
    rotspring=Spring(1, 0.5, 2, 0),
    camspring=Spring(0.1, 10, 2, glm.vec3(0, 0, 0)),
    bullets_pos=glm.array.zeros(100, glm.vec3),
    bullets_vel=glm.array.zeros(100, glm.vec3),
    enemy_bullets=[],
    bullet_i=0,
    spike_obstacles=glm.array.zeros(0, glm.vec3),
    enemies=[],
    explosions=[],
    water_particles=[],
    obstacles=[],
    checkpoint_dist=100,
    checkpoint_time=0,
    total_time=0,
    level_time=45,
    shark=None,
)


def level_1(state):
    state.spike_obstacles = glm.array(
        sorted(
            [
                glm.vec3(random.randrange(-20, 20), 0, random.randrange(-10000, -10))
                for _ in range(100)
            ],
            key=lambda v: v.z,
        )
    )
    state.obstacles = sorted(
        [glm.vec3(0, -1, -500), glm.vec3(0, -1, -2500), glm.vec3(0, 1, -3400)],
        key=lambda v: v.z,
    )
    state.checkpoint_dist = 10000

    while state.pos.z > -state.checkpoint_dist:
        yield

    global flash
    flash = 1


def level_2(state):
    state.checkpoint_dist = 10000
    state.obstacles = sorted(
        [
            glm.vec3(0, 1, -500),
            glm.vec3(0, -1, -200),
            glm.vec3(0, -1, -2000),
            glm.vec3(0, -1, -2300),
            glm.vec3(0, -1, -2600),
            glm.vec3(0, 1, -3400),
            glm.vec3(0, 1, -6000),
        ],
        key=lambda v: v.z,
    )
    state.spike_obstacles = glm.array(
        sorted(
            [
                glm.vec3(random.randrange(-20, 20), 0, random.randrange(-10000, -1000))
                for _ in range(100)
            ]
            + [
                glm.vec3(random.randrange(-20, 20), 0, random.randrange(-8020, -8000))
                for _ in range(20)
            ]
            + [
                glm.vec3(random.randrange(-20, 20), 0, random.randrange(-8100, -8080))
                for _ in range(20)
            ]
            + [
                glm.vec3(random.randrange(-20, 20), 0, random.randrange(-8420, -8400))
                for _ in range(20)
            ],
            key=lambda v: v.z,
        )
    )

    while state.pos.z > -500:
        yield

    state.enemies = [
        Enemy(
            glm.vec3(0, 30, -500),
            glm.vec3(),
            glm.vec3(),
            (random.random() - 0.5) * 5,
            i,
            2 + (random.random()),
        )
        for i in range(5)
    ]

    while state.pos.z > -6000:
        yield

    state.enemies = [
        Enemy(
            glm.vec3(0, 30, -6020),
            glm.vec3(),
            glm.vec3(),
            (random.random() - 0.5) * 5,
            i,
            2 + (random.random()),
        )
        for i in range(5)
    ]

    # state.checkpoint_dist = 10000

    while state.pos.z > -state.checkpoint_dist:
        yield

    global flash
    flash = 1

def display_runup(label, value):
    display = 0.0
    while display < value:
        rl.draw_rectangle(
            0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, rl.color_alpha(rl.BLACK, 0.5)
        )
        display = min(display + 1, value)
        text = f"{label}: {display:.2f}s"
        draw_text_centered(text, 30, rl.WHITE)
        rl.play_sound(click_sound)
        if display == value:
            rl.play_sound(impact)
        yield
    rl.play_sound(impact)

def intermission(state):
    level_time_seconds = state.level_time
    state.obstacles = []
    state.enemies = []
    state.spike_obstacles = []
    state.total_time += level_time_seconds
    state.checkpoint_dist = float("inf")
    state.armor += 1
    yield from display_runup('LEVEL TIME', level_time_seconds)
    while True:
        state.water = 1
        rl.draw_rectangle(
            0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, rl.color_alpha(rl.BLACK, 0.5)
        )
        draw_text_centered(f"LEVEL TIME: {level_time_seconds:.2f}s", 40, rl.WHITE)
        if rl.get_time() % 0.8 < 0.6:
            draw_text_centered(f"REWARD: +1 ARMR", 40, rl.GREEN, y=SCREEN_HEIGHT//2 + 40)
        draw_text_centered(
            f"Hit down arrow to continue", 20, rl.WHITE, y=SCREEN_HEIGHT // 2 + 20
        )
        if rl.is_key_pressed(rl.KEY_DOWN):
            return
        yield


low = [0.0, 0.0]
cutoff = 300.0 / 44100.0
k = cutoff / (cutoff + 0.1591549431)  # RC filter formula


@rl.ffi.callback("void(*)(void *, unsigned int)")
def lowpass(buffer, frames: int):
    global low
    buffer = rl.ffi.cast("float *", buffer)

    for i in range(0, frames * 2, 2):
        l = buffer[i]
        r = buffer[i + 1]
        low[0] += k * (l - low[0])
        low[1] += k * (r - low[1])
        buffer[i] = low[0]
        buffer[i + 1] = low[1]

def level_3(state):
    rl.set_shader_value(
        skydome.materials[0].shader,
        invert_sky_loc,
        rl.ffi.new("int *", 1),
        rl.SHADER_UNIFORM_INT,
    )
    state.checkpoint_dist = float('infinity')
    t = rl.get_time()
    while rl.get_time() - t < 4:
        yield

    state.shark=Namespace(
        loop_timer=0,
        jawspring=Spring(1, 0.5, 2, 0),
        posspring=Spring(1, 1, -0.01, state.pos + glm.vec3(0, 0, -500)),
        pos=state.pos + glm.vec3(0, 0, -500),
        fired=False,
        step=0,
        phase=0,
        shake_amount=0,
        hitpoints=70,
        )

    while state.shark.hitpoints > 0:
        yield

    global flash
    flash = 1

    rl.set_shader_value(
        skydome.materials[0].shader,
        invert_sky_loc,
        rl.ffi.new("int *", 0),
        rl.SHADER_UNIFORM_INT,
    )

    while state.shark.pos.y > -10:
        yield

    rl.pause_music_stream(bg)
    rl.play_sound(fanfare_sound)
    level_time_seconds = state.level_time
    state.total_time += level_time_seconds
    yield from display_runup('LEVEL TIME', level_time_seconds)
    t = rl.get_time()
    while rl.get_time() - t < 4:
        rl.draw_rectangle(
            0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, rl.color_alpha(rl.BLACK, 0.5)
        )
        text = f"LEVEL TIME: {level_time_seconds:.2f}s"
        draw_text_centered(text, 40, rl.WHITE)
        yield

    yield from display_runup('TOTAL TIME', state.total_time)
    t = rl.get_time()
    while True:
        rl.draw_rectangle(
            0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, rl.color_alpha(rl.BLACK, 0.5)
        )
        text = f"TOTAL TIME: {state.total_time:.2f}s"
        draw_text_centered(text, 40, rl.WHITE)
        draw_text_centered(f'ARMOR: {state.armor}', 40, rl.GREEN, y=SCREEN_HEIGHT // 2 + 50)
        score = max(0, 500 - state.total_time) * state.armor
        if score > 300:
            rank = 'S'
        elif score > 60:
            rank = 'A'
        else:
            rank = 'B'
        draw_text_centered(f'SCORE: {int(score)}', 40, rl.WHITE, y=SCREEN_HEIGHT // 2 + 100)
        if rl.get_time() % 0.4 < 0.2:
            draw_text_centered(f'RANK: {rank}', 40, rl.WHITE, y=SCREEN_HEIGHT // 2 + 150)
        yield


levels = [
    level_1,
    intermission,
    level_2,
    intermission,
    level_3,
]


level_coro = levels[state.level](state)


def reset_level(state):
    global cam, level_coro
    rl.set_shader_value(
        skydome.materials[0].shader,
        invert_sky_loc,
        rl.ffi.new("int *", 0),
        rl.SHADER_UNIFORM_INT,
    )
    cam.position = (0, 0, 0)
    state.died_pos = glm.vec3(0)
    state.shark = None
    state.camspring.x = glm.vec3(0)
    state.enemies = []
    state.camspring.xp = glm.vec3(0)
    state.camspring.y = glm.vec3(0)
    state.camspring.yd = glm.vec3(0)
    state.pos = glm.vec3(0)
    state.vel = glm.vec3(0)
    state.pstate = "alive"
    state.water = 0.3
    state.bullet_i = 0
    state.invincible_time = 0
    rll.DetachAudioStreamProcessor(bg.stream, lowpass)
    level_coro = levels[state.level](state)
    next(level_coro)
    state.level_time = 0


PLAYER_RADIUS = 0.5
ENEMY_RADIUS = 1
BULLET_RADIUS = 0.1


def obstacle_bounding_box(y, z):
    if y == -1:
        LOW_OBSTACLE_DIMENSIONS = (40, 800, 4)
        return rl.BoundingBox(
            rl.Vector3(
                -20,
                -800,
                -LOW_OBSTACLE_DIMENSIONS[2] / 2 + z,
            ),
            rl.Vector3(
                20,
                -1,
                LOW_OBSTACLE_DIMENSIONS[2] / 2 + z,
            ),
        )
    elif y == 0:
        return rl.BoundingBox(
            rl.Vector3(-20, WATER_LEVEL, -2 + z), rl.Vector3(20, 1, 2 + z)
        )
    elif y == 1:
        return rl.BoundingBox(rl.Vector3(-20, 2, -2 + z), rl.Vector3(20, 10, 2 + z))


def depth_from_bb(bb):
    return bb.min.y + (bb.max.y - bb.min.y) / 2


def gen_cube_from_bb(bb):
    return rl.gen_mesh_cube(
        bb.max.x - bb.min.x, bb.max.y - bb.min.y, bb.max.z - bb.min.z
    )


obstacle_models = {
    -1.0: rl.load_model_from_mesh(gen_cube_from_bb(obstacle_bounding_box(-1, 0))),
    0.0: rl.load_model_from_mesh(gen_cube_from_bb(obstacle_bounding_box(0, 0))),
    1.0: rl.load_model_from_mesh(gen_cube_from_bb(obstacle_bounding_box(1, 0))),
}
obstacle_shader = rl.load_shader("obstacle.vert", "obstacle.frag")
for obstacle_model in obstacle_models.values():
    obstacle_model.materials[0].shader = obstacle_shader
    obstacle_shader.locs[rl.SHADER_LOC_MATRIX_MODEL] = rl.get_shader_location(
        obstacle_model.materials[0].shader, "matModel"
    )
obstacle_models[-1.0].materials[0].maps[0].color = rl.DARKGRAY
obstacle_models[0.0].materials[0].maps[0].color = rl.GRAY
obstacle_models[1.0].materials[0].maps[0].color = rl.WHITE
shark.materials[1].shader = obstacle_shader
shark.materials[2].shader = obstacle_shader
is_reflection_loc = rl.get_shader_location(obstacle_shader, "isReflection")


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


def collide_with_obstacle(state, pos, radius):
    for e in z_near(state.spike_obstacles, pos.z, radius + 2):
        if glm.distance(e, pos) < 2 + radius:
            return pos, 'spike'

    for p in state.enemy_bullets:
        if glm.distance(p, pos) < BULLET_RADIUS + radius:
            return pos, 'enemy_bullet'

    for p in z_near(state.obstacles, pos.z, radius + 4):
        if rl.check_collision_box_sphere(
            obstacle_bounding_box(p.y, p.z),
            pos.to_tuple(),
            radius,
        ):
            return glm.vec3(pos.xy, obstacle_bounding_box(p.y, p.z).max.z + 0.1), 'obstacle'

    if state.shark:
        for o, rad, cond in SHARK_WEAK_POINTS:
            if (cond is None or cond(state)) and glm.distance(state.shark.pos + o, pos) < rad + radius:
                return pos, 'shark'
        if glm.distance(state.shark.pos + glm.vec3(0, 0, -8), pos) < rad + 5:
            return pos, 'obstacle'

    return None, None


def render_scene(state, camera, interp_pos, reflected=False):
    rl.begin_mode_3d(camera)

    rll.rlDisableBackfaceCulling()
    rll.rlDisableDepthMask()
    rl.draw_model(skydome, (0, 0, 0), 1, rl.WHITE)
    rll.rlEnableBackfaceCulling()
    rll.rlEnableDepthMask()

    if reflected:
        rl.set_shader_value(
            obstacle_shader,
            is_reflection_loc,
            rl.ffi.new("int *", 1),
            rl.SHADER_UNIFORM_INT,
        )
    else:
        rl.set_shader_value(
            obstacle_shader,
            is_reflection_loc,
            rl.ffi.new("int *", 0),
            rl.SHADER_UNIFORM_INT,
        )

    if state.pstate == "alive":
        if not reflected or state.pos.y > WATER_LEVEL:
            if state.invincible_time <= 0 or (state.invincible_time % 0.2 < 0.1):
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
        bpm = 145
        if int(rl.get_music_time_played(bg) // (60 / bpm) % 4) == 0:
            scale = 1.1
        else:
            scale = 1
        rl.draw_mesh(
            spikeball,
            redmat,
            sum(
                glm.transpose(glm.translate(p) @ glm.scale(glm.vec3(scale))).to_tuple(),
                (),
            ),
        )

    for enemy in state.enemies:
        if enemy.state == "idle":
            rl.draw_mesh(
                enemyball,
                enemymat,
                sum(glm.transpose(glm.translate(enemy.pos)).to_tuple(), ()),
            )
            rl.draw_mesh(
                enemyspikes,
                enemymat,
                sum(
                    glm.transpose(
                        glm.translate(enemy.pos)
                        @ glm.rotate(rl.get_time() * enemy.rate, glm.vec3(0, 0, 1))
                    ).to_tuple(),
                    (),
                ),
            )
        elif enemy.state == "attack":
            rl.draw_mesh(
                enemyball,
                enemymat,
                sum(glm.transpose(glm.translate(enemy.pos)).to_tuple(), ()),
            )
            rl.draw_mesh(
                enemyspikes,
                redmat,
                sum(
                    glm.transpose(
                        glm.translate(enemy.pos) @ glm.scale(glm.vec3(2))
                    ).to_tuple(),
                    (),
                ),
            )
        else:
            assert False

    if state.died_pos != glm.vec3(0):
        rl.draw_sphere_wires(state.died_pos.to_tuple(), PLAYER_RADIUS, 4, 4, rl.BLACK)

    # SHARK

    if state.shark:
        rl.draw_mesh(jaw, shark.materials[1], sum(glm.transpose(glm.translate(state.shark.pos + glm.vec3(0, 0.5, 0)) @ glm.rotate(state.shark.jawspring.y, glm.vec3(1, 0, 0))).to_tuple(), ()))
        rl.draw_mesh(eyes, shark.materials[2], sum(glm.transpose(glm.translate(state.shark.pos + glm.vec3(0, 2, 0))).to_tuple(), ()))
        rl.draw_mesh(body, shark.materials[1], sum(glm.transpose(glm.translate(state.shark.pos + glm.vec3(0, 2, 0))).to_tuple(), ()))

    #for p, rad, _ in SHARK_WEAK_POINTS:
    #    rl.draw_sphere_wires((state.shark.pos + p).to_tuple(), rad, 8, 4, rl.PINK)

    # rl.draw_grid(50, 2)

    for p in state.bullets_pos:
        if glm.length2(p) != 0:
            rl.draw_sphere(p.to_tuple(), BULLET_RADIUS, rl.GREEN)

    for p in state.enemy_bullets:
        if not reflected or p.y > WATER_LEVEL:
            rl.draw_sphere(p.to_tuple(), BULLET_RADIUS * 3, rl.RED)

    for o in state.obstacles:
        rl.draw_model(
            obstacle_models[o.y],
            (0, depth_from_bb(obstacle_bounding_box(o.y, 0)), o.z),
            1,
            rl.WHITE,
        )

    # DEBUG
    # for o in lanes:
    #   rl.draw_line_3d((o, -3, state.pos.z), (o, -3, state.pos.z - 40), rl.WHITE)

    if not reflected:
        rl.draw_model(ocean_container, (0, WATER_LEVEL, state.pos.z), 1, rl.WHITE)
        rl.set_shader_value(
            water_plane.materials[0].shader,
            water_time_loc,
            rl.ffi.new("float *", rl.get_time()),
            rl.SHADER_UNIFORM_FLOAT,
        )
        rl.draw_model(
            water_plane,
            (0, WATER_LEVEL, state.pos.z - 180),
            1,
            rl.color_alpha(rl.BLUE, 0.4),
        )

    if not reflected:
        for i in range(20):
            rl.draw_sphere(
                (-20, WATER_LEVEL, (state.pos.z - 60 + i * 4) // 4 * 4), 0.1, rl.YELLOW
            )
            rl.draw_sphere(
                (20, WATER_LEVEL, (state.pos.z - 60 + i * 4) // 4 * 4), 0.1, rl.YELLOW
            )

    # rl.draw_plane(
    #    glm.vec3(0, WATER_LEVEL, state.pos.z - 170).to_tuple(),
    #    (40, 400),
    #    rl.color_alpha(rl.BLUE, 0.3),
    # )

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

    rl.draw_model(checkpoint_model, (0, 0, -state.checkpoint_dist), 1, rl.WHITE)

    rl.end_mode_3d()


flash = 0

def update(state):
    global flash

    state.level_time += rl.get_frame_time()

    rl.update_music_stream(bg)
    rl.update_music_stream(engine)

    # input
    inputv = glm.vec2()
    dive = 0
    rotation = 0

    rl.pause_music_stream(engine)

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
                (state.pos + glm.vec3(-0.3, -0.5, -0.1), state.vel + glm.vec3(0, -1, 0))
            )
            state.water_particles.append(
                (state.pos + glm.vec3(0.3, -0.5, -0.1), state.vel + glm.vec3(0, -1, 0))
            )
            if not rl.is_music_stream_playing(engine):
                rl.play_music_stream(engine)
    else:
        dive = 0


    interp_pos = state.pos

    waterlog_factor = 1 - (max(state.water - 1.5, 0) * 0.5)
    state.invincible_time -= rl.get_frame_time()

    if state.pstate == "alive":
        state.pos.y = dive
        interp_pos = glm.vec3(state.pos.x, state.yspring.update(dive), state.pos.z)
        if rl.is_key_down(rl.KEY_SPACE):
            if state.bullet_cooldown <= 0:
                state.bullets_pos[state.bullet_i] = glm.vec3(interp_pos)
                state.bullets_vel[state.bullet_i] = glm.vec3(
                    0, 0, state.vel.z
                ) + glm.vec3(0, 0, -0.4)
                state.bullet_i = (state.bullet_i + 1) % len(state.bullets_pos)
                state.bullet_cooldown = 0.1
                rl.set_sound_pitch(gun_sound, 1.0 + random.random() * 0.1 - 0.2)
                rl.play_sound(gun_sound)

        if glm.length(inputv) != 0:
            inputv = glm.normalize(inputv) * 0.1
            state.vel.x += inputv.x

        if inputv.y < 0 and state.water > 0:
            state.water_particles.append(
                (interp_pos + glm.vec3(0, -0.5, -0.5), state.vel + glm.vec3(0, 0, 1))
            )
            state.water -= 0.1 * rl.get_frame_time()
            state.vel.z += inputv.y * (1 if glm.length(state.vel.z) < 2 else 0.03)
            if not rl.is_music_stream_playing(engine):
                rl.play_music_stream(engine)
        elif inputv.y > 0:
            state.vel.z += inputv.y / 2
        else:
            state.vel.z *= 0.99

        if inputv.x == 0:
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

        state.pos += state.vel * rl.get_frame_time() * 50

        if glm.abs(state.pos.x) > 20:
            state.pos.x = glm.sign(state.pos.x) * 20
            state.vel.x = 0

        state.rotspring.update(rotation)
        state.bullet_cooldown -= rl.get_frame_time()

        if state.invincible_time < 0:
            died_pos, other = collide_with_obstacle(state, interp_pos, PLAYER_RADIUS)
            if died_pos:
                if state.armor > 1:
                    state.armor -= 1
                    state.invincible_time = 2
                else:
                    state.pstate = "dead"
                    state.died_pos = died_pos
                state.explosions.append((died_pos, rl.get_time()))
                rl.play_sound(explosion_sound)
                state.cam_shake_amount = 0.3

        if state.water > 2:
            state.pstate = "dead"
            state.died_pos = interp_pos
            state.explosions.append((interp_pos, rl.get_time()))

    elif state.pstate == "dead":
        if rl.is_key_pressed(rl.KEY_DOWN):
            reset_level(state)
            cam.position = (0, 4, 4)

    # cam update
    if state.pstate == "alive":
        camera_goal_pos = interp_pos + glm.vec3(0, glm.clamp(5 + state.vel.z * 1.5, 3, 10), 8)
    elif state.pstate == "dead":
        camera_goal_pos = state.pos + glm.vec3(0, -0.3 + state.camspring.y.y, 12)
    else:
        assert False
    state.camspring.update(camera_goal_pos)
    is_under_water_pre = cam.position.y < WATER_LEVEL
    state.cam_shake_amount = max(state.cam_shake_amount - rl.get_frame_time(), 0)
    cam_shake_vec =  glm.vec3(pnoise(rl.get_time() * 40) * 4,
                                                 pnoise(rl.get_time() * 40 + 2) * 4,
                                                 0) * state.cam_shake_amount
    cam.position = (state.camspring.y + cam_shake_vec
                    ).to_tuple()
    cam.target = (interp_pos + glm.vec3(0, 0, -4) + cam_shake_vec * 10).to_tuple()
    is_under_water = cam.position.y < WATER_LEVEL
    if not is_under_water_pre and is_under_water:
        rll.AttachAudioStreamProcessor(bg.stream, lowpass)
    if is_under_water_pre and not is_under_water:
        rll.DetachAudioStreamProcessor(bg.stream, lowpass)

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
            rl.ffi.new("float *", 0.4),
            rl.SHADER_UNIFORM_FLOAT,
        )
        rl.set_shader_value(
            fog_shader,
            FOG_COLOR_LOC,
            rl.ffi.new("float[3]", [12 / 255, 20 / 255, 86 / 255]),
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
            rl.ffi.new("float[3]", [0.1, 0.1, 0.2]),
            rl.SHADER_UNIFORM_VEC3,
        )

    if state.pstate == "alive":
        cam.fovy = (
            45
            + (
                glm.clamp(glm.distance(
                    interp_pos, (cam.position.x, cam.position.y, cam.position.z)
                ), 4, 22)
                / 22
            )
            * 20
        )

    LANE_WIDTH = 4
    lanes = []
    # update enemies
    if state.enemies:
        lanes = [x for x in range(-20, 20, LANE_WIDTH)]
        for obstacle in z_near(
            state.spike_obstacles,
            max(state.enemies, key=lambda e: e.pos.z).pos.z - 10,
            20,
        ):
            for off in (-2, 0, 2):
                try:
                    lanes.remove(
                        math.floor((obstacle.x + off) / LANE_WIDTH) * LANE_WIDTH
                    )
                except ValueError:
                    pass

        available_lanes = list(lanes)
        start_i = (
            len(available_lanes) // 2 - (len(state.enemies) // 2)
            if len(available_lanes) > len(state.enemies)
            else 0
        )
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
                    rl.play_sound(explosion_sound)
                    del state.enemies[i]

            enemy.state_time -= rl.get_frame_time()
            if enemy.state == "idle":
                if enemy.state_time < 0:
                    enemy.state = "attack"
                    enemy.state_time = 0.5
            elif enemy.state == "attack":
                if enemy.state_time < 0:
                    for i in range(5):
                        state.enemy_bullets.append(enemy.pos + glm.vec3(0, 0, 0.5) * i)
                    rl.play_sound(enemy_shot)
                    enemy.state = "idle"
                    enemy.state_time = 2 + random.random()

    # update shark
    if state.shark:
        state.shark.loop_timer += rl.get_frame_time()
        if state.shark.hitpoints > 50:
            state.shark.phase = 0
        elif state.shark.hitpoints > 20:
            state.shark.phase = 1
        elif state.shark.hitpoints > 0:
            state.shark.phase = 2
        else:
            if state.shark.phase != 3:
                for i in range(30):
                    state.explosions.append(
                        (state.shark.pos + glm.vec3(random.random() * 5,
                                                    random.random() * 5,
                                                    4 + random.random() * 5,
                                                    ),
                         rl.get_time()))
                    rl.play_sound(explosion_sound)

            state.shark.phase = 3

        if state.shark.phase == 0:
            goal_pos = glm.vec3(state.shark.pos.x, 0, state.pos.z) + glm.vec3(0, 0, -40)
            if state.shark.loop_timer % 10 < 4:
                goal_pos.y = -8
            else:
                if state.shark.pos.y > WATER_LEVEL:
                    goal_pos.x = state.pos.x
                goal_pos.y = 0

            if 7 < state.shark.loop_timer % 10 < 9:
                state.shark.jawspring.update(0.5)
                shark.materials[2].maps[0].color = (255, 0, 0)
                if not state.shark.fired:
                    state.shark.fired = True
                    rl.play_sound(enemy_shot)
                    for i in range(5):
                        state.enemy_bullets.append(state.shark.pos + glm.vec3(0, 0, 0.5) * i)
            else:
                state.shark.fired = False
                state.shark.jawspring.update(0)
                shark.materials[2].maps[0].color = (100, 0, 0)
        elif state.shark.phase == 1:
            goal_pos = glm.vec3(state.shark.pos.x, 0, state.pos.z) + glm.vec3(0, 0, -40)
            if state.shark.loop_timer % 5 < 2:
                goal_pos.y = -8
            else:
                if state.shark.pos.y > WATER_LEVEL:
                    goal_pos.x = state.pos.x
                goal_pos.y = 0

            if 3 < state.shark.loop_timer % 5 < 4:
                state.shark.jawspring.update(0.5)
                shark.materials[2].maps[0].color = (255, 0, 0)
                if not state.shark.fired:
                    rl.play_sound(enemy_shot)
                    state.shark.fired = True
                    for i in range(5):
                        state.enemy_bullets.append(state.shark.pos + glm.vec3(0, 0, 0.5) * i)
            else:
                state.shark.fired = False
                state.shark.jawspring.update(0)
                shark.materials[2].maps[0].color = (100, 0, 0)
        elif state.shark.phase == 2:
            goal_pos = glm.vec3(state.shark.pos.x, 0, state.pos.z) + glm.vec3(0, 0, -40)
            tstep = int(state.shark.loop_timer % 20)
            state.shark.jawspring.update(0.7)
            if tstep < 10:
                i = tstep
                goal_pos.y =  0
            else:
                i = 20 - tstep
                goal_pos.y = -6
            if tstep != state.shark.step:
                state.shark.step = tstep
                rl.play_sound(enemy_shot)
                for i in range(5):
                    state.enemy_bullets.append(state.shark.pos + glm.vec3(0, 0, 0.5) * i)
                    state.shark.fired = True
            goal_pos.x = (40 / 10) * i - 20
        elif state.shark.phase == 3:
            goal_pos = state.shark.pos + glm.vec3(0, -1, 0)


        state.shark.shake_amount = max(0, state.shark.shake_amount - rl.get_frame_time())
        state.shark.pos = state.shark.posspring.update(goal_pos) + glm.vec3(
            pnoise(rl.get_time() * 20),
            pnoise(rl.get_time() * 20),
            pnoise(rl.get_time() * 20)) * state.shark.shake_amount


    # update bullets
    for i, pos in enumerate(state.enemy_bullets):
        pos.z += 0.3
        if pos.z > cam.position.z:
            del state.enemy_bullets[0]
    state.bullets_pos += state.bullets_vel
    for i, bullet in enumerate(state.bullets_pos):
        collide, other = collide_with_obstacle(state, bullet, BULLET_RADIUS)
        if other == 'shark':
            state.explosions.append((bullet, rl.get_time()))
            state.shark.shake_amount = 1
            state.shark.hitpoints -= 1

        for i, e in enumerate(state.enemies):
            if glm.distance(bullet, e.pos) < BULLET_RADIUS + ENEMY_RADIUS:
                state.explosions.append((bullet, rl.get_time()))
                rl.play_sound(explosion_sound)
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
    reflect_cam.position.y = -cam.position.y + WATER_LEVEL * 2
    reflect_cam.target.y = -cam.target.y + WATER_LEVEL * 2
    render_scene(state, reflect_cam, interp_pos, reflected=True)
    rl.end_texture_mode()

    rl.begin_texture_mode(render_target)
    rl.clear_background(rl.PURPLE)

    render_scene(state, cam, interp_pos)

    rl.end_texture_mode()

    rl.clear_background(rl.BLACK)

    rl.begin_shader_mode(psx_shader)
    # rl.set_shader_value_texture(psx_shader, rl.get_shader_location(psx_shader, "texture1"), render_target.depth)
    rl.draw_texture_pro(
        render_target.texture,
        rl.Rectangle(0, 0, render_target.texture.width, -render_target.texture.height),
        rl.Rectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT),
        rl.Vector2(0, 0),
        0,
        rl.SKYBLUE if is_under_water else rl.WHITE,
    )
    if flash > 0:
        rl.draw_rectangle(
            0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, rl.color_alpha(rl.WHITE, flash)
        )
        flash -= rl.get_frame_time()
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
            rl.BLUE,
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
            rl.GRAY,
        )
        rl.draw_text(
            "ARMR", SCREEN_WIDTH // 2 + 40, SCREEN_HEIGHT // 2, 5, rl.DARKGREEN
        )
        for i in range(3):
            if state.armor > i:
                rl.draw_rectangle(
                    SCREEN_WIDTH // 2 + 40 + 31 + i * 15,
                    SCREEN_HEIGHT // 2 + 1,
                    10,
                    7,
                    rl.DARKGREEN,
                )
            else:
                rl.draw_rectangle(
                    SCREEN_WIDTH // 2 + 40 + 31 + i * 15,
                    SCREEN_HEIGHT // 2 + 1,
                    10,
                    7,
                    rl.color_alpha(rl.DARKGREEN, 0.2),
                )

    if state.pstate == "dead":
        rl.draw_text(
            "Press down arrow to restart", 10, int(SCREEN_HEIGHT / 2), 50, rl.WHITE
        )

    if state.water - 1 > 0.1:
        if (rl.get_time() % 0.4) < 0.2:
            x = SCREEN_WIDTH // 2 + 40
            rl.draw_text(
                "WARNING: WATERLOGGED", x, SCREEN_HEIGHT // 2 - 120, 30, rl.RED
            )
            if waterlog_factor < 1:
                rl.draw_text(
                    f"{int((2-state.water)*100)}% capacity",
                    x,
                    SCREEN_HEIGHT // 2 - 90,
                    30,
                    rl.RED,
                )
    elif state.water < 0.05:
        rl.draw_text(
            "WATER LOW - DIVE TO FILL TANK",
            SCREEN_WIDTH // 2 + 40,
            SCREEN_HEIGHT // 2 - 120,
            20,
            rl.BLUE,
        )

    # level management
    global level_coro
    try:
        next(level_coro)
    except StopIteration:
        state.level += 1
        reset_level(state)
    rl.draw_fps(10, 10)

if __name__ == "__main__":
    while not rl.window_should_close():
        rl.begin_drawing()
        rl.clear_background(rl.BLACK)

        update(state)
        rl.end_drawing()

    rl.close_window()
