import pyray as rl
from importlib import reload
import traceback as tb
import copy

rl.init_window(320 * 3, 240 * 3, 'game [debug]')
rl.init_audio_device()
rl.set_target_fps(60)

import game

orig_state = copy.deepcopy(game.state)
state = copy.deepcopy(game.state)
error = None

while not rl.window_should_close():
    rl.begin_drawing()
    rl.clear_background(rl.BLACK)

    if rl.is_key_pressed(rl.KEY_R):
        print('reset')
        state = copy.deepcopy(orig_state)
        reload(game)
        game.reinit(state)
        error = None

    if error is None:
        try:
            game.update(state)
        except Exception as e:
            error = e
            print(''.join(tb.format_exception(None, error, error.__traceback__)))
    else:
        rl.end_shader_mode()
        rl.draw_text(''.join(tb.format_exception(None, error, error.__traceback__)), 10, 40, 10, rl.RED)

    rl.draw_fps(10, 10)
    rl.end_drawing()

rl.close_window()
