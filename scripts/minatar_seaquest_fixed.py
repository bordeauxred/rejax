"""
Fixed JAX-compatible Seaquest MinAtar environment.

Adapted from pgx-minatar implementation by Sotetsu KOYAMADA:
https://github.com/sotetsuk/pgx-minatar/blob/main/seaquest.py

Original MinAtar by Kenny Young and Tian Tian (GNU GPL v3.0)
"""
import jax
import jax.numpy as jnp
import jax.lax as lax
from flax import struct
from gymnax.environments import environment, spaces

# Game constants
RAMP_INTERVAL = jnp.int32(100)
MAX_OXYGEN = jnp.int32(200)
INIT_SPAWN_SPEED = jnp.int32(20)
DIVER_SPAWN_SPEED = jnp.int32(30)
INIT_MOVE_INTERVAL = jnp.int32(5)
SHOT_COOL_DOWN = jnp.int32(5)
ENEMY_SHOT_INTERVAL = jnp.int32(10)
ENEMY_MOVE_INTERVAL = jnp.int32(5)
DIVER_MOVE_INTERVAL = jnp.int32(5)

ZERO = jnp.int32(0)
NINE = jnp.int32(9)
TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


@struct.dataclass
class EnvState(environment.EnvState):
    oxygen: jnp.int32
    diver_count: jnp.int32
    sub_x: jnp.int32
    sub_y: jnp.int32
    sub_or: jnp.bool_
    f_bullets: jax.Array  # (5, 3) - x, y, direction
    e_bullets: jax.Array  # (25, 3)
    e_fish: jax.Array     # (25, 4) - x, y, direction, timer
    e_subs: jax.Array     # (25, 5) - x, y, direction, move_timer, shot_timer
    divers: jax.Array     # (5, 4) - x, y, direction, timer
    e_spawn_speed: jnp.int32
    e_spawn_timer: jnp.int32
    d_spawn_timer: jnp.int32
    move_speed: jnp.int32
    ramp_index: jnp.int32
    shot_timer: jnp.int32
    surface: jnp.bool_
    terminal: jnp.bool_
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = 1000


class MinSeaquestFixed(environment.Environment[EnvState, EnvParams]):
    """JAX-compatible Seaquest MinAtar environment."""

    def __init__(self, use_minimal_action_set: bool = True):
        super().__init__()
        self.obs_shape = (10, 10, 10)
        self.action_set = jnp.array([0, 1, 2, 3, 4, 5])

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self, key: jax.Array, state: EnvState, action: int, params: EnvParams
    ):
        keys = jax.random.split(key, 5)
        action = jnp.int32(action)

        enemy_lr = jax.random.choice(keys[0], jnp.array([True, False]))
        is_sub = jax.random.choice(keys[1], jnp.array([True, False]), p=jnp.array([1/3, 2/3]))
        enemy_y = jax.random.choice(keys[2], jnp.arange(1, 9))
        diver_lr = jax.random.choice(keys[3], jnp.array([True, False]))
        diver_y = jax.random.choice(keys[4], jnp.arange(1, 9))

        state, reward = _step_det(state, action, enemy_lr, is_sub, enemy_y, diver_lr, diver_y)
        state = state.replace(time=state.time + 1)
        done = self.is_terminal(state, params)
        return self.get_obs(state, params), state, reward, done, {"discount": self.discount(state, params)}

    def reset_env(self, key: jax.Array, params: EnvParams):
        state = EnvState(
            oxygen=MAX_OXYGEN,
            diver_count=ZERO,
            sub_x=jnp.int32(5),
            sub_y=ZERO,
            sub_or=FALSE,
            f_bullets=-jnp.ones((5, 3), dtype=jnp.int32),
            e_bullets=-jnp.ones((25, 3), dtype=jnp.int32),
            e_fish=-jnp.ones((25, 4), dtype=jnp.int32),
            e_subs=-jnp.ones((25, 5), dtype=jnp.int32),
            divers=-jnp.ones((5, 4), dtype=jnp.int32),
            e_spawn_speed=INIT_SPAWN_SPEED,
            e_spawn_timer=INIT_SPAWN_SPEED,
            d_spawn_timer=DIVER_SPAWN_SPEED,
            move_speed=INIT_MOVE_INTERVAL,
            ramp_index=ZERO,
            shot_timer=ZERO,
            surface=TRUE,
            terminal=FALSE,
            time=0,
        )
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        """Render observation - uses masks instead of dynamic slicing."""
        obs = jnp.zeros((11, 11, 10), dtype=jnp.float32)

        # Submarine front and back
        obs = obs.at[state.sub_y, state.sub_x, 0].set(1.0)
        back_x = lax.cond(state.sub_or, lambda: state.sub_x - 1, lambda: state.sub_x + 1)
        obs = obs.at[state.sub_y, back_x, 1].set(1.0)

        # Oxygen gauge - use mask instead of dynamic slice
        oxygen_level = state.oxygen * 10 // MAX_OXYGEN
        oxygen_level = lax.cond(state.oxygen < 0, lambda: jnp.int32(9), lambda: oxygen_level)
        oxygen_mask = jnp.arange(11) < oxygen_level
        obs = obs.at[9, :, 7].set(jnp.where(oxygen_mask, 1.0, obs[9, :, 7]))

        # Diver gauge - use mask instead of dynamic slice
        diver_mask = (9 - state.diver_count <= jnp.arange(11)) & (jnp.arange(11) < 9)
        obs = obs.at[9, :, 8].set(jnp.where(diver_mask, 1.0, obs[9, :, 8]))

        # Friendly bullets
        obs = obs.at[state.f_bullets[:, 1], state.f_bullets[:, 0], 2].set(1.0)

        # Enemy bullets
        obs = obs.at[state.e_bullets[:, 1], state.e_bullets[:, 0], 4].set(1.0)

        # Enemy fish and trails
        obs = obs.at[state.e_fish[:, 1], state.e_fish[:, 0], 5].set(1.0)
        fish_back_x = state.e_fish[:, 0] + jnp.array([1, -1], dtype=jnp.int32)[state.e_fish[:, 2].astype(jnp.int32)]
        obs = obs.at[state.e_fish[:, 1], fish_back_x, 3].set(1.0)

        # Enemy subs and trails
        obs = obs.at[state.e_subs[:, 1], state.e_subs[:, 0], 6].set(1.0)
        sub_back_x = state.e_subs[:, 0] + jnp.array([1, -1], dtype=jnp.int32)[state.e_subs[:, 2].astype(jnp.int32)]
        obs = obs.at[state.e_subs[:, 1], sub_back_x, 3].set(1.0)

        # Divers and trails
        obs = obs.at[state.divers[:, 1], state.divers[:, 0], 9].set(1.0)
        diver_back_x = state.divers[:, 0] + jnp.array([1, -1], dtype=jnp.int32)[state.divers[:, 2].astype(jnp.int32)]
        obs = obs.at[state.divers[:, 1], diver_back_x, 3].set(1.0)

        return obs[:10, :10, :]

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.bool_:
        done_steps = state.time >= params.max_steps_in_episode
        return jnp.logical_or(state.terminal, done_steps)

    @property
    def name(self) -> str:
        return "Seaquest-MinAtar"

    @property
    def num_actions(self) -> int:
        return 6

    def action_space(self, params: EnvParams = None) -> spaces.Discrete:
        return spaces.Discrete(6)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(0, 1, self.obs_shape, dtype=jnp.float32)


# Helper functions
def _find_ix(arr):
    """Find first empty slot (marked by -1)."""
    return (arr[:, 0] == -1).argmax()


def _is_filled(row):
    return jnp.any(row != -1)


def _is_out(row):
    return (row[0] < 0) | (row[0] > 9)


def _is_hit(row, x, y):
    return (row[0] == x) & (row[1] == y)


def _remove_i(arr, i):
    """Remove element at index i by shifting."""
    mask = jnp.tile(jnp.arange(arr.shape[0]) < i, (arr.shape[1], 1)).T
    rolled = jnp.roll(arr, -1, axis=0)
    return jnp.where(mask, arr, rolled).at[-1, :].set(-1)


def _step_det(state, action, enemy_lr, is_sub, enemy_y, diver_lr, diver_y):
    """Deterministic step function."""
    ramping = TRUE

    oxygen = state.oxygen
    diver_count = state.diver_count
    sub_x = state.sub_x
    sub_y = state.sub_y
    sub_or = state.sub_or
    f_bullets = state.f_bullets
    e_bullets = state.e_bullets
    e_fish = state.e_fish
    e_subs = state.e_subs
    divers = state.divers
    e_spawn_speed = state.e_spawn_speed
    e_spawn_timer = state.e_spawn_timer
    d_spawn_timer = state.d_spawn_timer
    move_speed = state.move_speed
    ramp_index = state.ramp_index
    shot_timer = state.shot_timer
    surface = state.surface
    terminal = state.terminal

    r = jnp.float32(0)

    # Spawn enemy if timer is up
    e_subs, e_fish = lax.cond(
        e_spawn_timer == 0,
        lambda: _spawn_enemy(e_subs, e_fish, move_speed, enemy_lr, is_sub, enemy_y),
        lambda: (e_subs, e_fish),
    )
    e_spawn_timer = lax.cond(e_spawn_timer == 0, lambda: e_spawn_speed, lambda: e_spawn_timer)

    # Spawn diver if timer is up
    divers, d_spawn_timer = lax.cond(
        d_spawn_timer == 0,
        lambda: (_spawn_diver(divers, diver_lr, diver_y), DIVER_SPAWN_SPEED),
        lambda: (divers, d_spawn_timer),
    )

    # Resolve player action
    f_bullets, shot_timer, sub_x, sub_y, sub_or = _resolve_action(
        action, shot_timer, f_bullets, sub_x, sub_y, sub_or
    )

    # Update friendly bullets
    f_bullets, e_subs, e_fish, r = _update_friendly_bullets(f_bullets, e_subs, e_fish, r)

    # Update divers
    divers, diver_count = _update_divers(divers, diver_count, sub_x, sub_y)

    # Update enemy subs
    f_bullets, e_subs, e_bullets, terminal, r = _update_enemy_subs(
        f_bullets, e_subs, e_bullets, sub_x, sub_y, move_speed, terminal, r
    )

    # Update enemy bullets
    e_bullets, terminal = _update_enemy_bullets(e_bullets, sub_x, sub_y, terminal)

    # Update enemy fish
    f_bullets, e_fish, terminal, r = _update_enemy_fish(
        f_bullets, e_fish, sub_x, sub_y, move_speed, terminal, r
    )

    # Update timers
    e_spawn_timer = lax.cond(e_spawn_timer > 0, lambda: e_spawn_timer - 1, lambda: e_spawn_timer)
    d_spawn_timer = lax.cond(d_spawn_timer > 0, lambda: d_spawn_timer - 1, lambda: d_spawn_timer)
    shot_timer = lax.cond(shot_timer > 0, lambda: shot_timer - 1, lambda: shot_timer)

    # Oxygen and surface logic
    terminal = terminal | (oxygen < 0)
    tmp = surface
    oxygen = lax.cond(sub_y > 0, lambda: oxygen - 1, lambda: oxygen)
    surface = lax.cond(sub_y > 0, lambda: FALSE, lambda: surface)
    terminal = lax.cond((sub_y <= 0) & ~tmp & (diver_count == 0), lambda: TRUE, lambda: terminal)
    surface = surface | ((sub_y <= 0) & ~tmp & (diver_count != 0))

    _r, oxygen, diver_count, move_speed, e_spawn_speed, ramp_index = lax.cond(
        (sub_y <= 0) & ~tmp & (diver_count != 0),
        lambda: _surface(diver_count, oxygen, e_spawn_speed, move_speed, ramping, ramp_index),
        lambda: (jnp.int32(0), oxygen, diver_count, move_speed, e_spawn_speed, ramp_index),
    )
    r = r + _r

    new_state = state.replace(
        oxygen=oxygen,
        diver_count=diver_count,
        sub_x=sub_x,
        sub_y=sub_y,
        sub_or=sub_or,
        f_bullets=f_bullets,
        e_bullets=e_bullets,
        e_fish=e_fish,
        e_subs=e_subs,
        divers=divers,
        e_spawn_speed=e_spawn_speed,
        e_spawn_timer=e_spawn_timer,
        d_spawn_timer=d_spawn_timer,
        move_speed=move_speed,
        ramp_index=ramp_index,
        shot_timer=shot_timer,
        surface=surface,
        terminal=terminal,
    )
    return new_state, r


def _resolve_action(action, shot_timer, f_bullets, sub_x, sub_y, sub_or):
    # Fire bullet
    f_bullets, shot_timer = lax.cond(
        (action == 5) & (shot_timer == 0),
        lambda: (
            f_bullets.at[_find_ix(f_bullets)].set(jnp.int32([sub_x, sub_y, sub_or])),
            SHOT_COOL_DOWN,
        ),
        lambda: (f_bullets, shot_timer),
    )
    # Move left
    sub_x, sub_or = lax.cond(action == 1, lambda: (lax.max(ZERO, sub_x - 1), FALSE), lambda: (sub_x, sub_or))
    # Move right
    sub_x, sub_or = lax.cond(action == 3, lambda: (lax.min(NINE, sub_x + 1), TRUE), lambda: (sub_x, sub_or))
    # Move up
    sub_y = lax.cond(action == 2, lambda: lax.max(ZERO, sub_y - 1), lambda: sub_y)
    # Move down
    sub_y = lax.cond(action == 4, lambda: lax.min(jnp.int32(8), sub_y + 1), lambda: sub_y)
    return f_bullets, shot_timer, sub_x, sub_y, sub_or


def _update_by_f_bullets_hit(j, f_bullets, e):
    is_hit = (f_bullets[j, 0] == e[:, 0]) & (f_bullets[j, 1] == e[:, 1])
    k = jnp.argmax(is_hit)
    k = jax.lax.select(jnp.sum(is_hit) == 0, 25, k)
    f_bullets, e, removed = lax.cond(
        k < 25,
        lambda: (_remove_i(f_bullets, j), _remove_i(e, k), TRUE),
        lambda: (f_bullets, e, FALSE),
    )
    return f_bullets, e, removed


def _update_friendly_bullets(f_bullets, e_subs, e_fish, r):
    def _remove(j, _f_bullets, _e_subs, _e_fish, _r):
        _f_bullets, _e_fish, fish_removed = _update_by_f_bullets_hit(j, _f_bullets, _e_fish)
        _r = _r + fish_removed
        _f_bullets, _e_subs, sub_removed = lax.cond(
            fish_removed,
            lambda: (_f_bullets, _e_subs, FALSE),
            lambda: _update_by_f_bullets_hit(j, _f_bullets, _e_subs),
        )
        _r = _r + sub_removed
        return _f_bullets, _e_subs, _e_fish, _r

    def _update_each(i, x):
        _f_bullets, _e_subs, _e_fish, _r = x
        j = 5 - i - 1
        is_filled = _is_filled(_f_bullets[j])
        _f_bullets = lax.cond(
            is_filled,
            lambda: _f_bullets.at[j, 0].add(lax.cond(_f_bullets[j, 2], lambda: 1, lambda: -1)),
            lambda: _f_bullets,
        )
        _f_bullets, _e_subs, _e_fish, _r = lax.cond(
            is_filled,
            lambda: lax.cond(
                _is_out(_f_bullets[j]),
                lambda: (_remove_i(_f_bullets, j), _e_subs, _e_fish, _r),
                lambda: _remove(j, _f_bullets, _e_subs, _e_fish, _r),
            ),
            lambda: (_f_bullets, _e_subs, _e_fish, _r),
        )
        return _f_bullets, _e_subs, _e_fish, _r

    f_bullets, e_subs, e_fish, r = lax.fori_loop(0, 5, _update_each, (f_bullets, e_subs, e_fish, r))
    return f_bullets, e_subs, e_fish, r


def _update_divers(divers, diver_count, sub_x, sub_y):
    def _update_by_move(_divers, _diver_count, j):
        _divers = _divers.at[j, 3].set(DIVER_MOVE_INTERVAL)
        _divers = _divers.at[j, 0].add(lax.cond(_divers[j, 2], lambda: 1, lambda: -1))
        _divers, _diver_count = lax.cond(
            _is_out(_divers[j]),
            lambda: (_remove_i(_divers, j), _diver_count),
            lambda: lax.cond(
                _is_hit(_divers[j], sub_x, sub_y) & (_diver_count < 6),
                lambda: (_remove_i(_divers, j), _diver_count + 1),
                lambda: (_divers, _diver_count),
            ),
        )
        return _divers, _diver_count

    def _update_each(i, x):
        _divers, _diver_count = x
        j = 5 - i - 1
        return lax.cond(
            _is_filled(_divers[j]),
            lambda: lax.cond(
                _is_hit(_divers[j], sub_x, sub_y) & (_diver_count < 6),
                lambda: (_remove_i(_divers, j), _diver_count + 1),
                lambda: lax.cond(
                    _divers[j, 3] == 0,
                    lambda: _update_by_move(_divers, _diver_count, j),
                    lambda: (_divers.at[j, 3].add(-1), _diver_count),
                ),
            ),
            lambda: (_divers, _diver_count),
        )

    divers, diver_count = lax.fori_loop(0, 5, _update_each, (divers, diver_count))
    return divers, diver_count


def _update_by_hit(j, f_bullets, e):
    is_hit = (e[j, 0] == f_bullets[:, 0]) & (e[j, 1] == f_bullets[:, 1])
    k = jnp.argmax(is_hit)
    k = jax.lax.select(jnp.sum(is_hit) == 0, 5, k)
    f_bullets, e, removed = lax.cond(
        k < 5,
        lambda: (_remove_i(f_bullets, k), _remove_i(e, j), TRUE),
        lambda: (f_bullets, e, FALSE),
    )
    return f_bullets, e, removed


def _update_enemy_subs(f_bullets, e_subs, e_bullets, sub_x, sub_y, move_speed, terminal, r):
    def _update_sub(j, _f_bullets, _e_subs, _terminal, _r):
        _e_subs = _e_subs.at[j, 3].set(move_speed)
        _e_subs = _e_subs.at[j, 0].add(lax.cond(_e_subs[j, 2], lambda: 1, lambda: -1))
        is_out = _is_out(_e_subs[j])
        is_hit = _is_hit(_e_subs[j], sub_x, sub_y)
        _e_subs = lax.cond(is_out, lambda: _remove_i(_e_subs, j), lambda: _e_subs)
        _terminal = lax.cond(~is_out & is_hit, lambda: TRUE, lambda: _terminal)
        _f_bullets, _e_subs, removed = lax.cond(
            ~is_out & ~is_hit,
            lambda: _update_by_hit(j, _f_bullets, _e_subs),
            lambda: (_f_bullets, _e_subs, FALSE),
        )
        _r = _r + removed
        return (removed | is_out), _f_bullets, _e_subs, _terminal, _r

    def _update_each_filled(j, x):
        _f_bullets, _e_subs, _e_bullets, _terminal, _r = x
        _terminal = _terminal | _is_hit(_e_subs[j], sub_x, sub_y)
        removed, _f_bullets, _e_subs, _terminal, _r = lax.cond(
            _e_subs[j, 3] == 0,
            lambda: _update_sub(j, _f_bullets, _e_subs, _terminal, _r),
            lambda: (FALSE, _f_bullets, _e_subs.at[j, 3].add(-1), _terminal, _r),
        )
        timer_zero = _e_subs[j, 4] == 0
        _e_subs = lax.cond(
            removed,
            lambda: _e_subs,
            lambda: lax.cond(
                timer_zero,
                lambda: _e_subs.at[j, 4].set(ENEMY_SHOT_INTERVAL),
                lambda: _e_subs.at[j, 4].add(-1),
            ),
        )
        _e_bullets = lax.cond(
            removed,
            lambda: _e_bullets,
            lambda: lax.cond(
                timer_zero,
                lambda: _e_bullets.at[_find_ix(_e_bullets)].set(
                    jnp.int32([_e_subs[j, 0], _e_subs[j, 1], _e_subs[j, 2]])
                ),
                lambda: _e_bullets,
            ),
        )
        return _f_bullets, _e_subs, _e_bullets, _terminal, _r

    def _update_each(i, x):
        j = 25 - i - 1
        return lax.cond(_is_filled(x[1][j]), lambda: _update_each_filled(j, x), lambda: x)

    f_bullets, e_subs, e_bullets, terminal, r = lax.fori_loop(
        0, 25, _update_each, (f_bullets, e_subs, e_bullets, terminal, r)
    )
    return f_bullets, e_subs, e_bullets, terminal, r


def _step_obj(arr, ix):
    arr_p = arr.at[:, 0].add(1)
    arr_m = arr.at[:, 0].add(-1)
    arr_2 = jnp.where(jnp.tile(arr[:, 2], reps=(arr.shape[1], 1)).T, arr_p, arr_m)
    arr = jnp.where(jnp.tile(jnp.arange(arr.shape[0]) < ix, reps=(arr.shape[1], 1)).T, arr_2, arr)
    return arr


def _remove_out_of_bound(arr, ix):
    def body(i, a):
        return lax.cond((a[i][0] < 0) | (a[i][0] > 9), lambda: _remove_i(a, i), lambda: a)
    return lax.fori_loop(0, ix, body, arr)


def _hit(arr, ix, x, y):
    return ((arr[:, 0] == x) & (arr[:, 1] == y) & (jnp.arange(arr.shape[0]) < ix)).any()


def _update_enemy_bullets(e_bullets, sub_x, sub_y, terminal):
    ix = _find_ix(e_bullets)
    terminal = terminal | _hit(e_bullets, ix, sub_x, sub_y)
    e_bullets = _step_obj(e_bullets, ix)
    e_bullets = _remove_out_of_bound(e_bullets, ix)
    terminal = terminal | _hit(e_bullets, ix, sub_x, sub_y)
    return e_bullets, terminal


def _update_enemy_fish(f_bullets, e_fish, sub_x, sub_y, move_speed, terminal, r):
    def _update_by_hit_fish(j, _f_bullets, e, _terminal, _r):
        _f_bullets, e, removed = _update_by_hit(j, _f_bullets, e)
        return _f_bullets, e, _terminal, _r + removed

    def _update_fish(j, _f_bullets, _e_fish, _terminal, _r):
        _e_fish = _e_fish.at[j, 3].set(move_speed)
        _e_fish = _e_fish.at[j, 0].add(lax.cond(_e_fish[j, 2], lambda: 1, lambda: -1))
        _f_bullets, _e_fish, _terminal, _r = lax.cond(
            _is_out(_e_fish[j]),
            lambda: (_f_bullets, _remove_i(_e_fish, j), _terminal, _r),
            lambda: lax.cond(
                _is_hit(_e_fish[j], sub_x, sub_y),
                lambda: (_f_bullets, _e_fish, TRUE, _r),
                lambda: _update_by_hit_fish(j, _f_bullets, _e_fish, _terminal, _r),
            ),
        )
        return _f_bullets, _e_fish, _terminal, _r

    def _update_each(i, x):
        j = 25 - i - 1
        _f_bullets, _e_fish, _terminal, _r = x
        _terminal = _terminal | _is_hit(_e_fish[j], sub_x, sub_y)
        _f_bullets, _e_fish, _terminal, _r = lax.cond(
            _is_filled(_e_fish[j]),
            lambda: lax.cond(
                _e_fish[j, 3] == 0,
                lambda: _update_fish(j, _f_bullets, _e_fish, _terminal, _r),
                lambda: (_f_bullets, _e_fish.at[j, 3].add(-1), _terminal, _r),
            ),
            lambda: (_f_bullets, _e_fish, _terminal, _r),
        )
        return _f_bullets, _e_fish, _terminal, _r

    f_bullets, e_fish, terminal, r = lax.fori_loop(0, 25, _update_each, (f_bullets, e_fish, terminal, r))
    return f_bullets, e_fish, terminal, r


def _surface(diver_count, oxygen, e_spawn_speed, move_speed, ramping, ramp_index):
    diver_count, r = lax.cond(
        diver_count == 6,
        lambda: (ZERO, oxygen * 10 // MAX_OXYGEN),
        lambda: (diver_count, jnp.int32(0)),
    )
    oxygen = MAX_OXYGEN
    diver_count = diver_count - 1
    ramp_update = ramping & ((e_spawn_speed > 1) | (move_speed > 2))
    ramp_index = lax.cond(ramp_update, lambda: ramp_index + 1, lambda: ramp_index)
    move_speed = lax.cond(
        ramp_update & ((move_speed > 2) & (ramp_index % 2 == 1)),
        lambda: move_speed - 1,
        lambda: move_speed,
    )
    e_spawn_speed = lax.cond(
        ramp_update & (e_spawn_speed > 1),
        lambda: e_spawn_speed - 1,
        lambda: e_spawn_speed,
    )
    return r, oxygen, diver_count, move_speed, e_spawn_speed, ramp_index


def _spawn_enemy(e_subs, e_fish, move_speed, enemy_lr, is_sub, enemy_y):
    x = lax.cond(enemy_lr, lambda: ZERO, lambda: NINE)
    has_collision = ((e_subs[:, 1] == enemy_y) & (e_subs[:, 2] != enemy_lr)).sum() > 0
    has_collision = has_collision | (((e_fish[:, 1] == enemy_y) & (e_fish[:, 2] != enemy_lr)).sum() > 0)
    return lax.cond(
        has_collision,
        lambda: (e_subs, e_fish),
        lambda: lax.cond(
            is_sub,
            lambda: (
                e_subs.at[_find_ix(e_subs)].set(jnp.int32([x, enemy_y, enemy_lr, move_speed, ENEMY_SHOT_INTERVAL])),
                e_fish,
            ),
            lambda: (
                e_subs,
                e_fish.at[_find_ix(e_fish)].set(jnp.int32([x, enemy_y, enemy_lr, move_speed])),
            ),
        ),
    )


def _spawn_diver(divers, diver_lr, diver_y):
    x = lax.cond(diver_lr, lambda: ZERO, lambda: NINE)
    ix = _find_ix(divers)
    return divers.at[ix].set(jnp.array([x, diver_y, diver_lr, DIVER_MOVE_INTERVAL], dtype=jnp.int32))
