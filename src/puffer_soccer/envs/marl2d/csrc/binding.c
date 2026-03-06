#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PLAYERS 22
#define MAX_PER_TEAM 11
#define MAX_BALL_SPEED 5.0f

typedef struct {
    float score;
    float episode_return;
    float episode_length;
    float wins_blue;
    float wins_red;
    float draws;
    float n;
} Log;

typedef struct {
    float x;
    float y;
    float rot;
    float last_move;
    float last_rot;
    int team;
} Agent;

typedef struct {
    Log log;
    float* observations;
    void* actions;
    float* rewards;
    unsigned char* terminals;
    unsigned char* truncations;
    float* global_states;
    Agent agents[MAX_PLAYERS];
    int players_per_team;
    int num_players;
    int game_length;
    int num_steps;
    int do_team_switch;
    int blue_left;
    int reset_setup;
    int action_mode;
    int last_goals_blue;
    int last_goals_red;
    unsigned char last_done;
    float vision_range;
    float x_out_start;
    float x_out_end;
    float y_out_start;
    float y_out_end;
    float goal_half_h;
    float rot_speed;
    float move_speed;
    float ball_x;
    float ball_y;
    float ball_vx;
    float ball_vy;
    int goals_blue;
    int goals_red;
    int obs_size;
    int state_size;
    uint32_t rng;
} Env;

typedef struct {
    Env* envs;
    int num_envs;
} Vec;

static const float init_position_11[11][2] = {
    {0.0f, -0.45f}, {-0.225f, -0.3f}, {-0.075f, -0.3f}, {0.075f, -0.3f}, {0.225f, -0.3f},
    {-0.2f, -0.2f}, {0.0f, -0.2f}, {0.2f, -0.2f}, {-0.2f, -0.1f}, {0.0f, -0.1f}, {0.2f, -0.1f}
};

static float clampf(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static float dist2(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx*dx + dy*dy);
}

static float speed2(float vx, float vy) {
    return sqrtf(vx*vx + vy*vy);
}

static uint32_t next_u32(uint32_t* s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

static float randf(uint32_t* s, float lo, float hi) {
    float r = (next_u32(s) & 0xFFFFFF) / (float)0x1000000;
    return lo + (hi - lo) * r;
}

static int obs_size_for_team(int n) {
    return 16 + 14*n;
}

static int state_size_for_team(int n) {
    return 5 + 34*n;
}

static void get_one_hot(int id, float* out11) {
    memset(out11, 0, sizeof(float) * 11);
    out11[id % 11] = 1.0f;
}

static int team_on_left(const Env* env, int team) {
    if (team == 0) return env->blue_left;
    return !env->blue_left;
}

static void reset_field(Env* env) {
    float pos_noise = 0.05f;
    for (int i = 0; i < env->num_players; i++) {
        Agent* a = &env->agents[i];
        int place_left = team_on_left(env, a->team);

        if (env->reset_setup == 0 && env->num_players == 22) {
            int pidx = i % 11;
            float new_x = (init_position_11[pidx][1] + randf(&env->rng, -pos_noise, pos_noise)) * 110.0f;
            float new_y = (init_position_11[pidx][0] + randf(&env->rng, -pos_noise, pos_noise)) * 110.0f;
            a->x = clampf(new_x, env->x_out_start, 0.0f);
            a->y = clampf(new_y, env->y_out_start, env->y_out_end);
        } else {
            a->x = randf(&env->rng, env->x_out_start, 0.0f);
            a->y = randf(&env->rng, env->y_out_start, env->y_out_end);
        }

        a->rot = randf(&env->rng, -1.0f, 1.0f) * (float)M_PI;
        if (!place_left) {
            a->x *= -1.0f;
            a->y *= -1.0f;
            a->rot += (float)M_PI;
            if (a->rot > 2.0f * (float)M_PI) a->rot -= 2.0f * (float)M_PI;
        }
        a->last_move = 0.0f;
        a->last_rot = 0.0f;
    }

    env->ball_vx = 0.0f;
    env->ball_vy = 0.0f;
    env->ball_x = randf(&env->rng, env->x_out_start, env->x_out_end);
    env->ball_y = randf(&env->rng, env->y_out_start, env->y_out_end);
    if (!env->blue_left) {
        env->ball_x *= -1.0f;
    }
}

static void full_reset(Env* env, int hard_reset_score) {
    env->num_steps = 0;
    if (hard_reset_score) {
        env->goals_blue = 0;
        env->goals_red = 0;
        if (env->do_team_switch) {
            env->blue_left = !env->blue_left;
        }
    }
    reset_field(env);
}

static void calc_line_ball_stats(float angle, float circ_rad, float circ_x, float circ_y,
        float cen_x, float cen_y, float* delta, float* cos_comp, float* sin_comp) {
    *sin_comp = sinf(angle);
    *cos_comp = cosf(angle);
    float x1 = cen_x + 0.5f*(*sin_comp) - circ_x;
    float y1 = cen_y - 0.5f*(*cos_comp) - circ_y;
    float x2 = cen_x - 0.5f*(*sin_comp) - circ_x;
    float y2 = cen_y + 0.5f*(*cos_comp) - circ_y;
    float D = x1 * y2 - x2 * y1;
    *delta = circ_rad*circ_rad - D*D;
}

static void ball_check_hit(Env* env) {
    const float ball_radius = 1.0f;
    const float body_speed = 0.6f;
    const float leg_speed = 4.0f;
    const float agent_radius = 1.0f;
    const float leg_length = 3.0f;

    for (int i = 0; i < env->num_players; i++) {
        Agent* a = &env->agents[i];
        float d = dist2(env->ball_x, env->ball_y, a->x, a->y);
        if (d < ball_radius + agent_radius) {
            float x_diff = env->ball_x - a->x;
            float y_diff = env->ball_y - a->y;
            env->ball_vx += body_speed * x_diff / (d + 1e-4f);
            env->ball_vy += body_speed * y_diff / (d + 1e-4f);
        }

        if (d < ball_radius + leg_length) {
            float delta, cc, ss;
            calc_line_ball_stats(a->rot, ball_radius, env->ball_x, env->ball_y, a->x, a->y, &delta, &cc, &ss);
            if (delta >= 0.0f && d < 2.0f * agent_radius + 2.0f * ball_radius) {
                env->ball_vx += leg_speed * cc;
                env->ball_vy += leg_speed * ss;
            }
        }
    }

    float spd = speed2(env->ball_vx, env->ball_vy);
    if (spd > MAX_BALL_SPEED) {
        float ratio = MAX_BALL_SPEED / spd;
        env->ball_vx *= ratio;
        env->ball_vy *= ratio;
    }
}

static int visible(float focus_rot, float obj_rot, float vision_range, float* obj_rot_view) {
    float start = focus_rot - vision_range / 2.0f;
    float end = focus_rot + vision_range / 2.0f;

    if (start > end) start -= 2.0f * (float)M_PI;

    if (obj_rot > end) obj_rot -= 2.0f * (float)M_PI;
    else if (obj_rot < start) obj_rot += 2.0f * (float)M_PI;

    if (start <= obj_rot && obj_rot <= end) {
        *obj_rot_view = obj_rot - (start + end) / 2.0f;
        return 1;
    }
    return 0;
}

static void rel_obs_agent(const Env* env, const Agent* focus, const Agent* other, float* out7) {
    float obj_rot = atan2f(other->y - focus->y, other->x - focus->x);
    float obj_view = 0.0f;
    if (!visible(focus->rot, obj_rot, env->vision_range, &obj_view)) {
        memset(out7, 0, sizeof(float) * 7);
        return;
    }

    float max_dist = dist2(0, 0, 100.0f, 70.0f);
    out7[0] = 1.0f;
    out7[1] = dist2(focus->x, focus->y, other->x, other->y) / max_dist;
    out7[2] = obj_view / (env->vision_range / 2.0f);

    float rot = other->rot;
    float x_rot = cosf(rot);
    float y_rot = sinf(rot);
    float rel_x = cosf(rot - focus->rot);
    float rel_y = sinf(rot - focus->rot);
    (void)x_rot;
    (void)y_rot;
    out7[3] = rel_x;
    out7[4] = rel_y;
    out7[5] = other->last_move;
    out7[6] = other->last_rot;
}

static void rel_obs_ball(const Env* env, const Agent* focus, float* out5) {
    float obj_rot = atan2f(env->ball_y - focus->y, env->ball_x - focus->x);
    float obj_view = 0.0f;
    if (!visible(focus->rot, obj_rot, env->vision_range, &obj_view)) {
        memset(out5, 0, sizeof(float) * 5);
        return;
    }

    float max_dist = dist2(0, 0, 100.0f, 70.0f);
    out5[0] = 1.0f;
    out5[1] = dist2(focus->x, focus->y, env->ball_x, env->ball_y) / max_dist;
    out5[2] = obj_view / (env->vision_range / 2.0f);

    float vel_rot = atan2f(env->ball_vy, env->ball_vx);
    float abs_val = speed2(env->ball_vx, env->ball_vy);
    float rel_x = cosf(vel_rot - focus->rot) * abs_val / MAX_BALL_SPEED;
    float rel_y = sinf(vel_rot - focus->rot) * abs_val / MAX_BALL_SPEED;
    out5[3] = rel_x;
    out5[4] = rel_y;
}

static void compute_observations(Env* env, int goal_scored_team) {
    int np = env->num_players;
    float sign_state = team_on_left(env, 0) ? 1.0f : -1.0f;
    float* state = env->global_states;
    float time_left = 1.0f - ((float)env->num_steps / (float)env->game_length);
    float* state_base = state;
    state_base[0] = time_left;
    state_base[1] = sign_state * env->ball_x / 50.0f;
    state_base[2] = sign_state * env->ball_y / 50.0f;
    state_base[3] = sign_state * env->ball_vx / MAX_BALL_SPEED;
    state_base[4] = sign_state * env->ball_vy / MAX_BALL_SPEED;

    int sidx = 5;
    for (int i = 0; i < np; i++) {
        Agent* a = &env->agents[i];
        float onehot[11];
        get_one_hot(i, onehot);
        state_base[sidx++] = sign_state * a->x / 50.0f;
        state_base[sidx++] = sign_state * a->y / 50.0f;
        state_base[sidx++] = sign_state * cosf(a->rot);
        state_base[sidx++] = sign_state * sinf(a->rot);
        state_base[sidx++] = a->last_move;
        state_base[sidx++] = a->last_rot;
        for (int k = 0; k < 11; k++) state_base[sidx++] = onehot[k];
    }

    for (int a = 1; a < np; a++) {
        memcpy(state + a * env->state_size, state_base, sizeof(float) * env->state_size);
    }

    for (int i = 0; i < np; i++) {
        Agent* focus = &env->agents[i];
        float sign = team_on_left(env, focus->team) ? 1.0f : -1.0f;
        float* out = env->observations + (i * env->obs_size);
        int o = 0;

        out[o++] = time_left;
        out[o++] = sign * focus->x / 50.0f;
        out[o++] = sign * focus->y / 50.0f;
        out[o++] = sign * cosf(focus->rot);
        out[o++] = sign * sinf(focus->rot);
        out[o++] = focus->last_move;
        out[o++] = focus->last_rot;

        float onehot[11];
        get_one_hot(i, onehot);
        for (int k = 0; k < 11; k++) out[o++] = onehot[k];

        float brobs[5];
        rel_obs_ball(env, focus, brobs);
        for (int k = 0; k < 5; k++) out[o++] = brobs[k];

        for (int j = 0; j < np; j++) {
            if (j == i) continue;
            if (env->agents[j].team != focus->team) continue;
            float aobs[7];
            rel_obs_agent(env, focus, &env->agents[j], aobs);
            for (int k = 0; k < 7; k++) out[o++] = aobs[k];
        }
        for (int j = 0; j < np; j++) {
            if (env->agents[j].team == focus->team) continue;
            float aobs[7];
            rel_obs_agent(env, focus, &env->agents[j], aobs);
            for (int k = 0; k < 7; k++) out[o++] = aobs[k];
        }

        float r = 0.0f;
        if (goal_scored_team >= 0) {
            r = (focus->team == goal_scored_team) ? 1.0f : -1.0f;
        }
        env->rewards[i] = r;
    }
}

static void clear_outputs(Env* env) {
    for (int i = 0; i < env->num_players; i++) {
        env->rewards[i] = 0.0f;
        env->terminals[i] = 0;
        env->truncations[i] = 0;
    }
}

static void init_env_common(
    Env* env,
    int seed,
    int players_per_team,
    int game_length,
    int action_mode,
    int do_team_switch,
    float vision_range,
    int reset_setup
) {
    memset(&env->log, 0, sizeof(Log));
    env->players_per_team = players_per_team;
    env->num_players = players_per_team * 2;
    env->game_length = game_length;
    env->do_team_switch = do_team_switch;
    env->blue_left = 1;
    env->reset_setup = reset_setup;
    env->vision_range = vision_range;
    env->action_mode = action_mode;
    env->obs_size = obs_size_for_team(players_per_team);
    env->state_size = state_size_for_team(players_per_team);
    env->last_goals_blue = 0;
    env->last_goals_red = 0;
    env->last_done = 0;
    env->x_out_start = -50.0f;
    env->x_out_end = 50.0f;
    env->y_out_start = -35.0f;
    env->y_out_end = 35.0f;
    env->goal_half_h = 20.0f;
    env->rot_speed = 0.4f;
    env->move_speed = 1.0f;
    env->rng = (uint32_t)(seed + 1);

    for (int a = 0; a < env->num_players; a++) {
        env->agents[a].team = (a < players_per_team) ? 0 : 1;
    }

    full_reset(env, 1);
    clear_outputs(env);
    compute_observations(env, -1);
}

static void c_reset(Env* env, int seed) {
    env->rng = (uint32_t)(seed + 1);
    env->blue_left = 1;
    full_reset(env, 1);
    env->last_goals_blue = 0;
    env->last_goals_red = 0;
    env->last_done = 0;
    clear_outputs(env);
    compute_observations(env, -1);
}

static void c_step(Env* env) {
    int np = env->num_players;
    env->num_steps += 1;
    clear_outputs(env);

    for (int i = 0; i < np; i++) {
        Agent* a = &env->agents[i];
        float move = 0.0f;
        float rot = 0.0f;

        if (env->action_mode == 0) {
            int* atn = (int*)env->actions;
            int action = atn[i];
            if (action < 0) action = 0;
            if (action > 8) action = 8;
            move = (float)(action / 3) - 1.0f;
            rot = (float)(action % 3) - 1.0f;
        } else {
            float* atn = (float*)env->actions;
            float* arow = atn + (i * 2);
            move = clampf(arow[0], -1.0f, 1.0f);
            rot = clampf(arow[1], -1.0f, 1.0f);
        }

        a->last_move = move;
        a->last_rot = rot;

        a->rot -= rot * env->rot_speed;
        if (a->rot > 2.0f*(float)M_PI) a->rot -= 2.0f*(float)M_PI;
        else if (a->rot < 0.0f) a->rot += 2.0f*(float)M_PI;

        float sin_comp = move * env->move_speed * sinf(a->rot);
        float cos_comp = move * env->move_speed * cosf(a->rot);

        a->x = a->x + cos_comp;
        a->y = a->y + sin_comp;
        a->x = clampf(a->x + cos_comp, env->x_out_start, env->x_out_end);
        a->y = clampf(a->y + sin_comp, env->y_out_start, env->y_out_end);
    }

    ball_check_hit(env);

    env->ball_x += env->ball_vx;
    env->ball_y += env->ball_vy;

    env->ball_vx *= 0.9f;
    env->ball_vy *= 0.9f;
    if (speed2(env->ball_vx, env->ball_vy) < 0.01f) {
        env->ball_vx = 0.0f;
        env->ball_vy = 0.0f;
    }

    int goal_scored = -1;
    if (fabsf(env->ball_y) <= env->goal_half_h) {
        if (env->ball_x < env->x_out_start) {
            goal_scored = team_on_left(env, 1) ? 1 : 0;
            if (goal_scored == 0) env->goals_blue += 1;
            else env->goals_red += 1;
        } else if (env->ball_x > env->x_out_end) {
            goal_scored = team_on_left(env, 0) ? 0 : 1;
            if (goal_scored == 0) env->goals_blue += 1;
            else env->goals_red += 1;
        }
    } else {
        if (env->ball_x < env->x_out_start) {
            env->ball_x = env->x_out_start;
            env->ball_vx = -env->ball_vx;
            env->ball_vx *= 0.6f;
            env->ball_vy *= 0.6f;
        } else if (env->ball_x > env->x_out_end) {
            env->ball_x = env->x_out_end;
            env->ball_vx = -env->ball_vx;
            env->ball_vx *= 0.6f;
            env->ball_vy *= 0.6f;
        }
    }

    if (env->ball_y < env->y_out_start) {
        env->ball_y = env->y_out_start;
        env->ball_vy = -env->ball_vy;
        env->ball_vx *= 0.6f;
        env->ball_vy *= 0.6f;
    } else if (env->ball_y > env->y_out_end) {
        env->ball_y = env->y_out_end;
        env->ball_vy = -env->ball_vy;
        env->ball_vx *= 0.6f;
        env->ball_vy *= 0.6f;
    }

    if (goal_scored >= 0) {
        reset_field(env);
    }

    int done = (env->num_steps >= env->game_length);
    compute_observations(env, goal_scored);

    if (done) {
        float ret = 0.0f;
        int score_diff;
        for (int i = 0; i < np; i++) {
            env->terminals[i] = 1;
            ret += env->rewards[i];
        }
        env->last_goals_blue = env->goals_blue;
        env->last_goals_red = env->goals_red;
        env->last_done = 1;
        score_diff = env->goals_blue - env->goals_red;
        env->log.score += (float)score_diff;
        if (score_diff > 0) env->log.wins_blue += 1.0f;
        else if (score_diff < 0) env->log.wins_red += 1.0f;
        else env->log.draws += 1.0f;
        env->log.episode_return += ret;
        env->log.episode_length += env->num_steps;
        env->log.n += 1.0f;
        full_reset(env, 1);
    }
}

static int assign_log_dict(PyObject* d, const Log* log) {
    if (PyDict_SetItemString(d, "score", PyFloat_FromDouble(log->score / log->n)) < 0) return -1;
    if (PyDict_SetItemString(d, "score_diff", PyFloat_FromDouble(log->score / log->n)) < 0) return -1;
    if (PyDict_SetItemString(d, "episode_return", PyFloat_FromDouble(log->episode_return / log->n)) < 0) return -1;
    if (PyDict_SetItemString(d, "episode_length", PyFloat_FromDouble(log->episode_length / log->n)) < 0) return -1;
    if (PyDict_SetItemString(d, "wins_blue", PyFloat_FromDouble(log->wins_blue)) < 0) return -1;
    if (PyDict_SetItemString(d, "wins_red", PyFloat_FromDouble(log->wins_red)) < 0) return -1;
    if (PyDict_SetItemString(d, "draws", PyFloat_FromDouble(log->draws)) < 0) return -1;
    if (PyDict_SetItemString(d, "win_rate_blue", PyFloat_FromDouble(log->wins_blue / log->n)) < 0) return -1;
    if (PyDict_SetItemString(d, "n", PyFloat_FromDouble(log->n)) < 0) return -1;
    return 0;
}

static PyObject* build_log_dict(Log* log) {
    if (log->n <= 0.0f) {
        Py_RETURN_NONE;
    }
    PyObject* d = PyDict_New();
    if (assign_log_dict(d, log) < 0) {
        Py_DECREF(d);
        return NULL;
    }
    memset(log, 0, sizeof(Log));
    return d;
}

static Env* unpack_env_handle(PyObject* handle_obj) {
    if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "env_handle must be an integer");
        return NULL;
    }
    Env* env = (Env*)PyLong_AsVoidPtr(handle_obj);
    if (!env) {
        PyErr_SetString(PyExc_ValueError, "invalid env handle");
        return NULL;
    }
    return env;
}

static Vec* unpack_vec_handle(PyObject* handle_obj) {
    if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "vec_handle must be an integer");
        return NULL;
    }
    Vec* vec = (Vec*)PyLong_AsVoidPtr(handle_obj);
    if (!vec) {
        PyErr_SetString(PyExc_ValueError, "invalid vec handle");
        return NULL;
    }
    return vec;
}

static int validate_common_arrays(
    PyArrayObject* obs,
    PyArrayObject* act,
    PyArrayObject* rew,
    PyArrayObject* term,
    PyArrayObject* trunc,
    PyArrayObject* gst,
    int players_per_team
) {
    int num_players = players_per_team * 2;
    if (PyArray_ITEMSIZE(act) == sizeof(double)) {
        PyErr_SetString(PyExc_ValueError, "Action tensor passed as float64 (pass np.float32 buffer)");
        return -1;
    }
    if (PyArray_NDIM(rew) != 1 || PyArray_DIM(rew, 0) != num_players) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be shape (num_players,)");
        return -1;
    }
    if (PyArray_NDIM(term) != 1 || PyArray_DIM(term, 0) != num_players) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be shape (num_players,)");
        return -1;
    }
    if (PyArray_NDIM(trunc) != 1 || PyArray_DIM(trunc, 0) != num_players) {
        PyErr_SetString(PyExc_ValueError, "Truncations must be shape (num_players,)");
        return -1;
    }
    if (PyArray_NDIM(obs) != 2 || PyArray_DIM(obs, 0) != num_players) {
        PyErr_SetString(PyExc_ValueError, "Observations must be shape (num_players, obs_size)");
        return -1;
    }
    if (PyArray_NDIM(gst) != 2 || PyArray_DIM(gst, 0) != num_players) {
        PyErr_SetString(PyExc_ValueError, "Global states must be shape (num_players, state_size)");
        return -1;
    }
    return 0;
}

static int validate_vector_arrays(
    PyArrayObject* obs,
    PyArrayObject* act,
    PyArrayObject* rew,
    PyArrayObject* term,
    PyArrayObject* trunc,
    PyArrayObject* gst,
    int num_envs,
    int players_per_team
) {
    int total_players = num_envs * players_per_team * 2;
    if (PyArray_ITEMSIZE(act) == sizeof(double)) {
        PyErr_SetString(PyExc_ValueError, "Action tensor passed as float64 (pass np.float32 buffer)");
        return -1;
    }
    if (PyArray_NDIM(rew) != 1 || PyArray_DIM(rew, 0) != total_players) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be shape (num_envs * num_players,)");
        return -1;
    }
    if (PyArray_NDIM(term) != 1 || PyArray_DIM(term, 0) != total_players) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be shape (num_envs * num_players,)");
        return -1;
    }
    if (PyArray_NDIM(trunc) != 1 || PyArray_DIM(trunc, 0) != total_players) {
        PyErr_SetString(PyExc_ValueError, "Truncations must be shape (num_envs * num_players,)");
        return -1;
    }
    if (PyArray_NDIM(obs) != 2 || PyArray_DIM(obs, 0) != total_players) {
        PyErr_SetString(PyExc_ValueError, "Observations must be shape (num_envs * num_players, obs_size)");
        return -1;
    }
    if (PyArray_NDIM(gst) != 2 || PyArray_DIM(gst, 0) != total_players) {
        PyErr_SetString(PyExc_ValueError, "Global states must be shape (num_envs * num_players, state_size)");
        return -1;
    }
    return 0;
}

static PyObject* py_env_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *obs_obj, *act_obj, *rew_obj, *term_obj, *trunc_obj, *state_obj;
    int seed, players_per_team, game_length, action_mode, do_team_switch, reset_setup;
    float vision_range;

    static char* kwlist[] = {
        "observations", "actions", "rewards", "terminals", "truncations", "global_states",
        "seed", "players_per_team", "game_length", "action_mode", "do_team_switch",
        "vision_range", "reset_setup", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOOOiiiiifi", kwlist,
            &obs_obj, &act_obj, &rew_obj, &term_obj, &trunc_obj, &state_obj,
            &seed, &players_per_team, &game_length, &action_mode,
            &do_team_switch, &vision_range, &reset_setup)) {
        return NULL;
    }

    if (players_per_team < 1 || players_per_team > 11) {
        PyErr_SetString(PyExc_ValueError, "players_per_team must be in [1, 11]");
        return NULL;
    }

    PyArrayObject* obs = (PyArrayObject*)PyArray_FROM_OTF(obs_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* act = (PyArrayObject*)PyArray_FROM_O(act_obj);
    PyArrayObject* rew = (PyArrayObject*)PyArray_FROM_OTF(rew_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* term = (PyArrayObject*)PyArray_FROM_OTF(term_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* trunc = (PyArrayObject*)PyArray_FROM_OTF(trunc_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* gst = (PyArrayObject*)PyArray_FROM_OTF(state_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

    if (!obs || !act || !rew || !term || !trunc || !gst) {
        Py_XDECREF(obs); Py_XDECREF(act); Py_XDECREF(rew); Py_XDECREF(term); Py_XDECREF(trunc); Py_XDECREF(gst);
        return NULL;
    }
    if (validate_common_arrays(obs, act, rew, term, trunc, gst, players_per_team) < 0) {
        Py_DECREF(obs); Py_DECREF(act); Py_DECREF(rew); Py_DECREF(term); Py_DECREF(trunc); Py_DECREF(gst);
        return NULL;
    }

    Env* env = (Env*)calloc(1, sizeof(Env));
    if (!env) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate env");
        Py_DECREF(obs); Py_DECREF(act); Py_DECREF(rew); Py_DECREF(term); Py_DECREF(trunc); Py_DECREF(gst);
        return NULL;
    }

    env->observations = (float*)PyArray_DATA(obs);
    env->actions = PyArray_DATA(act);
    env->rewards = (float*)PyArray_DATA(rew);
    env->terminals = (unsigned char*)PyArray_DATA(term);
    env->truncations = (unsigned char*)PyArray_DATA(trunc);
    env->global_states = (float*)PyArray_DATA(gst);
    init_env_common(env, seed, players_per_team, game_length, action_mode, do_team_switch, vision_range, reset_setup);

    Py_DECREF(obs); Py_DECREF(act); Py_DECREF(rew); Py_DECREF(term); Py_DECREF(trunc); Py_DECREF(gst);
    return PyLong_FromVoidPtr((void*)env);
}

static PyObject* py_env_reset(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    int seed;
    if (!PyArg_ParseTuple(args, "Oi", &handle_obj, &seed)) {
        return NULL;
    }
    Env* env = unpack_env_handle(handle_obj);
    if (!env) return NULL;
    c_reset(env, seed);
    Py_RETURN_NONE;
}

static PyObject* py_env_step(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj)) {
        return NULL;
    }
    Env* env = unpack_env_handle(handle_obj);
    if (!env) return NULL;
    c_step(env);
    Py_RETURN_NONE;
}

static PyObject* py_env_log(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj)) {
        return NULL;
    }
    Env* env = unpack_env_handle(handle_obj);
    if (!env) return NULL;
    return build_log_dict(&env->log);
}

static PyObject* py_env_get_last_scores(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    int clear = 1;
    if (!PyArg_ParseTuple(args, "O|p", &handle_obj, &clear)) return NULL;
    Env* env = unpack_env_handle(handle_obj);
    if (!env) return NULL;
    if (!env->last_done) Py_RETURN_NONE;
    PyObject* scores = Py_BuildValue("(ii)", env->last_goals_blue, env->last_goals_red);
    if (clear) env->last_done = 0;
    return scores;
}

static PyObject* py_env_get_state(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj)) return NULL;
    Env* env = unpack_env_handle(handle_obj);
    if (!env) return NULL;

    npy_intp pos_dims[2] = {env->num_players, 2};
    npy_intp rot_dims[1] = {env->num_players};

    PyObject* pos = PyArray_SimpleNew(2, pos_dims, NPY_FLOAT32);
    PyObject* rot = PyArray_SimpleNew(1, rot_dims, NPY_FLOAT32);
    float* pdat = (float*)PyArray_DATA((PyArrayObject*)pos);
    float* rdat = (float*)PyArray_DATA((PyArrayObject*)rot);

    for (int i = 0; i < env->num_players; i++) {
        pdat[i*2] = env->agents[i].x;
        pdat[i*2 + 1] = env->agents[i].y;
        rdat[i] = env->agents[i].rot;
    }

    PyObject* ball = Py_BuildValue("(ffff)", env->ball_x, env->ball_y, env->ball_vx, env->ball_vy);
    PyObject* goals = Py_BuildValue("(ii)", env->goals_blue, env->goals_red);
    PyObject* d = PyDict_New();
    PyDict_SetItemString(d, "positions", pos);
    PyDict_SetItemString(d, "rotations", rot);
    PyDict_SetItemString(d, "ball", ball);
    PyDict_SetItemString(d, "goals", goals);
    PyDict_SetItemString(d, "num_steps", PyLong_FromLong(env->num_steps));
    PyDict_SetItemString(d, "blue_left", PyBool_FromLong(env->blue_left));
    Py_DECREF(pos);
    Py_DECREF(rot);
    Py_DECREF(ball);
    Py_DECREF(goals);
    return d;
}

static PyObject* py_env_close(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj)) return NULL;
    Env* env = unpack_env_handle(handle_obj);
    if (!env) return NULL;
    free(env);
    Py_RETURN_NONE;
}

static PyObject* py_vec_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *obs_obj, *act_obj, *rew_obj, *term_obj, *trunc_obj, *state_obj;
    int num_envs, seed, players_per_team, game_length, action_mode, do_team_switch, reset_setup;
    float vision_range;

    static char* kwlist[] = {
        "observations", "actions", "rewards", "terminals", "truncations", "global_states",
        "num_envs", "seed", "players_per_team", "game_length", "action_mode", "do_team_switch",
        "vision_range", "reset_setup", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOOOiiiiiifi", kwlist,
            &obs_obj, &act_obj, &rew_obj, &term_obj, &trunc_obj, &state_obj,
            &num_envs, &seed, &players_per_team, &game_length, &action_mode,
            &do_team_switch, &vision_range, &reset_setup)) {
        return NULL;
    }

    if (num_envs < 1) {
        PyErr_SetString(PyExc_ValueError, "num_envs must be positive");
        return NULL;
    }
    if (players_per_team < 1 || players_per_team > 11) {
        PyErr_SetString(PyExc_ValueError, "players_per_team must be in [1, 11]");
        return NULL;
    }

    PyArrayObject* obs = (PyArrayObject*)PyArray_FROM_OTF(obs_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* act = (PyArrayObject*)PyArray_FROM_O(act_obj);
    PyArrayObject* rew = (PyArrayObject*)PyArray_FROM_OTF(rew_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* term = (PyArrayObject*)PyArray_FROM_OTF(term_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* trunc = (PyArrayObject*)PyArray_FROM_OTF(trunc_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* gst = (PyArrayObject*)PyArray_FROM_OTF(state_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

    if (!obs || !act || !rew || !term || !trunc || !gst) {
        Py_XDECREF(obs); Py_XDECREF(act); Py_XDECREF(rew); Py_XDECREF(term); Py_XDECREF(trunc); Py_XDECREF(gst);
        return NULL;
    }
    if (validate_vector_arrays(obs, act, rew, term, trunc, gst, num_envs, players_per_team) < 0) {
        Py_DECREF(obs); Py_DECREF(act); Py_DECREF(rew); Py_DECREF(term); Py_DECREF(trunc); Py_DECREF(gst);
        return NULL;
    }

    Vec* vec = (Vec*)calloc(1, sizeof(Vec));
    if (!vec) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate vec env");
        Py_DECREF(obs); Py_DECREF(act); Py_DECREF(rew); Py_DECREF(term); Py_DECREF(trunc); Py_DECREF(gst);
        return NULL;
    }
    vec->num_envs = num_envs;
    vec->envs = (Env*)calloc(num_envs, sizeof(Env));
    if (!vec->envs) {
        free(vec);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate env array");
        Py_DECREF(obs); Py_DECREF(act); Py_DECREF(rew); Py_DECREF(term); Py_DECREF(trunc); Py_DECREF(gst);
        return NULL;
    }

    int num_players = players_per_team * 2;
    npy_intp obs_stride = PyArray_STRIDE(obs, 0) * num_players;
    npy_intp act_stride = PyArray_STRIDE(act, 0) * num_players;
    npy_intp rew_stride = PyArray_STRIDE(rew, 0) * num_players;
    npy_intp term_stride = PyArray_STRIDE(term, 0) * num_players;
    npy_intp trunc_stride = PyArray_STRIDE(trunc, 0) * num_players;
    npy_intp state_stride = PyArray_STRIDE(gst, 0) * num_players;

    for (int i = 0; i < num_envs; i++) {
        Env* env = &vec->envs[i];
        env->observations = (float*)((char*)PyArray_DATA(obs) + i * obs_stride);
        env->actions = (void*)((char*)PyArray_DATA(act) + i * act_stride);
        env->rewards = (float*)((char*)PyArray_DATA(rew) + i * rew_stride);
        env->terminals = (unsigned char*)((char*)PyArray_DATA(term) + i * term_stride);
        env->truncations = (unsigned char*)((char*)PyArray_DATA(trunc) + i * trunc_stride);
        env->global_states = (float*)((char*)PyArray_DATA(gst) + i * state_stride);
        init_env_common(env, seed + i*7919, players_per_team, game_length, action_mode, do_team_switch, vision_range, reset_setup);
    }

    Py_DECREF(obs); Py_DECREF(act); Py_DECREF(rew); Py_DECREF(term); Py_DECREF(trunc); Py_DECREF(gst);
    return PyLong_FromVoidPtr((void*)vec);
}

static PyObject* py_vec_reset(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    int seed;
    if (!PyArg_ParseTuple(args, "Oi", &handle_obj, &seed)) {
        return NULL;
    }
    Vec* vec = unpack_vec_handle(handle_obj);
    if (!vec) return NULL;
    for (int i = 0; i < vec->num_envs; i++) {
        c_reset(&vec->envs[i], seed + i*7919);
    }
    Py_RETURN_NONE;
}

static PyObject* py_vec_step(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj)) {
        return NULL;
    }
    Vec* vec = unpack_vec_handle(handle_obj);
    if (!vec) return NULL;
    for (int i = 0; i < vec->num_envs; i++) {
        c_step(&vec->envs[i]);
    }
    Py_RETURN_NONE;
}

static PyObject* py_vec_log(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj)) return NULL;
    Vec* vec = unpack_vec_handle(handle_obj);
    if (!vec) return NULL;
    Log aggregate;
    memset(&aggregate, 0, sizeof(Log));
    for (int i = 0; i < vec->num_envs; i++) {
        Log* log = &vec->envs[i].log;
        aggregate.score += log->score;
        aggregate.episode_return += log->episode_return;
        aggregate.episode_length += log->episode_length;
        aggregate.wins_blue += log->wins_blue;
        aggregate.wins_red += log->wins_red;
        aggregate.draws += log->draws;
        aggregate.n += log->n;
        memset(log, 0, sizeof(Log));
    }
    return build_log_dict(&aggregate);
}

static PyObject* py_vec_get_last_scores(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    int env_idx;
    int clear = 1;
    if (!PyArg_ParseTuple(args, "Oi|p", &handle_obj, &env_idx, &clear)) return NULL;
    Vec* vec = unpack_vec_handle(handle_obj);
    if (!vec) return NULL;
    if (env_idx < 0 || env_idx >= vec->num_envs) {
        PyErr_SetString(PyExc_ValueError, "invalid env index");
        return NULL;
    }
    Env* env = &vec->envs[env_idx];
    if (!env->last_done) Py_RETURN_NONE;
    PyObject* scores = Py_BuildValue("(ii)", env->last_goals_blue, env->last_goals_red);
    if (clear) env->last_done = 0;
    return scores;
}

static PyObject* py_vec_get_state(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    int env_idx;
    if (!PyArg_ParseTuple(args, "Oi", &handle_obj, &env_idx)) return NULL;
    Vec* vec = unpack_vec_handle(handle_obj);
    if (!vec) return NULL;
    if (env_idx < 0 || env_idx >= vec->num_envs) {
        PyErr_SetString(PyExc_ValueError, "invalid env index");
        return NULL;
    }
    PyObject* env_handle = PyLong_FromVoidPtr((void*)&vec->envs[env_idx]);
    if (!env_handle) return NULL;
    PyObject* packed = PyTuple_Pack(1, env_handle);
    Py_DECREF(env_handle);
    if (!packed) return NULL;
    PyObject* result = py_env_get_state(self, packed);
    Py_DECREF(packed);
    return result;
}

static PyObject* py_vec_close(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj)) return NULL;
    Vec* vec = unpack_vec_handle(handle_obj);
    if (!vec) return NULL;
    free(vec->envs);
    free(vec);
    Py_RETURN_NONE;
}

static PyMethodDef Methods[] = {
    {"env_init", (PyCFunction)py_env_init, METH_VARARGS | METH_KEYWORDS, "Initialize one env"},
    {"env_reset", py_env_reset, METH_VARARGS, "Reset one env"},
    {"env_step", py_env_step, METH_VARARGS, "Step one env"},
    {"env_log", py_env_log, METH_VARARGS, "Get one env log"},
    {"env_get_last_scores", py_env_get_last_scores, METH_VARARGS, "Get last scalar env scores"},
    {"env_get_state", py_env_get_state, METH_VARARGS, "Get one env state"},
    {"env_close", py_env_close, METH_VARARGS, "Close one env"},
    {"vec_init", (PyCFunction)py_vec_init, METH_VARARGS | METH_KEYWORDS, "Initialize vector env"},
    {"vec_reset", py_vec_reset, METH_VARARGS, "Reset vector env"},
    {"vec_step", py_vec_step, METH_VARARGS, "Step vector env"},
    {"vec_log", py_vec_log, METH_VARARGS, "Get vector log"},
    {"vec_get_last_scores", py_vec_get_last_scores, METH_VARARGS, "Get last vector env scores"},
    {"vec_get_state", py_vec_get_state, METH_VARARGS, "Get one env state from vector env"},
    {"vec_close", py_vec_close, METH_VARARGS, "Close vector env"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "binding",
    "MARL2D C binding",
    -1,
    Methods,
};

PyMODINIT_FUNC PyInit_binding(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
