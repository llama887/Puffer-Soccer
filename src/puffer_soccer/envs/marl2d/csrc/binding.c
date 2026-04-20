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
#define BALL_VELOCITY_DECAY 0.85f
#define DISCRETE_ACTION_NOOP 0
#define DISCRETE_ACTION_MOVE_FORWARD 1
#define DISCRETE_ACTION_MOVE_BACKWARD 2
#define DISCRETE_ACTION_ROTATE_LEFT 3
#define DISCRETE_ACTION_ROTATE_RIGHT 4
#define DISCRETE_KICK_ACTION_START 5
#define DISCRETE_KICK_BUCKETS 8
#define DISCRETE_ACTION_COUNT (DISCRETE_KICK_ACTION_START + DISCRETE_KICK_BUCKETS)

#define STAT_SIGMA 0.15f
#define WHEELBASE 2.0f
#define MAX_STEER_ANGLE 1.2f
#define STEER_RATE 0.4f

typedef struct {
    float score;
    float episode_return;
    float blue_team_episode_return;
    float red_team_episode_return;
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
    float stat_kick;
    float stat_speed;
    float stat_turn;
    float steer_angle;
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
    float cumulative_episode_return;
    float cumulative_blue_team_episode_return;
    float cumulative_red_team_episode_return;
    int do_team_switch;
    int warm_start_reward_shaping;
    int blue_left;
    int reset_setup;
    int action_mode;
    int last_goals_blue;
    int last_goals_red;
    unsigned char last_done;
    unsigned char has_terminal_render_state;
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
    float field_scale;
    float base_x_out_start;
    float base_x_out_end;
    float base_y_out_start;
    float base_y_out_end;
    float base_goal_half_h;
    int last_touch_team;
    int throw_in_active;
    int throw_in_player;
    /* Warm-start dense-shaping coefficients. Exposed as runtime parameters so
     * Optuna can tune the balance between dense bootstrap signal and the
     * sparse +/-1 goal reward. Only consulted when warm_start_reward_shaping
     * is set on the Env. Defaults (set in init_env_common if 0/negative) are
     * scaled down ~10x from the original hard-coded values so the goal
     * reward dominates discounted returns. */
    float shaping_distance_penalty;
    float shaping_touch_bonus;
    float shaping_velocity_bonus;
    /* Warm-start red placement selector. 0 = red clusters at endline corners
     * (easy bootstrap, blue has open lane to goal). 1 = red uses the regular
     * self-play formation (harder, matches main-game conditions). Only
     * consulted when warm_start_reward_shaping is set. The curriculum
     * controller toggles this via py_env_set_red_in_formation after the
     * field-scale ladder completes. */
    int warm_start_red_in_formation;
    /* Per-agent "touched ball this step" flag, cleared at the start of each step
       and set inside ball_check_hit. Used for dense warm-start reward shaping. */
    int ball_touched_this_step[MAX_PLAYERS];
    int obs_size;
    int state_size;
    uint32_t rng;
    Agent terminal_render_agents[MAX_PLAYERS];
    int terminal_render_num_steps;
    int terminal_render_blue_left;
    float terminal_render_ball_x;
    float terminal_render_ball_y;
    float terminal_render_ball_vx;
    float terminal_render_ball_vy;
    int terminal_render_goals_blue;
    int terminal_render_goals_red;
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

static float rand_normal(uint32_t* s, float mean, float sigma) {
    float u1 = randf(s, 1e-6f, 1.0f);
    float u2 = randf(s, 0.0f, 1.0f);
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    return mean + sigma * z;
}

static int obs_size_for_team(int n) {
    /* Per-focus observation breakdown (for n = players_per_team):
       - 1 time_left
       - 2 self x, y
       - 2 self cos(rot), sin(rot)
       - 2 self last_move, last_rot
       - 2 self vx, vy (world frame, team-signed)
       - 11 self id one-hot
       - 4 self stats + steer
       - 2 set-piece flags (throw_in_active, is_thrower)
       - 5 ball (visibility, distance, angle, rel_vx, rel_vy)
       - 12 per teammate and per opponent (9 rel-obs + 3 stats: kick, speed, turn),
         with (2n - 1) other agents total => 12 * (2n - 1).
       Self+ball block = 31; other-agent block = 24n - 12; total = 19 + 24n. */
    return 19 + 24*n;
}

static int state_size_for_team(int n) {
    return 7 + 43*n;
}

static void get_one_hot(int id, float* out11) {
    memset(out11, 0, sizeof(float) * 11);
    out11[id % 11] = 1.0f;
}

static int team_on_left(const Env* env, int team) {
    if (team == 0) return env->blue_left;
    return !env->blue_left;
}

static float field_half_width(const Env* env) {
    return fmaxf(fabsf(env->x_out_start), fabsf(env->x_out_end));
}

static float field_half_height(const Env* env) {
    return fmaxf(fabsf(env->y_out_start), fabsf(env->y_out_end));
}

static void clamp_live_state_to_field(Env* env) {
    for (int i = 0; i < env->num_players; i++) {
        Agent* a = &env->agents[i];
        a->x = clampf(a->x, env->x_out_start, env->x_out_end);
        a->y = clampf(a->y, env->y_out_start, env->y_out_end);
    }
    env->ball_x = clampf(env->ball_x, env->x_out_start, env->x_out_end);
    env->ball_y = clampf(env->ball_y, env->y_out_start, env->y_out_end);
}

static void apply_field_scale(Env* env, float scale) {
    float clamped = clampf(scale, 0.1f, 1.0f);
    env->field_scale = clamped;
    env->x_out_start = env->base_x_out_start * clamped;
    env->x_out_end = env->base_x_out_end * clamped;
    env->y_out_start = env->base_y_out_start * clamped;
    env->y_out_end = env->base_y_out_end * clamped;
    env->goal_half_h = fminf(env->base_goal_half_h, fabsf(env->y_out_end));
    clamp_live_state_to_field(env);
}

static void compute_formation_pos(int pidx, int players_per_team, float* out_x, float* out_y) {
    /* Deterministic formation for any team size.
     * Index 0 is the deepest position (likely emergent goalie under offside
     * pressure). Remaining players spread in rows on own half.
     * Coordinates are in field units (own half is x<0, to be mirrored). */
    if (pidx == 0) {
        /* Deepest position - near own goal */
        *out_x = -45.0f;
        *out_y = 0.0f;
        return;
    }
    int n = players_per_team - 1;  /* field players */
    if (n == 0) { *out_x = -25.0f; *out_y = 0.0f; return; }
    /* Arrange in columns (y-spread) and rows (x-depth).
     * Up to 4 players: single row. 5+: two rows. */
    int cols, rows;
    if (n <= 4) { cols = n; rows = 1; }
    else { cols = (n + 1) / 2; rows = 2; }
    int r = (pidx - 1) / cols;
    int c = (pidx - 1) % cols;
    /* Rows from x=-30 (defenders) to x=-15 (attackers) */
    *out_x = (rows == 1) ? -22.0f : (-30.0f + r * 15.0f);
    /* Columns spread from y=-20 to y=+20 */
    *out_y = (cols == 1) ? 0.0f : (-20.0f + (float)c * (40.0f / (float)(cols - 1)));
}

static void reset_field(Env* env) {
    float pos_noise = 0.05f;
    for (int i = 0; i < env->num_players; i++) {
        Agent* a = &env->agents[i];
        int place_left = team_on_left(env, a->team);

        /* Sample per-agent stats */
        a->stat_kick  = clampf(rand_normal(&env->rng, 1.0f, STAT_SIGMA), 0.5f, 1.5f);
        a->stat_speed = clampf(rand_normal(&env->rng, 1.0f, STAT_SIGMA), 0.5f, 1.5f);
        a->stat_turn  = clampf(rand_normal(&env->rng, 1.0f, STAT_SIGMA), 0.5f, 1.5f);
        a->steer_angle = 0.0f;

        /* Warm-start corner placement for red. Only used when shaping is
         * active AND the curriculum has not yet asked for formation red.
         * Cluster red on their defensive endline near the top/bottom
         * sidelines so they stay clear of the goalmouth. This makes the
         * bootstrap task tractable (blue has an open lane to goal) while
         * keeping every other env mechanic identical to self-play. Once the
         * curriculum reaches the formation stage, `warm_start_red_in_formation`
         * is flipped and this branch is skipped so red spawns at the regular
         * self-play positions. */
        if (env->warm_start_reward_shaping
                && !env->warm_start_red_in_formation
                && a->team == 1) {
            int red_idx = i - env->players_per_team;
            int row = red_idx / 2;
            float sign = (red_idx % 2 == 0) ? 1.0f : -1.0f;
            float pullback = fminf(0.2f * (float)row, 0.6f);
            float endline_x = place_left ? env->x_out_start : env->x_out_end;
            float y_edge = env->y_out_end * (0.85f - fminf(0.05f * (float)row, 0.3f));
            a->x = endline_x * (0.95f - pullback);
            a->y = sign * y_edge;
            a->rot = place_left ? 0.0f : (float)M_PI;
            a->last_move = 0.0f;
            a->last_rot = 0.0f;
            continue;
        }

        float new_x, new_y;
        if (env->reset_setup == 0) {
            if (env->num_players == 22) {
                int pidx = i % 11;
                new_x = (init_position_11[pidx][1] + randf(&env->rng, -pos_noise, pos_noise)) * 110.0f;
                new_y = (init_position_11[pidx][0] + randf(&env->rng, -pos_noise, pos_noise)) * 110.0f;
            } else {
                int pidx = i % env->players_per_team;
                compute_formation_pos(pidx, env->players_per_team, &new_x, &new_y);
                new_x += randf(&env->rng, -pos_noise, pos_noise) * 110.0f;
                new_y += randf(&env->rng, -pos_noise, pos_noise) * 110.0f;
            }
            a->x = clampf(new_x, env->x_out_start, 0.0f);
            a->y = clampf(new_y, env->y_out_start, env->y_out_end);
        } else {
            a->x = randf(&env->rng, env->x_out_start, 0.0f);
            a->y = randf(&env->rng, env->y_out_start, env->y_out_end);
        }

        a->rot = place_left ? 0.0f : (float)M_PI;
        if (!place_left) {
            a->x *= -1.0f;
            a->y *= -1.0f;
        }
        a->last_move = 0.0f;
        a->last_rot = 0.0f;
    }

    /* Ball at center */
    env->ball_vx = 0.0f;
    env->ball_vy = 0.0f;
    env->ball_x = 0.0f;
    env->ball_y = 0.0f;

    env->throw_in_active = 0;
    env->throw_in_player = -1;
    env->last_touch_team = -1;
}

static void full_reset(Env* env, int hard_reset_score) {
    env->num_steps = 0;
    env->cumulative_episode_return = 0.0f;
    env->cumulative_blue_team_episode_return = 0.0f;
    env->cumulative_red_team_episode_return = 0.0f;
    env->throw_in_active = 0;
    env->throw_in_player = -1;
    env->last_touch_team = -1;
    if (hard_reset_score) {
        env->goals_blue = 0;
        env->goals_red = 0;
        if (env->do_team_switch) {
            env->blue_left = !env->blue_left;
        }
    }
    reset_field(env);
}

static void capture_terminal_render_state(Env* env) {
    memcpy(
        env->terminal_render_agents,
        env->agents,
        sizeof(Agent) * env->num_players
    );
    env->terminal_render_num_steps = env->num_steps;
    env->terminal_render_blue_left = env->blue_left;
    env->terminal_render_ball_x = env->ball_x;
    env->terminal_render_ball_y = env->ball_y;
    env->terminal_render_ball_vx = env->ball_vx;
    env->terminal_render_ball_vy = env->ball_vy;
    env->terminal_render_goals_blue = env->goals_blue;
    env->terminal_render_goals_red = env->goals_red;
    env->has_terminal_render_state = 1;
}

static void clear_terminal_render_state(Env* env) {
    env->has_terminal_render_state = 0;
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

static float discrete_kick_scale(int action) {
    static const float kick_scales[DISCRETE_KICK_BUCKETS] = {
        0.1f, 0.22857143f, 0.35714287f, 0.4857143f,
        0.6142857f, 0.74285716f, 0.87142855f, 1.0f
    };
    int kick_idx = action - DISCRETE_KICK_ACTION_START;
    if (kick_idx < 0 || kick_idx >= DISCRETE_KICK_BUCKETS) {
        return 0.0f;
    }
    return kick_scales[kick_idx];
}

static void ball_check_hit(Env* env, const float* kick_scales) {
    const float ball_radius = 1.0f;
    const float body_speed = 0.6f;
    const float leg_speed = 4.0f;
    const float agent_radius = 1.0f;
    const float leg_length = 1.0f;

    for (int i = 0; i < env->num_players; i++) {
        Agent* a = &env->agents[i];
        float d = dist2(env->ball_x, env->ball_y, a->x, a->y);
        int touched = 0;
        if (d < ball_radius + agent_radius) {
            float x_diff = env->ball_x - a->x;
            float y_diff = env->ball_y - a->y;
            env->ball_vx += body_speed * x_diff / (d + 1e-4f);
            env->ball_vy += body_speed * y_diff / (d + 1e-4f);
            touched = 1;
        }

        if (d < ball_radius + leg_length) {
            float delta, cc, ss;
            calc_line_ball_stats(a->rot, ball_radius, env->ball_x, env->ball_y, a->x, a->y, &delta, &cc, &ss);
            if (delta >= 0.0f && d < 2.0f * agent_radius + 2.0f * ball_radius) {
                float ks = kick_scales[i] * a->stat_kick;
                env->ball_vx += leg_speed * ks * cc;
                env->ball_vy += leg_speed * ks * ss;
                touched = 1;

                /* End throw-in only when the throwing player actually kicks */
                if (env->throw_in_active && i == env->throw_in_player
                        && kick_scales[i] > 0.0f) {
                    env->throw_in_active = 0;
                    env->throw_in_player = -1;
                }
            }
        }

        if (touched) {
            env->last_touch_team = a->team;
            env->ball_touched_this_step[i] = 1;
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

static void rel_obs_agent(const Env* env, const Agent* focus, const Agent* other, float* out9) {
    float obj_rot = atan2f(other->y - focus->y, other->x - focus->x);
    float obj_view = 0.0f;
    if (!visible(focus->rot, obj_rot, env->vision_range, &obj_view)) {
        memset(out9, 0, sizeof(float) * 9);
        return;
    }

    float max_dist = dist2(0, 0, field_half_width(env), field_half_height(env));
    out9[0] = 1.0f;
    out9[1] = dist2(focus->x, focus->y, other->x, other->y) / max_dist;
    out9[2] = obj_view / (env->vision_range / 2.0f);

    float rot = other->rot;
    float rel_x = cosf(rot - focus->rot);
    float rel_y = sinf(rot - focus->rot);
    out9[3] = rel_x;
    out9[4] = rel_y;
    out9[5] = other->last_move;
    out9[6] = other->last_rot;

    /* Ego-frame velocity of `other`. Because the bicycle model updates position
       as v = last_move * move_speed * stat_speed applied along the new heading,
       the actual world-frame velocity just applied is well-approximated by the
       last action scaled by move_speed * stat_speed. Exposing this directly
       saves the policy from having to infer it from commands + unobserved
       stats, which is especially important because `other->stat_speed` is not
       otherwise visible in the focus agent's observation. */
    float other_v = other->last_move * env->move_speed * other->stat_speed;
    float other_vx = other_v * cosf(other->rot);
    float other_vy = other_v * sinf(other->rot);
    /* Rotate into focus's ego frame so the features are invariant to the
       focus agent's heading (matching the convention used by the ball's
       relative-velocity features). */
    float cos_f = cosf(focus->rot);
    float sin_f = sinf(focus->rot);
    float ego_vx = cos_f * other_vx + sin_f * other_vy;
    float ego_vy = -sin_f * other_vx + cos_f * other_vy;
    /* Normalize by the peak plausible agent speed (move_speed * max stat_speed).
       stat_speed lives in a narrow sampled band, so move_speed alone is a safe
       upper bound divisor and keeps the features near [-1, 1]. */
    float norm = env->move_speed > 1e-6f ? env->move_speed : 1.0f;
    out9[7] = ego_vx / norm;
    out9[8] = ego_vy / norm;
}

static void rel_obs_ball(const Env* env, const Agent* focus, float* out5) {
    float obj_rot = atan2f(env->ball_y - focus->y, env->ball_x - focus->x);
    float obj_view = 0.0f;
    if (!visible(focus->rot, obj_rot, env->vision_range, &obj_view)) {
        memset(out5, 0, sizeof(float) * 5);
        return;
    }

    float max_dist = dist2(0, 0, field_half_width(env), field_half_height(env));
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
    float half_width = field_half_width(env);
    float half_height = field_half_height(env);
    float* state = env->global_states;
    float time_left = 1.0f - ((float)env->num_steps / (float)env->game_length);
    float* state_base = state;

    /* Global state header: time, ball, throw-in info */
    state_base[0] = time_left;
    state_base[1] = sign_state * env->ball_x / half_width;
    state_base[2] = sign_state * env->ball_y / half_height;
    state_base[3] = sign_state * env->ball_vx / MAX_BALL_SPEED;
    state_base[4] = sign_state * env->ball_vy / MAX_BALL_SPEED;
    state_base[5] = env->throw_in_active ? 1.0f : 0.0f;
    state_base[6] = (float)env->throw_in_player / (float)(np > 0 ? np : 1);

    int sidx = 7;
    for (int i = 0; i < np; i++) {
        Agent* a = &env->agents[i];
        float onehot[11];
        get_one_hot(i, onehot);
        state_base[sidx++] = sign_state * a->x / half_width;
        state_base[sidx++] = sign_state * a->y / half_height;
        state_base[sidx++] = sign_state * cosf(a->rot);
        state_base[sidx++] = sign_state * sinf(a->rot);
        state_base[sidx++] = a->last_move;
        state_base[sidx++] = a->last_rot;
        for (int k = 0; k < 11; k++) state_base[sidx++] = onehot[k];
        state_base[sidx++] = a->stat_kick;
        state_base[sidx++] = a->stat_speed;
        state_base[sidx++] = a->stat_turn;
        state_base[sidx++] = a->steer_angle / MAX_STEER_ANGLE;
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
        out[o++] = sign * focus->x / half_width;
        out[o++] = sign * focus->y / half_height;
        out[o++] = sign * cosf(focus->rot);
        out[o++] = sign * sinf(focus->rot);
        out[o++] = focus->last_move;
        out[o++] = focus->last_rot;

        /* Self world-frame velocity from last applied action, sign-flipped with
           the team so symmetric self-play sees a consistent canonical frame.
           See rel_obs_agent for the velocity derivation. */
        float self_v = focus->last_move * env->move_speed * focus->stat_speed;
        float self_vx = self_v * cosf(focus->rot);
        float self_vy = self_v * sinf(focus->rot);
        float v_norm = env->move_speed > 1e-6f ? env->move_speed : 1.0f;
        out[o++] = sign * self_vx / v_norm;
        out[o++] = sign * self_vy / v_norm;

        float onehot[11];
        get_one_hot(i, onehot);
        for (int k = 0; k < 11; k++) out[o++] = onehot[k];

        /* Ego state fields: stats, steer, set-piece (throw-in or free-kick) */
        out[o++] = focus->stat_kick;
        out[o++] = focus->stat_speed;
        out[o++] = focus->stat_turn;
        out[o++] = focus->steer_angle / MAX_STEER_ANGLE;
        out[o++] = env->throw_in_active ? 1.0f : 0.0f;
        out[o++] = (env->throw_in_active && env->throw_in_player == i) ? 1.0f : 0.0f;

        float brobs[5];
        rel_obs_ball(env, focus, brobs);
        for (int k = 0; k < 5; k++) out[o++] = brobs[k];

        for (int j = 0; j < np; j++) {
            if (j == i) continue;
            if (env->agents[j].team != focus->team) continue;
            float aobs[9];
            rel_obs_agent(env, focus, &env->agents[j], aobs);
            for (int k = 0; k < 9; k++) out[o++] = aobs[k];
            /* Per-other-agent stats: each player can read its teammates' and
               opponents' kick/speed/turn so policies can reason about skill
               asymmetries that currently only the self-vector exposes. */
            out[o++] = env->agents[j].stat_kick;
            out[o++] = env->agents[j].stat_speed;
            out[o++] = env->agents[j].stat_turn;
        }
        for (int j = 0; j < np; j++) {
            if (env->agents[j].team == focus->team) continue;
            float aobs[9];
            rel_obs_agent(env, focus, &env->agents[j], aobs);
            for (int k = 0; k < 9; k++) out[o++] = aobs[k];
            out[o++] = env->agents[j].stat_kick;
            out[o++] = env->agents[j].stat_speed;
            out[o++] = env->agents[j].stat_turn;
        }

        float r = 0.0f;
        if (goal_scored_team >= 0) {
            r = (focus->team == goal_scored_team) ? 1.0f : -1.0f;
        }

        /* Dense reward shaping applied during warm-start only (and only to the
           team being trained, team 0). The goal is to give the policy a
           learnable signal long before it stumbles on a full scoring sequence.
           We keep the shaping small relative to the +/-1 goal reward so the
           policy still prefers finishing a goal over hovering near the ball,
           and we deliberately do NOT apply it during self-play so learned
           self-play behavior is not biased by hand-tuned shaping terms. */
        if (env->warm_start_reward_shaping && focus->team == 0) {
            float half_width_s = field_half_width(env);
            float half_height_s = field_half_height(env);
            float max_dist_s = dist2(0, 0, half_width_s, half_height_s);
            float d_ball = dist2(focus->x, focus->y, env->ball_x, env->ball_y);
            /* Distance-to-ball penalty: keeps blue close to the ball early.
               Coefficient tunable; default scaled so full-field distance
               contributes ~-0.0001 per step. */
            r += -env->shaping_distance_penalty * (d_ball / (max_dist_s + 1e-6f));

            /* Per-touch bonus so the policy gets a strong signal the first
               time it manages to contact the ball. Coefficient tunable. */
            if (env->ball_touched_this_step[i]) {
                r += env->shaping_touch_bonus;
            }

            /* Bonus for ball velocity toward the opponent goal. Coefficient
               tunable; magnitude capped by MAX_BALL_SPEED so the term stays
               small relative to the +/-1 goal reward. */
            float goal_sign = team_on_left(env, 0) ? 1.0f : -1.0f;
            float ball_toward_goal = goal_sign * env->ball_vx;
            if (ball_toward_goal > 0.0f) {
                r += env->shaping_velocity_bonus * (ball_toward_goal / MAX_BALL_SPEED);
            }
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
    int warm_start_reward_shaping,
    float shaping_distance_penalty,
    float shaping_touch_bonus,
    float shaping_velocity_bonus,
    float vision_range,
    int reset_setup
) {
    memset(&env->log, 0, sizeof(Log));
    env->players_per_team = players_per_team;
    env->num_players = players_per_team * 2;
    env->game_length = game_length;
    env->do_team_switch = do_team_switch;
    env->warm_start_reward_shaping = warm_start_reward_shaping;
    /* Defaults match the original hard-coded coefficients that produced
     * working warm-start runs before the scaling experiment. Negative values
     * fall back to these defaults so callers can pass 0 to zero out one
     * shaping term without affecting the rest. */
    env->shaping_distance_penalty = shaping_distance_penalty >= 0.0f
        ? shaping_distance_penalty : 0.001f;
    env->shaping_touch_bonus = shaping_touch_bonus >= 0.0f
        ? shaping_touch_bonus : 0.05f;
    env->shaping_velocity_bonus = shaping_velocity_bonus >= 0.0f
        ? shaping_velocity_bonus : 0.01f;
    env->warm_start_red_in_formation = 0;
    env->blue_left = 1;
    env->reset_setup = reset_setup;
    env->vision_range = vision_range;
    env->action_mode = action_mode;
    env->obs_size = obs_size_for_team(players_per_team);
    env->state_size = state_size_for_team(players_per_team);
    env->last_goals_blue = 0;
    env->last_goals_red = 0;
    env->last_done = 0;
    env->has_terminal_render_state = 0;
    env->base_x_out_start = -50.0f;
    env->base_x_out_end = 50.0f;
    env->base_y_out_start = -35.0f;
    env->base_y_out_end = 35.0f;
    env->base_goal_half_h = 20.0f;
    env->x_out_start = env->base_x_out_start;
    env->x_out_end = env->base_x_out_end;
    env->y_out_start = env->base_y_out_start;
    env->y_out_end = env->base_y_out_end;
    env->goal_half_h = env->base_goal_half_h;
    env->rot_speed = 0.4f;
    env->move_speed = 1.0f;
    env->last_touch_team = -1;
    env->throw_in_active = 0;
    env->throw_in_player = -1;
    env->rng = (uint32_t)(seed + 1);

    for (int a = 0; a < env->num_players; a++) {
        env->agents[a].team = (a < players_per_team) ? 0 : 1;
        env->agents[a].stat_kick = 1.0f;
        env->agents[a].stat_speed = 1.0f;
        env->agents[a].stat_turn = 1.0f;
        env->agents[a].steer_angle = 0.0f;
    }

    apply_field_scale(env, 1.0f);
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
    clear_terminal_render_state(env);
    clear_outputs(env);
    compute_observations(env, -1);
}

/* Check offside (canonical FIFA Law 11): offside line = second-to-last defender.
 * Returns the defending team (which gets the free kick) or -1 if no offside.
 * On detection, writes the offside player's position to *out_x, *out_y. */
static int check_offside(const Env* env, float* out_x, float* out_y) {
    if (env->throw_in_active) return -1;
    /* Disable offside while the warm-start dense shaping is on so the bootstrap
     * task isn't constantly interrupted by offside teleports against
     * stationary scripted red defenders. Self-play (shaping off) keeps the
     * full FIFA-style rule. */
    if (env->warm_start_reward_shaping) return -1;

    for (int team = 0; team < 2; team++) {
        int other_team = 1 - team;
        int attacks_right = team_on_left(env, team);
        int other_defends_left = team_on_left(env, other_team);
        int other_start = other_team == 0 ? 0 : env->players_per_team;
        int other_end = other_start + env->players_per_team;

        /* Find the two deepest defenders of other_team. The offside line is at
         * the position of the LESS-deep one (the second-to-last defender).
         * "Deeper" means closer to one's own goal. */
        float deepest = 0.0f, second = 0.0f;
        int n_found = 0;
        for (int j = other_start; j < other_end; j++) {
            float xj = env->agents[j].x;
            if (n_found == 0) {
                deepest = xj;
                n_found = 1;
            } else if (n_found == 1) {
                if (other_defends_left ? (xj < deepest) : (xj > deepest)) {
                    second = deepest;
                    deepest = xj;
                } else {
                    second = xj;
                }
                n_found = 2;
            } else {
                if (other_defends_left ? (xj < deepest) : (xj > deepest)) {
                    second = deepest;
                    deepest = xj;
                } else if (other_defends_left ? (xj < second) : (xj > second)) {
                    second = xj;
                }
            }
        }
        if (n_found < 2) continue;  /* Need at least two defenders for offside */

        float offside_line = second;

        /* Check each attacker on this team. */
        int atk_start = team == 0 ? 0 : env->players_per_team;
        int atk_end = atk_start + env->players_per_team;
        for (int i = atk_start; i < atk_end; i++) {
            float ax = env->agents[i].x;
            /* Must be in opposing half. */
            if (attacks_right && ax <= 0.0f) continue;
            if (!attacks_right && ax >= 0.0f) continue;
            /* Must be past the second-to-last defender. */
            int caught_by_defender = (attacks_right && ax > offside_line)
                                   || (!attacks_right && ax < offside_line);
            if (!caught_by_defender) continue;
            /* FIFA Law 11 also requires the attacker to be ahead of the ball.
             * Without this check the ball-carrier itself is constantly flagged
             * offside as soon as they cross midfield (since their x equals or
             * exceeds the ball's x). It also wrongly catches teammates who are
             * still BEHIND the ball — they have to be in front of the ball to
             * be in an offside position. */
            int caught_by_ball = (attacks_right && ax > env->ball_x)
                              || (!attacks_right && ax < env->ball_x);
            if (!caught_by_ball) continue;
            *out_x = ax;
            *out_y = env->agents[i].y;
            return other_team;
        }
    }
    return -1;
}

static void c_step(Env* env) {
    int np = env->num_players;
    float kick_scales[MAX_PLAYERS];
    float step_reward_sum = 0.0f;
    float step_reward_sum_blue = 0.0f;
    float step_reward_sum_red = 0.0f;
    clear_terminal_render_state(env);
    env->num_steps += 1;
    clear_outputs(env);
    /* Reset per-step ball-touch flags; ball_check_hit below will set them for
       any agent that actually contacts the ball during this physics tick. */
    memset(env->ball_touched_this_step, 0, sizeof(env->ball_touched_this_step));

    for (int i = 0; i < np; i++) {
        Agent* a = &env->agents[i];
        float move = 0.0f;
        float rot = 0.0f;

        if (env->action_mode == 0) {
            int* atn = (int*)env->actions;
            int action = atn[i];
            if (action < 0) action = 0;
            if (action >= DISCRETE_ACTION_COUNT) action = DISCRETE_ACTION_COUNT - 1;
            kick_scales[i] = discrete_kick_scale(action);

            switch (action) {
                case DISCRETE_ACTION_MOVE_FORWARD:
                    move = 1.0f;
                    break;
                case DISCRETE_ACTION_MOVE_BACKWARD:
                    move = -1.0f;
                    break;
                case DISCRETE_ACTION_ROTATE_LEFT:
                    rot = -1.0f;
                    break;
                case DISCRETE_ACTION_ROTATE_RIGHT:
                    rot = 1.0f;
                    break;
                default:
                    break;
            }
        } else {
            float* atn = (float*)env->actions;
            float* arow = atn + (i * 2);
            move = clampf(arow[0], -1.0f, 1.0f);
            rot = clampf(arow[1], -1.0f, 1.0f);
            kick_scales[i] = 1.0f;
        }

        /* Throw-in: locked player can only rotate in place and kick. */
        if (env->throw_in_active && i == env->throw_in_player) {
            move = 0.0f;
            /* Direct rotation (bypass bicycle model). */
            a->last_move = 0.0f;
            a->last_rot = rot;
            a->rot -= rot * env->rot_speed * a->stat_turn;
            if (a->rot > 2.0f*(float)M_PI) a->rot -= 2.0f*(float)M_PI;
            else if (a->rot < 0.0f) a->rot += 2.0f*(float)M_PI;
            continue;
        }

        a->last_move = move;
        a->last_rot = rot;

        /* Bicycle model: ROTATE changes steer_angle, heading changes when moving. */
        a->steer_angle += rot * STEER_RATE * a->stat_turn;
        a->steer_angle = clampf(a->steer_angle, -MAX_STEER_ANGLE, MAX_STEER_ANGLE);

        float v = move * env->move_speed * a->stat_speed;

        if (fabsf(v) > 1e-6f) {
            float dtheta = (v / WHEELBASE) * tanf(a->steer_angle);
            a->rot += dtheta;
            if (a->rot > 2.0f*(float)M_PI) a->rot -= 2.0f*(float)M_PI;
            else if (a->rot < 0.0f) a->rot += 2.0f*(float)M_PI;
        }

        a->x = clampf(a->x + v * cosf(a->rot), env->x_out_start, env->x_out_end);
        a->y = clampf(a->y + v * sinf(a->rot), env->y_out_start, env->y_out_end);
    }

    ball_check_hit(env, kick_scales);

    env->ball_x += env->ball_vx;
    env->ball_y += env->ball_vy;

    env->ball_vx *= BALL_VELOCITY_DECAY;
    env->ball_vy *= BALL_VELOCITY_DECAY;
    if (speed2(env->ball_vx, env->ball_vy) < 0.01f) {
        env->ball_vx = 0.0f;
        env->ball_vy = 0.0f;
    }

    /* --- Goal detection (x-boundary within goal posts) --- */
    int goal_scored = -1;
    if (fabsf(env->ball_y) <= env->goal_half_h) {
        if (env->ball_x < env->x_out_start) {
            goal_scored = team_on_left(env, 0) ? 1 : 0;
            if (goal_scored == 0) env->goals_blue += 1;
            else env->goals_red += 1;
        } else if (env->ball_x > env->x_out_end) {
            goal_scored = team_on_left(env, 0) ? 0 : 1;
            if (goal_scored == 0) env->goals_blue += 1;
            else env->goals_red += 1;
        }
    } else {
        /* x-boundary outside goal posts: bounce */
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

    /* --- Sideline (y-boundary): throw-in instead of bounce ---
     *
     * The ball clamp runs unconditionally so the ball cannot drift outside the
     * field regardless of the current set-piece state. Only the thrower-setup
     * (teleport + velocity zero) and the no-last-touch bounce remain gated on
     * !throw_in_active, so a stray body impulse during an active throw-in
     * cannot leak the ball past the sideline. */
    if (env->ball_y < env->y_out_start || env->ball_y > env->y_out_end) {
        env->ball_y = clampf(env->ball_y, env->y_out_start, env->y_out_end);
        if (!env->throw_in_active) {
            if (env->last_touch_team >= 0) {
                env->ball_vx = 0.0f;
                env->ball_vy = 0.0f;

                int opposing_team = 1 - env->last_touch_team;
                int best = -1;
                float best_dist = 1e9f;
                for (int i = 0; i < np; i++) {
                    if (env->agents[i].team != opposing_team) continue;
                    float d = dist2(env->ball_x, env->ball_y,
                                    env->agents[i].x, env->agents[i].y);
                    if (d < best_dist) { best_dist = d; best = i; }
                }
                if (best >= 0) {
                    env->agents[best].x = env->ball_x;
                    env->agents[best].y = env->ball_y;
                    env->agents[best].steer_angle = 0.0f;
                    env->throw_in_active = 1;
                    env->throw_in_player = best;
                }
            } else {
                /* No last touch known: bounce */
                env->ball_vy = -env->ball_vy;
                env->ball_vx *= 0.6f;
                env->ball_vy *= 0.6f;
            }
        }
    }

    /* --- Offside check: triggers an indirect free kick to the defending team
     *     (no goal scored). Reuses the throw-in player-lock plumbing. --- */
    if (goal_scored < 0 && !env->throw_in_active) {
        float off_x = 0.0f, off_y = 0.0f;
        int defending_team = check_offside(env, &off_x, &off_y);
        if (defending_team >= 0) {
            /* Place the ball at the offside spot. */
            env->ball_x = clampf(off_x, env->x_out_start, env->x_out_end);
            env->ball_y = clampf(off_y, env->y_out_start, env->y_out_end);
            env->ball_vx = 0.0f;
            env->ball_vy = 0.0f;

            /* Find nearest defender to the ball; teleport them to it. */
            int best = -1;
            float best_dist = 1e9f;
            int def_start = defending_team == 0 ? 0 : env->players_per_team;
            int def_end = def_start + env->players_per_team;
            for (int j = def_start; j < def_end; j++) {
                float d = dist2(env->ball_x, env->ball_y,
                                env->agents[j].x, env->agents[j].y);
                if (d < best_dist) { best_dist = d; best = j; }
            }
            if (best >= 0) {
                env->agents[best].x = env->ball_x;
                env->agents[best].y = env->ball_y;
                env->agents[best].steer_angle = 0.0f;
                env->throw_in_active = 1;     /* reuse lock mechanic */
                env->throw_in_player = best;
                env->last_touch_team = -1;
            }
        }
    }

    if (goal_scored >= 0) {
        reset_field(env);
    }

    int done = (env->num_steps >= env->game_length);
    compute_observations(env, goal_scored);
    for (int i = 0; i < np; i++) {
        step_reward_sum += env->rewards[i];
        if (env->agents[i].team == 0) step_reward_sum_blue += env->rewards[i];
        else step_reward_sum_red += env->rewards[i];
    }
    env->cumulative_episode_return += step_reward_sum;
    env->cumulative_blue_team_episode_return += step_reward_sum_blue;
    env->cumulative_red_team_episode_return += step_reward_sum_red;

    if (done) {
        int score_diff;
        for (int i = 0; i < np; i++) {
            env->terminals[i] = 1;
        }
        env->last_goals_blue = env->goals_blue;
        env->last_goals_red = env->goals_red;
        env->last_done = 1;
        score_diff = env->goals_blue - env->goals_red;
        env->log.score += (float)score_diff;
        if (score_diff > 0) env->log.wins_blue += 1.0f;
        else if (score_diff < 0) env->log.wins_red += 1.0f;
        else env->log.draws += 1.0f;
        env->log.episode_return += env->cumulative_episode_return;
        env->log.blue_team_episode_return += env->cumulative_blue_team_episode_return;
        env->log.red_team_episode_return += env->cumulative_red_team_episode_return;
        env->log.episode_length += env->num_steps;
        env->log.n += 1.0f;
        capture_terminal_render_state(env);
        full_reset(env, 1);
    }
}

static PyObject* build_state_dict(const Env* env, const Agent* agents, float ball_x, float ball_y,
    float ball_vx, float ball_vy, int goals_blue, int goals_red, int num_steps, int blue_left) {
    npy_intp pos_dims[2] = {env->num_players, 2};
    npy_intp rot_dims[1] = {env->num_players};

    PyObject* pos = PyArray_SimpleNew(2, pos_dims, NPY_FLOAT32);
    PyObject* rot = PyArray_SimpleNew(1, rot_dims, NPY_FLOAT32);
    if (!pos || !rot) {
        Py_XDECREF(pos);
        Py_XDECREF(rot);
        return NULL;
    }

    float* pdat = (float*)PyArray_DATA((PyArrayObject*)pos);
    float* rdat = (float*)PyArray_DATA((PyArrayObject*)rot);
    for (int i = 0; i < env->num_players; i++) {
        pdat[i*2] = agents[i].x;
        pdat[i*2 + 1] = agents[i].y;
        rdat[i] = agents[i].rot;
    }

    PyObject* ball = Py_BuildValue("(ffff)", ball_x, ball_y, ball_vx, ball_vy);
    PyObject* goals = Py_BuildValue("(ii)", goals_blue, goals_red);
    PyObject* d = PyDict_New();
    PyObject* num_steps_obj = PyLong_FromLong(num_steps);
    PyObject* blue_left_obj = PyBool_FromLong(blue_left);
    PyObject* field_scale_obj = PyFloat_FromDouble(env->field_scale);
    if (!ball || !goals || !d || !num_steps_obj || !blue_left_obj) {
        Py_XDECREF(pos);
        Py_XDECREF(rot);
        Py_XDECREF(ball);
        Py_XDECREF(goals);
        Py_XDECREF(d);
        Py_XDECREF(num_steps_obj);
        Py_XDECREF(blue_left_obj);
        Py_XDECREF(field_scale_obj);
        return NULL;
    }

    if (PyDict_SetItemString(d, "positions", pos) < 0 ||
        PyDict_SetItemString(d, "rotations", rot) < 0 ||
        PyDict_SetItemString(d, "ball", ball) < 0 ||
        PyDict_SetItemString(d, "goals", goals) < 0 ||
        PyDict_SetItemString(d, "num_steps", num_steps_obj) < 0 ||
        PyDict_SetItemString(d, "blue_left", blue_left_obj) < 0 ||
        PyDict_SetItemString(d, "field_scale", field_scale_obj) < 0) {
        Py_DECREF(pos);
        Py_DECREF(rot);
        Py_DECREF(ball);
        Py_DECREF(goals);
        Py_DECREF(d);
        Py_DECREF(num_steps_obj);
        Py_DECREF(blue_left_obj);
        Py_DECREF(field_scale_obj);
        return NULL;
    }

    Py_DECREF(pos);
    Py_DECREF(rot);
    Py_DECREF(ball);
    Py_DECREF(goals);
    Py_DECREF(num_steps_obj);
    Py_DECREF(blue_left_obj);
    Py_DECREF(field_scale_obj);
    return d;
}

static int assign_log_dict(PyObject* d, const Log* log) {
    if (PyDict_SetItemString(d, "score", PyFloat_FromDouble(log->score / log->n)) < 0) return -1;
    if (PyDict_SetItemString(d, "score_diff", PyFloat_FromDouble(log->score / log->n)) < 0) return -1;
    if (PyDict_SetItemString(d, "episode_return", PyFloat_FromDouble(log->episode_return / log->n)) < 0) return -1;
    if (PyDict_SetItemString(d, "blue_team_episode_return", PyFloat_FromDouble(log->blue_team_episode_return / log->n)) < 0) return -1;
    if (PyDict_SetItemString(d, "red_team_episode_return", PyFloat_FromDouble(log->red_team_episode_return / log->n)) < 0) return -1;
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
    int seed, players_per_team, game_length, action_mode, do_team_switch, warm_start_reward_shaping, reset_setup;
    float vision_range;
    float shaping_distance_penalty = -1.0f;
    float shaping_touch_bonus = -1.0f;
    float shaping_velocity_bonus = -1.0f;

    static char* kwlist[] = {
        "observations", "actions", "rewards", "terminals", "truncations", "global_states",
        "seed", "players_per_team", "game_length", "action_mode", "do_team_switch",
        "warm_start_reward_shaping", "vision_range", "reset_setup",
        "shaping_distance_penalty", "shaping_touch_bonus", "shaping_velocity_bonus",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOOOiiiiiifi|fff", kwlist,
            &obs_obj, &act_obj, &rew_obj, &term_obj, &trunc_obj, &state_obj,
            &seed, &players_per_team, &game_length, &action_mode,
            &do_team_switch, &warm_start_reward_shaping, &vision_range, &reset_setup,
            &shaping_distance_penalty, &shaping_touch_bonus, &shaping_velocity_bonus)) {
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
    init_env_common(
        env,
        seed,
        players_per_team,
        game_length,
        action_mode,
        do_team_switch,
        warm_start_reward_shaping,
        shaping_distance_penalty,
        shaping_touch_bonus,
        shaping_velocity_bonus,
        vision_range,
        reset_setup
    );

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

static PyObject* py_env_set_field_scale(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    float scale;
    if (!PyArg_ParseTuple(args, "Of", &handle_obj, &scale)) {
        return NULL;
    }
    Env* env = unpack_env_handle(handle_obj);
    if (!env) return NULL;
    apply_field_scale(env, scale);
    compute_observations(env, -1);
    Py_RETURN_NONE;
}

static PyObject* py_env_set_red_in_formation(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    int in_formation;
    if (!PyArg_ParseTuple(args, "Oi", &handle_obj, &in_formation)) {
        return NULL;
    }
    Env* env = unpack_env_handle(handle_obj);
    if (!env) return NULL;
    env->warm_start_red_in_formation = in_formation ? 1 : 0;
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

    if (env->has_terminal_render_state) {
        return build_state_dict(
            env,
            env->terminal_render_agents,
            env->terminal_render_ball_x,
            env->terminal_render_ball_y,
            env->terminal_render_ball_vx,
            env->terminal_render_ball_vy,
            env->terminal_render_goals_blue,
            env->terminal_render_goals_red,
            env->terminal_render_num_steps,
            env->terminal_render_blue_left
        );
    }

    return build_state_dict(
        env,
        env->agents,
        env->ball_x,
        env->ball_y,
        env->ball_vx,
        env->ball_vy,
        env->goals_blue,
        env->goals_red,
        env->num_steps,
        env->blue_left
    );
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
    int num_envs, seed, players_per_team, game_length, action_mode, do_team_switch, warm_start_reward_shaping, reset_setup;
    float vision_range;
    float shaping_distance_penalty = -1.0f;
    float shaping_touch_bonus = -1.0f;
    float shaping_velocity_bonus = -1.0f;

    static char* kwlist[] = {
        "observations", "actions", "rewards", "terminals", "truncations", "global_states",
        "num_envs", "seed", "players_per_team", "game_length", "action_mode", "do_team_switch",
        "warm_start_reward_shaping", "vision_range", "reset_setup",
        "shaping_distance_penalty", "shaping_touch_bonus", "shaping_velocity_bonus",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOOOiiiiiiifi|fff", kwlist,
            &obs_obj, &act_obj, &rew_obj, &term_obj, &trunc_obj, &state_obj,
            &num_envs, &seed, &players_per_team, &game_length, &action_mode,
            &do_team_switch, &warm_start_reward_shaping, &vision_range, &reset_setup,
            &shaping_distance_penalty, &shaping_touch_bonus, &shaping_velocity_bonus)) {
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
        init_env_common(
            env,
            seed + i*7919,
            players_per_team,
            game_length,
            action_mode,
            do_team_switch,
            warm_start_reward_shaping,
            shaping_distance_penalty,
            shaping_touch_bonus,
            shaping_velocity_bonus,
            vision_range,
            reset_setup
        );
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

static PyObject* py_vec_set_field_scale(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    float scale;
    if (!PyArg_ParseTuple(args, "Of", &handle_obj, &scale)) {
        return NULL;
    }
    Vec* vec = unpack_vec_handle(handle_obj);
    if (!vec) return NULL;
    for (int i = 0; i < vec->num_envs; i++) {
        apply_field_scale(&vec->envs[i], scale);
        compute_observations(&vec->envs[i], -1);
    }
    Py_RETURN_NONE;
}

static PyObject* py_vec_set_red_in_formation(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    int in_formation;
    if (!PyArg_ParseTuple(args, "Oi", &handle_obj, &in_formation)) {
        return NULL;
    }
    Vec* vec = unpack_vec_handle(handle_obj);
    if (!vec) return NULL;
    for (int i = 0; i < vec->num_envs; i++) {
        vec->envs[i].warm_start_red_in_formation = in_formation ? 1 : 0;
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
        aggregate.blue_team_episode_return += log->blue_team_episode_return;
        aggregate.red_team_episode_return += log->red_team_episode_return;
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
    {"env_set_field_scale", py_env_set_field_scale, METH_VARARGS, "Set one env field scale"},
    {"env_set_red_in_formation", py_env_set_red_in_formation, METH_VARARGS, "Toggle warm-start red formation placement"},
    {"env_log", py_env_log, METH_VARARGS, "Get one env log"},
    {"env_get_last_scores", py_env_get_last_scores, METH_VARARGS, "Get last scalar env scores"},
    {"env_get_state", py_env_get_state, METH_VARARGS, "Get one env state"},
    {"env_close", py_env_close, METH_VARARGS, "Close one env"},
    {"vec_init", (PyCFunction)py_vec_init, METH_VARARGS | METH_KEYWORDS, "Initialize vector env"},
    {"vec_reset", py_vec_reset, METH_VARARGS, "Reset vector env"},
    {"vec_step", py_vec_step, METH_VARARGS, "Step vector env"},
    {"vec_set_field_scale", py_vec_set_field_scale, METH_VARARGS, "Set vector env field scale"},
    {"vec_set_red_in_formation", py_vec_set_red_in_formation, METH_VARARGS, "Toggle warm-start red formation placement"},
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
