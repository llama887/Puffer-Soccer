from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pygame


class SoccerRenderer:
    def __init__(self, render_mode: str = "human", wait_period: float = 0.04):
        self.render_mode = render_mode
        self.wait_period = wait_period
        self._inited = False

        self.field_size = (110.0, 76.0)
        self.in_field_size = (100.0, 70.0)
        self.goal_size = (3.0, 40.0)
        self.r_to_s = (1080 / 1.5) / self.field_size[1]

        self.screen_size = (int(self.field_size[0] * self.r_to_s), int(self.field_size[1] * self.r_to_s))

    def _ensure(self):
        if self._inited:
            return
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Football 2D (Puffer)")
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = None
        self._inited = True

    def close(self):
        if self._inited:
            pygame.quit()
            self._inited = False

    def _draw(self, state: dict[str, Any]) -> pygame.Surface:
        self._ensure()

        green = (0, 200, 50)
        white = (255, 255, 255)
        blue = (0, 0, 255)
        red = (255, 0, 0)
        yellow = (255, 255, 0)

        field = pygame.Surface(self.screen_size)
        field.fill(green)

        x_out_start = (self.field_size[0] / 2 - self.in_field_size[0] / 2) * self.r_to_s
        y_out_start = (self.field_size[1] / 2 - self.in_field_size[1] / 2) * self.r_to_s

        pygame.draw.rect(
            field,
            white,
            (x_out_start, y_out_start, self.in_field_size[0] * self.r_to_s, self.in_field_size[1] * self.r_to_s),
            2,
        )

        x_goal1 = (self.field_size[0] / 2 - self.in_field_size[0] / 2 - self.goal_size[0]) * self.r_to_s
        x_goal2 = (self.field_size[0] / 2 + self.in_field_size[0] / 2) * self.r_to_s
        y_goal = (self.field_size[1] / 2 - self.goal_size[1] / 2) * self.r_to_s

        pygame.draw.rect(field, white, (x_goal1, y_goal, self.goal_size[0] * self.r_to_s, self.goal_size[1] * self.r_to_s), 2)
        pygame.draw.rect(field, white, (x_goal2, y_goal, self.goal_size[0] * self.r_to_s, self.goal_size[1] * self.r_to_s), 2)

        center_x = (self.field_size[0] / 2) * self.r_to_s
        pygame.draw.line(
            field,
            white,
            (center_x, y_out_start),
            (center_x, y_out_start + self.in_field_size[1] * self.r_to_s),
            2,
        )

        positions = state["positions"]
        rotations = state["rotations"]
        n = positions.shape[0] // 2

        off_x = self.field_size[0] / 2
        off_y = self.field_size[1] / 2

        for i in range(positions.shape[0]):
            col = blue if i < n else red
            x = (positions[i, 0] + off_x) * self.r_to_s
            y = (positions[i, 1] + off_y) * self.r_to_s
            pygame.draw.circle(field, col, (int(x), int(y)), int(1.0 * self.r_to_s))

            rot = rotations[i]
            dx = math.cos(rot) * 1.0 * self.r_to_s
            dy = math.sin(rot) * 1.0 * self.r_to_s
            pygame.draw.line(field, yellow, (x, y), (x + dx, y + dy), 2)

        bx, by, _, _ = state["ball"]
        bx = (bx + off_x) * self.r_to_s
        by = (by + off_y) * self.r_to_s
        pygame.draw.circle(field, white, (int(bx), int(by)), int(1.0 * self.r_to_s))

        goals_blue, goals_red = state["goals"]
        font = pygame.font.SysFont("comicsansms", 48)
        text = font.render(f"{goals_blue} : {goals_red}", True, (0, 128, 0))
        field.blit(text, (self.field_size[0] * self.r_to_s / 2 - 32, 30))

        step_text = font.render(f"Step: {state['num_steps']}", True, (0, 128, 0))
        field.blit(step_text, (50, 30))

        return field

    @staticmethod
    def _circle(frame: np.ndarray, x: int, y: int, r: int, color: tuple[int, int, int]):
        h, w = frame.shape[:2]
        x0 = max(0, x - r)
        x1 = min(w, x + r + 1)
        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - x) * (xx - x) + (yy - y) * (yy - y) <= r * r
        frame[y0:y1, x0:x1][mask] = color

    @staticmethod
    def _line(
        frame: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
        thickness: int = 1,
    ):
        n = max(abs(x1 - x0), abs(y1 - y0), 1)
        xs = np.linspace(x0, x1, n + 1).astype(np.int32)
        ys = np.linspace(y0, y1, n + 1).astype(np.int32)
        for x, y in zip(xs, ys):
            SoccerRenderer._circle(frame, x, y, max(1, thickness // 2), color)

    def _draw_rgb_array(self, state: dict[str, Any]) -> np.ndarray:
        h, w = self.screen_size[1], self.screen_size[0]
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (0, 200, 50)

        white = (255, 255, 255)
        blue = (0, 0, 255)
        red = (255, 0, 0)
        yellow = (255, 255, 0)

        x_out_start = int((self.field_size[0] / 2 - self.in_field_size[0] / 2) * self.r_to_s)
        y_out_start = int((self.field_size[1] / 2 - self.in_field_size[1] / 2) * self.r_to_s)
        x_out_end = int(x_out_start + self.in_field_size[0] * self.r_to_s)
        y_out_end = int(y_out_start + self.in_field_size[1] * self.r_to_s)
        goal_w = int(self.goal_size[0] * self.r_to_s)
        goal_h = int(self.goal_size[1] * self.r_to_s)
        x_goal1 = x_out_start - goal_w
        x_goal2 = x_out_end
        y_goal = int((self.field_size[1] / 2 - self.goal_size[1] / 2) * self.r_to_s)

        self._line(frame, x_out_start, y_out_start, x_out_end, y_out_start, white, 2)
        self._line(frame, x_out_end, y_out_start, x_out_end, y_out_end, white, 2)
        self._line(frame, x_out_end, y_out_end, x_out_start, y_out_end, white, 2)
        self._line(frame, x_out_start, y_out_end, x_out_start, y_out_start, white, 2)
        self._line(frame, x_goal1, y_goal, x_goal1 + goal_w, y_goal, white, 2)
        self._line(frame, x_goal1 + goal_w, y_goal, x_goal1 + goal_w, y_goal + goal_h, white, 2)
        self._line(frame, x_goal1 + goal_w, y_goal + goal_h, x_goal1, y_goal + goal_h, white, 2)
        self._line(frame, x_goal1, y_goal + goal_h, x_goal1, y_goal, white, 2)
        self._line(frame, x_goal2, y_goal, x_goal2 + goal_w, y_goal, white, 2)
        self._line(frame, x_goal2 + goal_w, y_goal, x_goal2 + goal_w, y_goal + goal_h, white, 2)
        self._line(frame, x_goal2 + goal_w, y_goal + goal_h, x_goal2, y_goal + goal_h, white, 2)
        self._line(frame, x_goal2, y_goal + goal_h, x_goal2, y_goal, white, 2)

        center_x = int((self.field_size[0] / 2) * self.r_to_s)
        self._line(frame, center_x, y_out_start, center_x, y_out_end, white, 2)

        positions = state["positions"]
        rotations = state["rotations"]
        n = positions.shape[0] // 2
        off_x = self.field_size[0] / 2
        off_y = self.field_size[1] / 2

        for i in range(positions.shape[0]):
            col = blue if i < n else red
            x = int((positions[i, 0] + off_x) * self.r_to_s)
            y = int((positions[i, 1] + off_y) * self.r_to_s)
            self._circle(frame, x, y, int(1.0 * self.r_to_s), col)
            rot = rotations[i]
            dx = int(math.cos(rot) * 1.0 * self.r_to_s)
            dy = int(math.sin(rot) * 1.0 * self.r_to_s)
            self._line(frame, x, y, x + dx, y + dy, yellow, 2)

        bx, by, _, _ = state["ball"]
        bx = int((bx + off_x) * self.r_to_s)
        by = int((by + off_y) * self.r_to_s)
        self._circle(frame, bx, by, int(1.0 * self.r_to_s), white)
        return frame

    def render(self, state: dict[str, Any]) -> np.ndarray | None:
        if self.render_mode == "human":
            self._ensure()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

            field = self._draw(state)
            self.screen.blit(field, [0, 0])
            pygame.display.flip()
            if self.wait_period > 0:
                time.sleep(self.wait_period)
            return None

        return self._draw_rgb_array(state)
