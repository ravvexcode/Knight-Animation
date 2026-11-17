
import sys
import math
from pathlib import Path

import cv2
import numpy as np
import pygame


ROOT = Path(__file__).resolve().parent
IMAGE = ROOT / "assets" / "knight-exact.jpg"
MUSIC = ROOT / "assets" / "music.mp3"

FPS = 60
START_SIZE = (900, 760)

# Lower processing resolution = better performance
QUALITY_SCALE = 0.85


def make_layers(rgb):
    h, w = rgb.shape[:2]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    # Cape area: only the real red fabric
    polygon = np.array([
        [0.53*w, 0.43*h],
        [0.70*w, 0.43*h],
        [0.99*w, 0.34*h],
        [1.00*w, 0.68*h],
        [0.85*w, 0.70*h],
        [0.68*w, 0.65*h],
        [0.57*w, 0.59*h]
    ], dtype=np.int32)

    roi = np.zeros((h, w), np.uint8)
    cv2.fillPoly(roi, [polygon], 255)

    red = (
        ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170))
        & (hsv[:, :, 1] > 75)
        & (hsv[:, :, 2] > 35)
    )

    cape_mask = (red & (roi > 0)).astype(np.uint8) * 255

    cape_mask = cv2.morphologyEx(
        cape_mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8)
    )

    cape_mask = cv2.dilate(
        cape_mask,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    # Remove original cape from background
    clean = cv2.inpaint(
        rgb,
        cape_mask,
        7,
        cv2.INPAINT_TELEA
    )

    cape_alpha = cv2.GaussianBlur(
        cape_mask,
        (0, 0),
        1.1
    ).astype(np.float32) / 255.0

    cape_rgba = np.dstack([
        rgb,
        (cape_alpha * 255).astype(np.uint8)
    ])

    # Grass foreground
    grass_mask = np.zeros((h, w), np.uint8)

    left = np.array([
        [0, .77*h],
        [.19*w, .77*h],
        [.28*w, .84*h],
        [.43*w, .91*h],
        [.50*w, h],
        [0, h]
    ], np.int32)

    right = np.array([
        [.72*w, .84*h],
        [.84*w, .77*h],
        [w, .78*h],
        [w, h],
        [.60*w, h]
    ], np.int32)

    cv2.fillPoly(grass_mask, [left, right], 255)

    grass_mask = cv2.GaussianBlur(
        grass_mask,
        (0, 0),
        2.0
    ).astype(np.float32) / 255.0

    return clean, cape_rgba, grass_mask


class KnightAnimation:

    def __init__(self, original, size):

        self.original = original
        self.time = 0.0

        self.prepare(size)

    def prepare(self, size):

        dw, dh = size

        # Process at reduced resolution
        self.w = max(2, int(dw * QUALITY_SCALE))
        self.h = max(2, int(dh * QUALITY_SCALE))

        self.output_size = size

        resized = cv2.resize(
            self.original,
            (self.w, self.h),
            interpolation=cv2.INTER_AREA
        )

        self.clean, self.cape, self.grass = make_layers(resized)

        # Precompute coordinate grids ONCE
        self.yy, self.xx = np.mgrid[
            0:self.h,
            0:self.w
        ].astype(np.float32)

        # Precompute cape movement factors
        self.progress = np.clip(
            (self.xx / self.w - 0.53) / 0.47,
            0,
            1
        )

        self.region = np.clip(
            (self.yy / self.h - 0.39) / 0.34,
            0,
            1
        )

        # Precompute grass movement factors
        self.vertical = np.clip(
            (self.yy / self.h - 0.76) / 0.24,
            0,
            1
        )

        self.grass_phase = (
            self.xx * 0.018 +
            self.yy * 0.009
        )

        # Cache cape channels
        self.cape_rgb = self.cape[:, :, :3].astype(np.float32)

        self.cape_alpha = (
            self.cape[:, :, 3:4].astype(np.float32) / 255.0
        )

        self.grass_alpha = self.grass[:, :, None]

        print(
            f"Animation prepared: {self.w}x{self.h}"
        )

    def animate_cape(self, t):

        p = self.progress
        r = self.region

        phase = (
            t * 2.0
            - p * 7.0
            + r * 1.2
        )

        dx = (
            np.sin(phase) * 2.0
            + np.sin(phase * 0.53) * 1.3
        ) * p

        dy = (
            np.sin(phase + 0.8) * 5.0
            + np.sin(phase * 0.6) * 2.0
        ) * p

        map_x = self.xx - dx.astype(np.float32)
        map_y = self.yy - dy.astype(np.float32)

        return cv2.remap(
            self.cape,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

    def animate_grass(self, t):

        phase = (
            t * 1.55
            + self.grass_phase
        )

        dx = np.sin(phase) * 3.2 * self.vertical

        dy = (
            np.sin(phase * 0.72)
            * 1.0
            * self.vertical
        )

        map_x = self.xx - dx.astype(np.float32)
        map_y = self.yy - dy.astype(np.float32)

        warped = cv2.remap(
            self.clean,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        alpha = self.grass_alpha

        return np.clip(
            warped.astype(np.float32) * alpha
            + self.clean.astype(np.float32) * (1 - alpha),
            0,
            255
        ).astype(np.uint8)

    def render(self, dt):

        self.time += dt
        t = self.time

        # Animate grass
        frame = self.animate_grass(t)

        # Animate actual cape
        cape = self.animate_cape(t)

        alpha = self.cape_alpha

        # Composite cape over background
        frame = np.clip(
            cape[:, :, :3].astype(np.float32) * alpha
            + frame.astype(np.float32) * (1 - alpha),
            0,
            255
        ).astype(np.uint8)

        # Upscale only once for display
        if frame.shape[1] != self.output_size[0]:
            frame = cv2.resize(
                frame,
                self.output_size,
                interpolation=cv2.INTER_LINEAR
            )

        return frame


def main():

    if not IMAGE.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE}"
        )

    cv2.setNumThreads(2)

    bgr = cv2.imread(
        str(IMAGE),
        cv2.IMREAD_COLOR
    )

    if bgr is None:
        raise RuntimeError("Cannot load knight image.")

    original = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB
    )

    pygame.init()

    try:
        pygame.mixer.init()

        if MUSIC.exists():
            pygame.mixer.music.load(str(MUSIC))
            pygame.mixer.music.set_volume(0.75)
            pygame.mixer.music.play(-1)
        else:
            print("Music not found. Continuing without music.")

    except pygame.error as exc:
        print("Audio error:", exc)

    screen = pygame.display.set_mode(
        START_SIZE,
        pygame.RESIZABLE
    )

    pygame.display.set_caption("Knight Experience")

    clock = pygame.time.Clock()

    animation = None

    paused = False
    fullscreen = False
    running = True

    last_size = None

    while running:

        dt = min(clock.tick(FPS) / 1000.0, 0.04)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key in (
                    pygame.K_ESCAPE,
                    pygame.K_q
                ):
                    running = False

                elif event.key == pygame.K_SPACE:

                    paused = not paused

                    if pygame.mixer.get_init():

                        if paused:
                            pygame.mixer.music.pause()
                        else:
                            pygame.mixer.music.unpause()

                elif event.key == pygame.K_F11:

                    fullscreen = not fullscreen

                    if fullscreen:

                        screen = pygame.display.set_mode(
                            (0, 0),
                            pygame.FULLSCREEN
                        )

                    else:

                        screen = pygame.display.set_mode(
                            START_SIZE,
                            pygame.RESIZABLE
                        )

        sw, sh = screen.get_size()

        margin = 18

        scale = min(
            (sw - margin * 2) / original.shape[1],
            (sh - margin * 2) / original.shape[0]
        )

        dw = max(2, int(original.shape[1] * scale))
        dh = max(2, int(original.shape[0] * scale))

        ox = (sw - dw) // 2
        oy = (sh - dh) // 2

        # Rebuild layers only when window dimensions change
        if last_size != (dw, dh):

            animation = KnightAnimation(
                original,
                (dw, dh)
            )

            last_size = (dw, dh)

        screen.fill((0, 0, 0))

        if paused:
            dt = 0

        frame = animation.render(dt)

        surface = pygame.image.frombuffer(
            frame.tobytes(),
            (dw, dh),
            "RGB"
        )

        screen.blit(surface, (ox, oy))

        pygame.display.flip()

    if pygame.mixer.get_init():
        pygame.mixer.music.stop()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
