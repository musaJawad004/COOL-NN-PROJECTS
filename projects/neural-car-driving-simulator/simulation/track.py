"""Road geometry, checkpoints, and sensor ray casting."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


Point = tuple[float, float]


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def distance_to_segment(point: Point, start: Point, end: Point) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return distance(point, start)
    projection = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared))
    nearest = (start[0] + projection * dx, start[1] + projection * dy)
    return distance(point, nearest)


@dataclass
class Track:
    points: list[Point]
    road_width: float = 76.0
    _road_masks: dict[float, set[tuple[int, int]]] = field(default_factory=dict, init=False, repr=False)
    _mask_resolution: int = field(default=4, init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.points) < 6:
            raise ValueError("A track needs at least six center points.")
        self._build_road_mask(0.0)
        self._build_road_mask(5.0)

    @property
    def segments(self):
        for index, start in enumerate(self.points):
            yield start, self.points[(index + 1) % len(self.points)]

    def distance_from_road(self, point: Point) -> float:
        return min(distance_to_segment(point, start, end) for start, end in self.segments)

    def on_road(self, point: Point, margin: float = 0.0) -> bool:
        normalized_margin = round(margin, 1)
        if normalized_margin not in self._road_masks:
            self._build_road_mask(normalized_margin)
        key = (
            round(point[0] / self._mask_resolution),
            round(point[1] / self._mask_resolution),
        )
        return key in self._road_masks[normalized_margin]

    def _build_road_mask(self, margin: float) -> None:
        resolution = self._mask_resolution
        padding = self.road_width / 2 + resolution * 2
        min_x = math.floor((min(point[0] for point in self.points) - padding) / resolution)
        max_x = math.ceil((max(point[0] for point in self.points) + padding) / resolution)
        min_y = math.floor((min(point[1] for point in self.points) - padding) / resolution)
        max_y = math.ceil((max(point[1] for point in self.points) + padding) / resolution)
        threshold = self.road_width / 2 - margin
        mask: set[tuple[int, int]] = set()
        for grid_y in range(min_y, max_y + 1):
            for grid_x in range(min_x, max_x + 1):
                point = (grid_x * resolution, grid_y * resolution)
                if any(
                    distance_to_segment(point, start, end) <= threshold
                    for start, end in self.segments
                ):
                    mask.add((grid_x, grid_y))
        self._road_masks[margin] = mask

    def ray_distance(self, origin: Point, angle: float, maximum: float = 145.0) -> float:
        step = 4.0
        ray = step
        while ray <= maximum:
            point = (origin[0] + math.cos(angle) * ray, origin[1] + math.sin(angle) * ray)
            if not self.on_road(point):
                return ray
            ray += step
        return maximum

    @property
    def starting_pose(self) -> tuple[Point, float]:
        start, following = self.points[0], self.points[1]
        return start, math.atan2(following[1] - start[1], following[0] - start[0])


def preset_track(number: int) -> Track:
    # Dense parametric center lines make the rendered road and collision model
    # agree, even through technical bends.
    points: list[Point] = []
    count = 44
    for index in range(count):
        angle = index / count * math.tau
        if number == 2:
            radius_x = 268 + 45 * math.sin(3 * angle)
            radius_y = 202 + 28 * math.sin(2 * angle + 0.6)
        elif number == 3:
            radius_x = 255 + 58 * math.cos(2 * angle + 0.4)
            radius_y = 195 + 42 * math.sin(3 * angle)
        else:
            radius_x, radius_y = 275, 205
        points.append((380 + radius_x * math.cos(angle), 310 + radius_y * math.sin(angle)))
    return Track(points, road_width=84.0)
