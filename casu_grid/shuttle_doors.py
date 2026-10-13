"""``ShuttleStartOpen``: the scripted spawn-shuttle doors of the canonical start.

Source behaviour (``SearchBase/ShuttleStartOpen.cs``): the first time a ``Body``
collider enters one of the two 1x1 trigger volumes, ``activated`` latches. From
then on ``progress`` grows with game time and the doors rise by
``Lerp(0, 6, (progress - 2) * 0.75)`` world units, so they start moving two
seconds after activation and finish 4/3 s later. The script is destroyed at
``progress > 10`` leaving the doors raised.

CasU advances ``progress`` with the render-frame ``Time.deltaTime``; here it advances
by the fixed 20 ms tick (the same game time), which is deterministic. Measured in
the CB1 harness the doors began moving ~2.3 s after first trigger contact and were
fully raised by ~3.6 s, with the physics collider following in coarse frame-sized
steps; the tick-exact rise is therefore an idealization of that.
"""
from __future__ import annotations

from dataclasses import dataclass

DOOR_LIFT = 6.0
OPEN_DELAY_S = 2.0
OPEN_RATE = 0.75
DESTROY_AFTER_S = 10.0


@dataclass
class ShuttleDoors:
    trigger_boxes: tuple          # ((x0, y0, x1, y1), ...) world-space trigger volumes
    base_centers: tuple           # ((x, y), ...) closed door centres
    activated: bool = False
    progress: float = 0.0
    destroyed: bool = False

    def offset(self) -> float:
        t = (self.progress - OPEN_DELAY_S) * OPEN_RATE
        return DOOR_LIFT * min(max(t, 0.0), 1.0)

    def tick(self, dt: float, body_box) -> bool:
        """Advance one fixed tick. ``body_box`` = (x0, y0, x1, y1) of the player collider.
        Returns True when the door positions changed."""
        before = self.offset()
        if not self.activated:
            bx0, by0, bx1, by1 = body_box
            for x0, y0, x1, y1 in self.trigger_boxes:
                if bx0 < x1 and bx1 > x0 and by0 < y1 and by1 > y0:
                    self.activated = True
                    break
        if self.activated and not self.destroyed:
            self.progress += dt
            if self.progress > DESTROY_AFTER_S:
                self.destroyed = True
        return self.offset() != before

    def centers(self):
        lift = self.offset()
        return [(x, y + lift) for x, y in self.base_centers]

    def snapshot(self) -> dict:
        return {"activated": self.activated, "progress": self.progress, "destroyed": self.destroyed}

    def restore(self, snap: dict) -> None:
        self.activated, self.progress, self.destroyed = snap["activated"], snap["progress"], snap["destroyed"]
