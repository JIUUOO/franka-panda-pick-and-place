from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import carb

from isaaclab.devices import DeviceBase, Se3Gamepad, Se3GamepadCfg


class FrankaPickPlaceGamepad(Se3Gamepad):
    """Project-specific gamepad mapping for Franka tabletop pick-and-place."""

    def __init__(self, cfg: Se3GamepadCfg):
        self._reset_callback = None
        self._reset_latched = False
        self._lb_pressed = False
        self._rb_pressed = False
        super().__init__(cfg)
        self._lb_button = self._get_first_gamepad_input(("LEFT_SHOULDER", "LEFT_BUMPER", "LB", "L1"))
        self._rb_button = self._get_first_gamepad_input(("RIGHT_SHOULDER", "RIGHT_BUMPER", "RB", "R1"))

    def add_callback(self, key, func: Callable):
        if isinstance(key, str):
            if key in ("R", "RESET"):
                self._reset_callback = func
            return
        super().add_callback(key, func)

    def reset(self):
        super().reset()
        self._reset_latched = False
        self._lb_pressed = False
        self._rb_pressed = False

    def _create_key_bindings(self):
        super()._create_key_bindings()
        self._INPUT_STICK_VALUE_MAPPING[carb.input.GamepadInput.LEFT_STICK_RIGHT] = (1, 1, self.pos_sensitivity)
        self._INPUT_STICK_VALUE_MAPPING[carb.input.GamepadInput.LEFT_STICK_LEFT] = (0, 1, self.pos_sensitivity)

    def _on_gamepad_event(self, event, *args, **kwargs):
        result = super()._on_gamepad_event(event, *args, **kwargs)
        if self._reset_callback is None:
            return result

        if self._lb_button is not None and event.input == self._lb_button:
            self._lb_pressed = event.value > 0.5
        if self._rb_button is not None and event.input == self._rb_button:
            self._rb_pressed = event.value > 0.5

        reset_pressed = self._lb_pressed and self._rb_pressed
        if reset_pressed and not self._reset_latched:
            self._reset_callback()
        self._reset_latched = reset_pressed
        return result

    @staticmethod
    def _get_first_gamepad_input(names: tuple[str, ...]):
        for name in names:
            if hasattr(carb.input.GamepadInput, name):
                return getattr(carb.input.GamepadInput, name)
        return None

    def __str__(self) -> str:
        msg = super().__str__()
        msg += "\tReset environment: LB + RB\n"
        msg += "\tProject mapping: Left stick y-axis is inverted\n"
        return msg


@dataclass
class FrankaPickPlaceGamepadCfg(Se3GamepadCfg):
    class_type: type[DeviceBase] = FrankaPickPlaceGamepad
