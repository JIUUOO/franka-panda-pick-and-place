from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import carb

from isaaclab.devices import DeviceBase, Se3Gamepad, Se3GamepadCfg


class FrankaPickPlaceGamepad(Se3Gamepad):
    """Project-specific gamepad mapping for Franka tabletop pick-and-place."""

    def __init__(self, cfg: Se3GamepadCfg):
        self._reset_callback = None
        self._start_callback = None
        self._reset_latched = False
        self._start_latched = False
        self._last_logged_button = None
        self._lb_pressed = False
        self._rb_pressed = False
        super().__init__(cfg)
        self._start_button_names = ("A", "BUTTON_A", "GAMEPAD_A", "FACE_BUTTON_DOWN", "SOUTH", "CROSS")
        self._start_button = self._get_first_gamepad_input(self._start_button_names)
        self._lb_button = self._get_first_gamepad_input(("LEFT_SHOULDER", "LEFT_BUMPER", "LB", "L1"))
        self._rb_button = self._get_first_gamepad_input(("RIGHT_SHOULDER", "RIGHT_BUMPER", "RB", "R1"))

    def add_callback(self, key, func: Callable):
        if isinstance(key, str):
            if key in ("START", "SPACE"):
                self._start_callback = func
            if key in ("R", "RESET"):
                self._reset_callback = func
            return
        super().add_callback(key, func)

    def reset(self):
        super().reset()
        self._reset_latched = False
        self._start_latched = False
        self._last_logged_button = None
        self._lb_pressed = False
        self._rb_pressed = False

    def _create_key_bindings(self):
        self._INPUT_STICK_VALUE_MAPPING = {
            # XY translation on the right stick.
            carb.input.GamepadInput.RIGHT_STICK_UP: (0, 0, self.pos_sensitivity),
            carb.input.GamepadInput.RIGHT_STICK_DOWN: (1, 0, self.pos_sensitivity),
            carb.input.GamepadInput.RIGHT_STICK_LEFT: (0, 1, self.pos_sensitivity),
            carb.input.GamepadInput.RIGHT_STICK_RIGHT: (1, 1, self.pos_sensitivity),
            # Orientation trim on the left stick.
            carb.input.GamepadInput.LEFT_STICK_UP: (0, 4, self.rot_sensitivity * 0.8),
            carb.input.GamepadInput.LEFT_STICK_DOWN: (1, 4, self.rot_sensitivity * 0.8),
            carb.input.GamepadInput.LEFT_STICK_LEFT: (1, 5, self.rot_sensitivity * 0.8),
            carb.input.GamepadInput.LEFT_STICK_RIGHT: (0, 5, self.rot_sensitivity * 0.8),
        }

        self._INPUT_DPAD_VALUE_MAPPING = {
            # Z translation on the vertical d-pad.
            carb.input.GamepadInput.DPAD_UP: (0, 2, self.pos_sensitivity),
            carb.input.GamepadInput.DPAD_DOWN: (1, 2, self.pos_sensitivity),
            # Roll trim on the horizontal d-pad.
            carb.input.GamepadInput.DPAD_LEFT: (0, 3, self.rot_sensitivity * 0.8),
            carb.input.GamepadInput.DPAD_RIGHT: (1, 3, self.rot_sensitivity * 0.8),
        }

    def _on_gamepad_event(self, event, *args, **kwargs):
        result = super()._on_gamepad_event(event, *args, **kwargs)
        if event.value > 0.5:
            self._log_button_event_once(event.input)

        if self._start_callback is not None and self._matches_input(event.input, self._start_button_names):
            start_pressed = event.value > 0.5
            if start_pressed and not self._start_latched:
                self._start_callback()
            self._start_latched = start_pressed

        if self._lb_button is not None and event.input == self._lb_button:
            self._lb_pressed = event.value > 0.5
        if self._rb_button is not None and event.input == self._rb_button:
            self._rb_pressed = event.value > 0.5

        reset_pressed = self._lb_pressed and self._rb_pressed
        if self._reset_callback is not None and reset_pressed and not self._reset_latched:
            self._reset_callback()
        self._reset_latched = reset_pressed
        return result

    @staticmethod
    def _get_first_gamepad_input(names: tuple[str, ...]):
        for name in names:
            if hasattr(carb.input.GamepadInput, name):
                return getattr(carb.input.GamepadInput, name)
        return None

    @staticmethod
    def _input_name(gamepad_input) -> str:
        name = getattr(gamepad_input, "name", None)
        if isinstance(name, str):
            return name
        text = str(gamepad_input)
        return text.rsplit(".", 1)[-1]

    def _matches_input(self, gamepad_input, names: tuple[str, ...]) -> bool:
        enum_value = self._get_first_gamepad_input(names)
        if enum_value is not None and gamepad_input == enum_value:
            return True
        return self._input_name(gamepad_input) in names

    def _log_button_event_once(self, gamepad_input):
        input_name = self._input_name(gamepad_input)
        if input_name == self._last_logged_button:
            return
        stick_or_dpad = ("STICK", "DPAD")
        if any(token in input_name for token in stick_or_dpad):
            return
        self._last_logged_button = input_name
        print(f"[GAMEPAD] pressed input={input_name}")

    def __str__(self) -> str:
        msg = super().__str__()
        msg += "\tStart recording: A\n"
        msg += "\tReset environment: LB + RB\n"
        msg += "\tProject mapping: Right stick controls X/Y translation\n"
        msg += "\tProject mapping: D-Pad Up/Down controls Z translation\n"
        msg += "\tProject mapping: Left stick controls pitch/yaw trim\n"
        msg += "\tProject mapping: D-Pad Left/Right controls roll trim\n"
        return msg


@dataclass
class FrankaPickPlaceGamepadCfg(Se3GamepadCfg):
    class_type: type[DeviceBase] = FrankaPickPlaceGamepad
