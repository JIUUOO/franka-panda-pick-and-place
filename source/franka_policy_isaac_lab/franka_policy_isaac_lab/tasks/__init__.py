"""Gym registrations for project tasks."""

from isaaclab_tasks.utils import import_packages


_BLACKLIST_PKGS = [".mdp"]
import_packages(__name__, _BLACKLIST_PKGS)
