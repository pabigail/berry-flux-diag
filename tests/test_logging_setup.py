"""Tests for configure_logging.

The package logs rather than prints, so progress messages are invisible
until a caller asks for them. These pin the two properties that make that
safe: a library must not configure the root logger, and turning logging on
twice must not double every message.
"""

from __future__ import annotations

import io
import logging

import berry_flux_diag as bfd

PACKAGE_LOGGER = "berry_flux_diag"


def restore(logger):
    """Undo what configure_logging did to the package logger."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def test_messages_reach_the_stream():
    stream = io.StringIO()
    logger = bfd.configure_logging("INFO", stream=stream)
    try:
        logging.getLogger("berry_flux_diag.Overlaps").info("hello %s", 42)
        assert "hello 42" in stream.getvalue()
    finally:
        restore(logger)


def test_info_is_silent_by_default():
    """Without a call, progress messages stay out of the way."""
    stream = io.StringIO()
    logger = bfd.configure_logging("WARNING", stream=stream)
    try:
        logging.getLogger("berry_flux_diag.Overlaps").info("progress")
        logging.getLogger("berry_flux_diag.Overlaps").warning("trouble")
        out = stream.getvalue()
        assert "progress" not in out
        assert "trouble" in out
    finally:
        restore(logger)


def test_calling_twice_does_not_duplicate_messages():
    """The regression case for the usual addHandler-in-a-loop mistake."""
    stream = io.StringIO()
    logger = bfd.configure_logging("INFO", stream=stream)
    try:
        bfd.configure_logging("INFO", stream=stream)
        assert len(logger.handlers) == 1
        logging.getLogger("berry_flux_diag.Overlaps").info("once")
        assert stream.getvalue().count("once") == 1
    finally:
        restore(logger)


def test_root_logger_is_left_alone():
    """A library must not hijack the logging of the program importing it."""
    root = logging.getLogger()
    before_handlers = list(root.handlers)
    before_level = root.level

    logger = bfd.configure_logging("DEBUG", stream=io.StringIO())
    try:
        assert list(root.handlers) == before_handlers
        assert root.level == before_level
    finally:
        restore(logger)


def test_configures_the_package_logger_not_a_child():
    logger = bfd.configure_logging("INFO", stream=io.StringIO())
    try:
        assert logger.name == PACKAGE_LOGGER
    finally:
        restore(logger)
