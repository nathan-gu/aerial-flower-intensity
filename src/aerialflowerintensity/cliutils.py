"""
Utility classes and functions for the command line interface configuration.
"""
import argparse
import ast
import logging
import logging.config
import sys
import tqdm

def parse_nested_list(input_string):
    """
    Parses a string representation of a nested list.

    :param str input_string: The input string to be parsed.
    :return: The parsed nested list.
    :rtype: list
    :raises argparse.ArgumentTypeError: If the input string is not a valid list format.
    """

    try:
        # Convert string representation of list to an actual list
        return ast.literal_eval(input_string)
    except (ValueError, SyntaxError) as exc:
        raise argparse.ArgumentTypeError(f"Invalid list format: {input_string}") from exc

def get_hsl_filter(input_string):
    """
    Retrieves an HSL filter definition based on input string or predefined aliases.

    :param str input_str: The input string representing the HSL filter or a predefined alias.

    :return: The HSL filter definition as a nested list of ranges.
    :rtype: list
    """
    # Define predefined aliases
    aliases = {
        "Apple": [[[0.77, 0.20, 0.25], [0.88, 1, 1]], [[0, 0.8, 0], [1, 1, 1]]],
        "Prunus": [[[0.77,0.20,0.25],[1,0.85,1]]],
        "Prunus_brighter": [[[0.77,0.40,0.25],[1,0.85,1]]],
        "White": [[[0, 0.85, 0], [1, 1, 1]]]
    }

    # Check if input_str matches any alias
    if input_string in aliases:
        return aliases[input_string]

    # Try to parse input_str as a list
    try:
        return parse_nested_list(input_string)
    except (argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError(f"Invalid alias or list format: {input_string}") from exc

def setup_logging(debug=False, filename=None):
    """
    Configure logging for the command line interface.

    :param bool debug: If set, debug logs will also be printed to the terminal.
    :param str filename: Path to the log file to create or extend. If not specified, logging
        to file is deactivated.
    """
    settings = {
        "version": 1,
        "disable_existing_loggers": False,
        "loggers": {
            "aerialflowerintensity": {
                "level": logging.DEBUG,
                "handlers": ["console", "console_error"]
                }
            },
        "handlers": {
            "console": {
                "class": "aerialflowerintensity.cliutils.TqdmStreamHandler",
                "level": logging.DEBUG if debug else logging.INFO,
                "formatter": "brief",
                "filters": ["skip_errors"],

                },            "console_error": {
                "class": "aerialflowerintensity.cliutils.TqdmStreamHandler",
                "level": logging.ERROR,
                "formatter": "brief",
                }
            },
        "formatters": {
            "brief": {
                "format": "[%(levelname)s] %(message)s"
                },
            "default": {
                "format": "%(asctime)s:%(name)s:%(levelname)s %(message)s",
                "datefmt" : "%Y/%m/%d %H:%M:%S"
                }
            },
        "filters": {
            "skip_errors": {
                "()": "aerialflowerintensity.cliutils.LevelFilter",
                "level": logging.ERROR
                }
            }
        }

    # Add file logging if a filename is provided
    if filename is not None:
        settings["root"] = {
            "level": logging.DEBUG,
            "handlers": ["file"]
            }
        settings["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "level": logging.DEBUG,
            "formatter": "default",
            "filename": filename
            }

    # Silence unwanted debug outputs
    settings["loggers"].update(
        matplotlib={"level": logging.WARNING},
        PIL={"level": logging.WARNING}
    )

    logging.config.dictConfig(settings)
    logging.captureWarnings(True)

    # Add a hook to log uncaught exceptions
    def _log_exception_hook(exc_type, exc_value, exc_tb):
        logging.getLogger(__package__).critical(
            "uncaught exception", exc_info=(exc_type, exc_value, exc_tb)
            )
    sys.excepthook = _log_exception_hook

class TqdmStreamHandler(logging.StreamHandler):
    """
    A logging handler that uses tqdm to write log messages, ensuring compatibility
    with tqdm progress bars.
    """
    def emit(self, record):
        """
        Emit a record.

        This method overrides the default emit method to write log messages
        using tqdm, ensuring they do not interfere with tqdm progress bars.

        :param logging.LogRecord record: The log record to be emitted.
        """
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
        except (IOError, OSError, AttributeError, ValueError, TypeError):
            self.handleError(record)

class LevelFilter:
    """Logging filter used to skip records with the given log level and above."""

    def __init__(self, level):
        """
        Initialize the filter with a minimum log level to skip.

        :param int level: Logging level to skip, levels that are above are also skipped.
        """
        self.level = level

    def filter(self, record):
        """
        Skip the record if its level is equal or above the configured level.

        :param record: Logging record.
        :type record: :class:`logging.LogRecord`
        :return: If this record should be logged.
        :rtype: bool
        """
        return record.levelno < self.level
