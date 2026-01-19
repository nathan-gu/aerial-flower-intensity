"""
Command line interface for the Aerial Flower Intensity module.
"""
import argparse
import logging
import sys
import pathlib
import os
import pdb
import traceback
from contextlib import contextmanager

import aerialflowerintensity
import aerialflowerintensity.flowering_intensity
from . import cliutils

_LOGGER = logging.getLogger(__name__)

os.environ["MPLBACKEND"] = "Agg"

# Boolean state mapping for CLI arguments
BOOLEAN_STATES = {
    "0": False, "1": True,
    "false": False, "true": True,
    "no": False, "yes": True,
    "off": False, "on": True
}

def sanitize_path(path):
    """
    Sanitize the given path, from the command line interface, and return it normalized and absolute.

    :param str path: Path to sanitize.
    :return: Path sanitized, normalized and as absolute.
    :rtype: pathlib.Path
    """
    return pathlib.Path(os.path.abspath(path.replace(""", "").replace(""", "")))

def boolean_state(value):
    """
    Convert the given boolean state as a boolean value.

    :param str value: Boolean state to convert.
    :return: Boolean value.
    :rtype: bool
    """
    try:
        return BOOLEAN_STATES[value.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"invalid boolean state value: {value}") from None

def add_boolean_flag(parser, name, help_str):
    """
    Helper to add a boolean state flag to the argument parser.

    :param argparse.ArgumentParser parser: Parser to update.
    :param str name: Name of the flag without leading hyphen.
    :param str help_str: Flag help description.
    """
    parser.add_argument(
        f"--{name}",
        dest=name.replace("-", "_"),
        type=boolean_state,
        default=False,
        const=True,
        nargs="?",
        metavar="BOOL_STATE",
        help=help_str
    )

@contextmanager
def debugger(enable=False):
    """
    Enter post-mortem debugging if an exception is raised.

    :param bool enable: If set, an uncaught exception will trigger post-mortem debugging.
    """
    try:
        yield
    except Exception as exc:
        if enable:
            traceback.print_exception(None, exc, exc.__traceback__.tb_next)
            pdb.post_mortem()
        raise


def create_parser():
    """
    Create the parser for the aerialflowerintensity command.

    :return: Configured parser.
    :rtype: argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "las_folder_path",
        type=sanitize_path,
        help="The path to the folder containing LAS plot files."
        )

    parser.add_argument(
        "hsl_filters",
        type=cliutils.get_hsl_filter,
        help="""
            HSL Color filters species aliases (Apple, Prunus) or
            HSl Color filters as a list to identify flowers and canopy. Following format:
            [[[H min bound, S min bound, L min bound],[H max bound, S max bound, L max bound]],
            [[Filter 2]] .... ]"""
        )

    parser.add_argument(
        "output_folder",
        type=sanitize_path,
        help="Path to output folder."
        )

    parser.add_argument(
        "--voxel_size",
        dest="voxel_size",
        type=float,
        nargs=3,
        default=[0.1, 0.1, 0.1],
        metavar=("x size", "y size", "z size"),
        help="Voxel size for partitioning the point cloud data.")

    parser.add_argument(
        "--dynamic_white_filter",
        dest="dynamic_white_filter",
        type=bool,
        default=False,
        help="Assess lightness of white HSL color filter using measured plots data lightness std")

    # parser.add_argument(
    #     "--normalization",
    #     dest="normalization",
    #     type=float,
    #     nargs=2,
    #     default=None,
    #     metavar=("mean", "std"),
    #     help="Luminosity statistics over multiple acquisiton for color normalization")

    parser.add_argument(
        "--ground_threshold",
        dest="ground_threshold",
        type=float,
        default=0.5,
        help="The height relative to the ground plane to filter out ground points.")

    parser.add_argument(
        "--nb_thread",
        dest="nb_thread",
        type=int,
        default=None,
        help="Number of threads to run the process on")

    parser.add_argument(
		"-v", "--version",
		action="version",
		version=f"%(prog)s v{aerialflowerintensity.__version__}"
		)

    # Optional arguments
    add_boolean_flag(parser, "debug", "Enable debug outputs. Imply --verbose.")
    add_boolean_flag(parser, "pdb", "Enable post-mortem debugging with pdb.")
    add_boolean_flag(parser, "verbose", "Enable debug logging.")

    return parser

def main(args=None):
    """
    Run the main procedure.

    :param list args: List of arguments for the command line interface. If not set, arguments are
        taken from ``sys.argv``.
    """
    parser = create_parser()
    args = parser.parse_args(args)
    args.verbose = args.verbose or args.debug

    with debugger(enable=args.pdb):
        # Ensure the directory exists to create the log file
        args.output_folder.mkdir(parents=True, exist_ok=True)
        log_filename = args.output_folder.joinpath("aerial_flower_intensity.log")

        cliutils.setup_logging(debug=args.verbose, filename=log_filename)
        _LOGGER.debug("command: %s", " ".join(sys.argv))
        _LOGGER.debug("version: %s", aerialflowerintensity.__version__)

        # Call the main function of the module
        _LOGGER.info("Got some arguments: %s", args)
        _LOGGER.debug("This log is very useful for debugging!")

        flowering_output_path = args.output_folder/"results_flowering_intensity.csv"
        visualization_output_path = args.output_folder/"visualization"

        aerialflowerintensity.flowering_intensity.measure_site_flowering_intensity(
            args.las_folder_path,
            nb_thread=args.nb_thread,
            hsl_filters=args.hsl_filters,
            voxel_size=args.voxel_size,
            ground_threshold=args.ground_threshold,
            dynamic_white_filter=args.dynamic_white_filter,
            visualization_output_path=visualization_output_path
            ).to_csv(flowering_output_path, index=False, sep=";")

if __name__ == "__main__":
    main()
