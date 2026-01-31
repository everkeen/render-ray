"""Render Ray Main Module"""

import logging
import sys
import argparse
import typing
import render.const
import render.obj_parse  # For unit tests


def init_logging(log_to_file: bool, debug: bool) -> None:
    """Initialize logging configuration."""
    if not render.const.LOGS_DIR.exists():
        render.const.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_level = logging.DEBUG if debug else logging.INFO
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_to_file:
        handlers.append(
            logging.FileHandler(
                filename=str(render.const.LOGS_DIR / "render_ray.log"),
                mode="a",
            )
        )

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
    )


def _obj_parse_unit_test() -> bool:
    """Unit test for OBJ parser."""
    parser = render.obj_parse.OBJParser()
    try:
        parser.load_file(str(render.const.MESHES_DIR / "cube.obj"))
    except FileNotFoundError:
        print("OBJ file for unit test not found.")
        return False
    obj_data = parser.parse()
    print(f"Parsed {len(obj_data.vertices)} vertices.")
    print(f"Parsed {len(obj_data.faces)} faces.")
    print(f"Parsed {len(obj_data.face_uvs)} face UVs.")
    print(f"Parsed {len(obj_data.face_normals)} face normals.")
    print(f"Parsed {len(obj_data.normals)} normals.")
    print(f"Smooth lighting: {obj_data.smooth_lighting}")
    print(f"Parsed {len(obj_data.uv)} texture coordinates.")
    print(
        f"Objects found: {', '.join(obj_data.obj_names)} ({len(obj_data.obj_names)} total)"
    )
    return True


UNIT_TESTS: dict[str, typing.Callable[[], bool]] = {
    "obj_parse": _obj_parse_unit_test,
}


def unit_tests() -> None:
    """Unit tests for the Render Ray modules."""
    print("Debug mode unit tester.")
    print("Available tests:")
    for test_name in UNIT_TESTS:
        print(f" - {test_name}")
    selected_test = input("Enter the test name to run (or 'all' to run all tests): ")
    if selected_test == "all":
        for test_name, test_func in UNIT_TESTS.items():
            print(f"Running test: {test_name}")
            success = test_func()
            print(f"Test {test_name} {'passed' if success else 'failed'}.")
    else:
        test_func = UNIT_TESTS.get(selected_test)
        if test_func is None:
            print(f"Test '{selected_test}' not found.")
            re_run = input("Would you like to try again? (y/n): ")
            if re_run.lower() == "y":
                unit_tests()
            return
        print(f"Running test: {selected_test}")
        success = test_func()
        print(f"Test {selected_test} {'passed' if success else 'failed'}.")
    print("Unit tests completed.")
    continue_running = input("Continue running the application? (y/n): ")
    if continue_running.lower() != "y":
        sys.exit(0)


def main(argv: list[str]) -> None:
    """Main function for the application."""
    parser = argparse.ArgumentParser(description="Render Ray Application")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable logging to a file.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run the application in test mode (unit testing and such).",
    )
    args = parser.parse_args(argv)
    init_logging(log_to_file=not args.no_log_file, debug=args.debug)
    if args.debug:
        logging.log(logging.DEBUG, "Debug mode is enabled.")
    if args.test_mode:
        logging.log(logging.INFO, "Running in test mode.")
        unit_tests()


def runtime_args(argv: list[str] | None = None) -> None:
    """Allow passing arguments before the main function. (Used for debugging in VS Code)"""
    argv = argv or sys.argv[1:]
    arguments = []
    input_val = ""
    while input_val not in ("end", ""):
        input_val = input("Enter argument (type 'end' or nothing to finish): ")
        if input_val not in ("end", ""):
            arguments.extend(input_val.split(" "))  # Allow multiple arguments at once
    argv.extend(arguments)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--runtime-args" in argv:
        runtime_args(argv)
        argv.remove("--runtime-args")
    main(argv)
