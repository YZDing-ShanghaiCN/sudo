"""CLI entry point: python -m deployenv_checker --config scene.yaml

Run from the project root:
    PYTHONPATH=hbmp:src python -m deployenv_checker --config configs/example_scene.yaml
"""

import argparse

from .app import App


def main():
    parser = argparse.ArgumentParser(description="Robot Deploy Environment Checker")
    parser.add_argument("--config", required=True, help="Path to scene YAML config")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    args = parser.parse_args()

    app = App(args.config, args.port)
    app.run()


if __name__ == "__main__":
    main()
