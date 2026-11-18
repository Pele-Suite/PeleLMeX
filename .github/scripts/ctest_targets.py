#!/usr/bin/env python3
"""Print the CMake targets whose executables a build tree's CTest tests run.

Usage: ctest_targets.py BUILD_DIR [CTEST_ARGS...]

The test commands are taken from `ctest --show-only=json-v1 CTEST_ARGS` and
matched against the artifacts of every target, read from the CMake File API
(codemodel-v2; the query is added and the build tree re-configured). The
target names are printed on one line, separated by spaces, ready for
`cmake --build BUILD_DIR --target ...`. Nothing is printed when no test
command names a target, so callers can fall back to building everything.
"""

import json
import pathlib
import shlex
import subprocess
import sys


def target_artifacts(build):
    """Map each artifact path (resolved) to the name of the target making it."""
    query = build / ".cmake" / "api" / "v1" / "query" / "codemodel-v2"
    query.parent.mkdir(parents=True, exist_ok=True)
    query.touch()
    subprocess.run(["cmake", str(build)], check=True, stdout=subprocess.DEVNULL)

    reply = build / ".cmake" / "api" / "v1" / "reply"
    index = json.loads(max(reply.glob("index-*.json")).read_text())
    codemodel = json.loads(
        (reply / index["reply"]["codemodel-v2"]["jsonFile"]).read_text()
    )
    artifacts = {}
    for config in codemodel["configurations"]:
        for target in config["targets"]:
            info = json.loads((reply / target["jsonFile"]).read_text())
            for artifact in info.get("artifacts", []):
                path = pathlib.Path(artifact["path"])
                if not path.is_absolute():
                    path = build / path
                artifacts[str(path.resolve())] = info["name"]
    return artifacts


def test_tokens(build, ctest_args):
    """Yield every word of every test command (bash -c strings are split)."""
    tests = json.loads(
        subprocess.run(
            ["ctest", "--show-only=json-v1", *ctest_args],
            cwd=build,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    for test in tests.get("tests", []):
        for arg in test.get("command", []):
            try:
                words = shlex.split(arg)
            except ValueError:
                words = arg.split()
            yield from words


def main():
    build = pathlib.Path(sys.argv[1]).resolve()
    artifacts = target_artifacts(build)
    targets = set()
    for word in test_tokens(build, sys.argv[2:]):
        if "/" in word:
            name = artifacts.get(str(pathlib.Path(word).resolve()))
            if name:
                targets.add(name)
    print(" ".join(sorted(targets)))


if __name__ == "__main__":
    main()
