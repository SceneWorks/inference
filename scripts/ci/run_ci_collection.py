#!/usr/bin/env python3
"""Run an unchanged CI command, preserving failure and stopping on runner poison."""

import argparse
import errno
import os
import shlex
import subprocess
import sys


RESOURCE_ERRORS = ("no space left on device", "disk quota exceeded")
# This native Metal error poisons the process; it is not an ordinary assertion failure.
NATIVE_POISON_ERRORS = ("SubmissionsIgnored",)


def mark_blocked(variable, reason):
    env = os.environ.get("GITHUB_ENV")
    if env:
        with open(env, "a", encoding="utf-8") as stream:
            stream.write(f"{variable}=true\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(f"\nDownstream collection **blocked**: {reason}. This is not an independent test failure.\n")


def run_command(command):
    print(f"::group::{shlex.join(command)}", flush=True)
    resource_failure = False
    native_fatal = False
    compile_blocked = False
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            errors="replace",
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            resource_failure |= any(error in line.lower() for error in RESOURCE_ERRORS)
            native_fatal |= any(error in line for error in NATIVE_POISON_ERRORS)
            compile_blocked |= "could not compile" in line
        code = process.wait()
    except OSError as error:
        print(str(error), flush=True)
        resource_failure = error.errno in (errno.ENOSPC, errno.EDQUOT)
        code = 1
    print("::endgroup::", flush=True)
    resource_failure |= code < 0
    if resource_failure:
        mark_blocked("CI_COLLECTION_RESOURCE_FAILURE", "storage exhaustion or terminated command")
    if native_fatal:
        mark_blocked("CI_COLLECTION_NATIVE_FATAL", "native Metal SubmissionsIgnored poison")
    if resource_failure or native_fatal:
        code = code or 1
    compile_blocked &= command[0] == "cargo" and "test" in command
    if code and compile_blocked:
        print("::warning::Compilation failed; dependent test binaries are compile-blocked, not executed. "
              "Other independent configurations may still run.", flush=True)
    print(f"CI command exit_code={code} resource_failure={resource_failure} "
          f"native_fatal={native_fatal} compile_blocked={bool(code and compile_blocked)}", flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required")
    if any(os.environ.get(variable) == "true" for variable in
           ("CI_COLLECTION_RESOURCE_FAILURE", "CI_COLLECTION_NATIVE_FATAL")):
        print("Collection blocked by an earlier resource/native fatal failure", file=sys.stderr)
        return 1
    code = run_command(command)
    return code if code >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
