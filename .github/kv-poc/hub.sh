#!/usr/bin/env bash
# The optional HF hub override of kv-poc-campaign.yml (run json / dispatch `hf_hub_override`).
#
# The default hub is /Volumes/Models/huggingface/hub. When a run names an override, every macOS job
# uses it instead (download, verify, precheck, every phase), it is created when missing, nothing is
# seeded into it (the build/w2-assets stage downloads every pinned file through it and the campaign
# parents still sha256 every file, so evidence identity is unchanged), and the `cleanup` job deletes
# it after every phase, whatever their outcome.
#
# An override must be a plain absolute path strictly under the runner user's $HOME, on $HOME's own
# volume (the internal disk), reached through no symlink. The cleanup `rm -rf` runs only behind the
# same checks, so it can never reach /Volumes/Models or anything outside the validated path.
#
#   source hub.sh                         the functions below (config.sh, common.sh, tests)
#   bash hub.sh cleanup <path> [<home>]   delete a validated override (home defaults to $HOME)
#
# Bash 3.2 (the runners' /bin/bash): no mapfile, no ${x,,}.

KV_DEFAULT_HF_HUB=/Volumes/Models/huggingface/hub

# Syntax only (no filesystem access, so the hosted `config` job can run it before a self-hosted
# job queues). <home> is the runner's $HOME, or empty on the hosted job, where only the macOS
# home layout /Users/<user>/... is required. Prints the problem and returns 1; silent 0 when valid.
hub_override_syntax_problem() {
  local path="$1" home="${2:-}" rest component
  case "$path" in
    /*) ;;
    *) echo "hf_hub_override must be an absolute path, got '$path'"; return 1 ;;
  esac
  case "$path" in
    *[!A-Za-z0-9._/-]*) echo "hf_hub_override may only contain A-Z a-z 0-9 . _ - /, got '$path'"; return 1 ;;
    *//*|*/) echo "hf_hub_override must not contain '//' or end in '/', got '$path'"; return 1 ;;
  esac
  rest="${path#/}"
  while [ -n "$rest" ]; do
    component="${rest%%/*}"
    case "$component" in .|..|-*) echo "hf_hub_override must not contain a '.', '..' or '-'-leading component, got '$path'"; return 1 ;; esac
    [ "$component" = "$rest" ] && break
    rest="${rest#*/}"
  done
  if [ -n "$home" ]; then
    case "$home" in
      /?*) ;;
      *) echo "the runner's HOME '$home' is not an absolute directory"; return 1 ;;
    esac
    case "$path" in
      "$home"/?*) ;;
      *) echo "hf_hub_override must be strictly under the runner's HOME ($home/...), got '$path'"; return 1 ;;
    esac
  else
    case "$path" in
      /Users/*/?*) ;;
      *) echo "hf_hub_override must be strictly under a macOS home (/Users/<user>/...), got '$path'"; return 1 ;;
    esac
  fi
  case "$path" in
    "$KV_DEFAULT_HF_HUB"|"$KV_DEFAULT_HF_HUB"/*|/Volumes|/Volumes/*) echo "hf_hub_override must not be on /Volumes, got '$path'"; return 1 ;;
  esac
  return 0
}

hub_device() { stat -c %d "$1" 2>/dev/null || stat -f %d "$1"; }

# Filesystem checks on the runner: the deepest existing prefix of <path> (the path itself once it
# exists) is a real directory reached through no symlink, on the same device as <home>.
hub_override_physical_problem() {
  local path="$1" home="$2" probe
  probe="$path"
  while [ ! -e "$probe" ] && [ ! -L "$probe" ]; do probe="$(dirname "$probe")"; done
  if [ -L "$probe" ] || [ ! -d "$probe" ]; then
    echo "hf_hub_override: $probe is a symlink or not a directory"; return 1
  fi
  if [ "$(cd -P "$probe" && pwd -P)" != "$probe" ]; then
    echo "hf_hub_override: $probe resolves through a symlink to $(cd -P "$probe" && pwd -P)"; return 1
  fi
  if [ "$(hub_device "$probe")" != "$(hub_device "$home")" ]; then
    echo "hf_hub_override: $probe is not on the volume of $home (the internal disk)"; return 1
  fi
  return 0
}

hub_override_problem() { # <path> <home>: syntax, then the filesystem checks
  hub_override_syntax_problem "$1" "$2" && hub_override_physical_problem "$1" "$2"
}

# Create a validated override (re-checked after mkdir). Prints the problem and returns 1 otherwise.
hub_override_prepare() { # <path> <home>
  hub_override_problem "$1" "$2" || return 1
  mkdir -p "$1" || { echo "hf_hub_override: cannot create $1"; return 1; }
  hub_override_physical_problem "$1" "$2"
}

# Delete a validated override. Refuses (exit 1, deleting nothing) unless <path> passes every check
# above; a path that no longer exists is already clean.
hub_override_cleanup() { # <path> <home>
  local path="$1" home="$2"
  hub_override_syntax_problem "$path" "$home" || return 1
  if [ ! -e "$path" ] && [ ! -L "$path" ]; then echo "hf_hub_override $path does not exist; nothing to delete"; return 0; fi
  hub_override_physical_problem "$path" "$home" || return 1
  echo "deleting the run's HF hub override $path ($(du -sh "$path" 2>/dev/null | cut -f1))"
  chmod -R u+w "$path" 2>/dev/null || true
  rm -rf -- "$path"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case "${1:-}" in
    cleanup)
      [ -n "${2:-}" ] || { echo "usage: hub.sh cleanup <path> [<home>]" >&2; exit 2; }
      if ! problem="$(hub_override_cleanup "$2" "${3:-$HOME}")"; then
        echo "::error title=hub override cleanup refused::$problem"
        exit 1
      fi
      printf '%s\n' "$problem"
      ;;
    *) echo "usage: hub.sh cleanup <path> [<home>]" >&2; exit 2 ;;
  esac
fi
