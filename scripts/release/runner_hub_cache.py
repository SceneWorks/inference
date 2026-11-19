#!/usr/bin/env python3
"""Print the runner's Hugging Face hub cache, where every real-weight Hub download lands.

For CI steps that build a snapshot path themselves (`<cache>/models--<org>--<name>/snapshots/<rev>`)
and need it in the same shell; `ensure_model_snapshot.py --export` covers the later-steps case.
"""

if __package__:
    from .ensure_model_snapshot import runner_hub_cache
else:
    from ensure_model_snapshot import runner_hub_cache

if __name__ == "__main__":
    print(runner_hub_cache())
