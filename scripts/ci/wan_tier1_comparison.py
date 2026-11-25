#!/usr/bin/env python3
"""Original Wan command line for its fixed curated Tier 1 dispatch."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.ci.video_tier1_comparison import main
from scripts.ci.video_tier1_cohorts import WAN

if __name__ == "__main__":
    main(allowed=(WAN,))
