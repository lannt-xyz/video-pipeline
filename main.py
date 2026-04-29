"""Video Production Pipeline — entry point.

Usage:
    uv run python main.py                           # run full pipeline from episode 1
    uv run python main.py --episode 1               # run episode 1 end-to-end
    uv run python main.py --episode 1 2 5           # run specific episodes (skips others)
    uv run python main.py --episode 1-5             # run a range of episodes (inclusive)
    uv run python main.py --episode 1 3-5 8         # mix singles + ranges
    uv run python main.py --episode 1 --from-phase llm   # resume episode 1 from LLM phase
    uv run python main.py --episode 1 --from-phase images
    uv run python main.py --from-episode 5          # resume full pipeline from episode 5
    uv run python main.py --dry-run                 # validate config + log plan, no execution

    uv run python main.py --probe-images 1          # inspect character→scene matching (no ComfyUI)
    uv run python main.py --probe-images 1 --probe-shots 3  # inspect + generate first 3 shots into data/probe/
"""

from pipeline.orchestrator import main

if __name__ == "__main__":
    main()
