"""Video Production Pipeline — entry point.

Usage:
    python main.py                           # run full pipeline from episode 1
    python main.py --episode 1               # run episode 1 end-to-end
    python main.py --episode 1 --from-phase llm   # resume episode 1 from LLM phase
    python main.py --from-episode 5          # resume full pipeline from episode 5
    python main.py --dry-run                 # validate config + log plan, no execution
"""

from pipeline.orchestrator import main

if __name__ == "__main__":
    main()
