import json
import argparse

import uvicorn

from train import train_and_run_ablation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DL+RL trading MVP runner")
    parser.add_argument("--mode", choices=["train", "serve", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=4, help="PPO epochs per ablation mode")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.mode in {"train", "all"}:
        results = train_and_run_ablation(epochs=args.epochs)
        print("Ablation + training complete")
        print(json.dumps(results, indent=2))

    if args.mode in {"serve", "all"}:
        uvicorn.run("backend_api:app", host=args.host, port=args.port, reload=False)
