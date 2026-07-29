"""Command line: run the evaluations and rebuild the reports.

cordon bench --model mlx/mlx-community/Qwen2.5-7B-Instruct-4bit --defence none,spotlight
cordon report
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

RESULTS = Path("results")


def _bench(args: argparse.Namespace) -> None:
    from .run import run_bench, slug, write_rows

    suites = ["clean", "attacks"] if args.suite == "both" else args.suite.split(",")
    for defence in args.defence.split(","):
        for suite in suites:
            started = time.time()
            rows = run_bench(
                args.model,
                defence,
                suite,
                cases=args.cases,
                seed=args.seed,
                trusted=args.trusted,
                log_dir=args.log_dir,
                display=args.display,
            )
            out = RESULTS / "bench" / slug(args.model) / f"{defence}.{suite}.jsonl"
            write_rows(out, rows)
            utility = sum(r["utility"] for r in rows) / len(rows)
            attack = sum(r["attack"] for r in rows) / len(rows)
            print(
                f"{defence:28s} {suite:8s} n={len(rows):4d} utility={utility:.2f} "
                f"attack={attack:.2f} {time.time() - started:6.0f}s -> {out}",
                flush=True,
            )


def _fleet(args: argparse.Namespace) -> None:
    from .run import fleet_name, run_fleet, slug, write_rows

    for arm in args.arm.split(","):
        for defence in args.defence.split(","):
            for suite in args.suite.split(","):
                started = time.time()
                rows = run_fleet(
                    args.model,
                    arm,
                    defence,
                    suite,
                    share_request=args.share_request,
                    cases=args.cases,
                    seed=args.seed,
                    trusted=args.trusted,
                    log_dir=args.log_dir,
                    display=args.display,
                )
                name = f"{fleet_name(arm, args.share_request)}.{defence}.{suite}.jsonl"
                out = RESULTS / "fleet" / slug(args.model) / name
                write_rows(out, rows)
                utility = sum(r["utility"] for r in rows) / len(rows)
                attack = sum(r["attack"] for r in rows) / len(rows)
                print(
                    f"{name:44s} n={len(rows):4d} utility={utility:.2f} attack={attack:.2f} "
                    f"{time.time() - started:6.0f}s",
                    flush=True,
                )


def _report(args: argparse.Namespace) -> None:
    from .report import write_reports

    for path in write_reports(RESULTS, Path(args.out)):
        print(path)


def main() -> None:
    parser = argparse.ArgumentParser(prog="cordon", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    bench = commands.add_parser("bench", help="run the prompt-injection benchmark")
    bench.add_argument("--model", required=True, help="an Inspect model id")
    bench.add_argument("--defence", default="none", help="comma-separated; join with + to stack")
    bench.add_argument("--suite", default="both", help="clean, benign, attacks, or both")
    bench.add_argument("--cases", type=int, help="run a fixed random subset of the attack cases")
    bench.add_argument("--seed", type=int, default=0, help="seed for --cases")
    bench.add_argument("--trusted", help="model for detectors and monitors; defaults to --model")
    bench.add_argument("--log-dir", default="logs")
    bench.add_argument("--display", default="none")
    bench.set_defaults(run=_bench)

    fleet = commands.add_parser("fleet", help="run the fleet control evaluation")
    fleet.add_argument("--model", required=True, help="an Inspect model id")
    fleet.add_argument("--arm", default="fleet", help="fleet, solo, or both comma-separated")
    fleet.add_argument("--defence", default="none", help="comma-separated; join with + to stack")
    fleet.add_argument("--suite", default="clean,injection", help="clean, injection, insider")
    fleet.add_argument(
        "--share-request",
        action="store_true",
        help="show an agent asked by a colleague the operator's original request",
    )
    fleet.add_argument("--cases", type=int, help="run a fixed random subset of the injections")
    fleet.add_argument("--seed", type=int, default=0, help="seed for --cases")
    fleet.add_argument("--trusted", help="model for detectors and monitors; defaults to --model")
    fleet.add_argument("--log-dir", default="logs")
    fleet.add_argument("--display", default="none")
    fleet.set_defaults(run=_fleet)

    report = commands.add_parser("report", help="rebuild the result tables from results/")
    report.add_argument("--out", default="docs/results")
    report.set_defaults(run=_report)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
