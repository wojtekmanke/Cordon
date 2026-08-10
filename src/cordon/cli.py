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
    model_args = dict(pair.split("=", 1) for pair in args.model_arg)
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
                model_args=model_args,
                label=args.label,
            )
            defence = args.label or defence
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


def _monitor(args: argparse.Namespace) -> None:
    import asyncio
    import glob
    import json

    from inspect_ai.model import get_model

    from . import _registry  # noqa: F401
    from .monitor import (
        llm_monitor,
        load_events,
        score_events,
        tainted_monitor,
        ungrounded_monitor,
    )
    from .run import write_rows

    events, counts = load_events(sorted(glob.glob(f"{args.logs}/*.eval")))
    if args.kind in ("llm", "llm+reads"):
        if not args.model:
            raise SystemExit("--model is required for an LLM monitor")
        monitor = llm_monitor(get_model(args.model), show_reads=args.kind == "llm+reads")
    else:
        monitor = {"ungrounded": ungrounded_monitor, "tainted": tainted_monitor}[args.kind]
    started = time.time()
    rows = asyncio.run(score_events(events, monitor))
    for row in rows:
        row["id"] = f"{row['run']}/{row['case']}/{row['index']:02d}"
        row["monitor"] = args.name
    out = RESULTS / "monitor" / args.set
    write_rows(out / f"{args.name}.jsonl", rows)
    (out / "counts.json").write_text(json.dumps(counts, indent=1) + "\n")
    print(f"wrote {len(rows)} scored events to {out}/{args.name}.jsonl in "
          f"{time.time() - started:.0f}s; runs: {counts}", flush=True)  # fmt: skip


def _backend(model: str):
    from .models.local import MLXBackend, TorchBackend

    kind, _, name = model.partition("/")
    if kind not in ("mlx", "torch"):
        raise SystemExit("activations need a local model: mlx/<repo> or torch/<repo>")
    return MLXBackend(name) if kind == "mlx" else TorchBackend(name)


def _interp_collect(args: argparse.Namespace) -> None:
    import glob
    import json

    import numpy as np

    from .interp.dataset import examples_from_logs, save

    backend = _backend(args.model)
    paths = sorted(glob.glob(f"{args.logs}/*.eval"))
    examples, skipped = examples_from_logs(paths, backend.render)
    print(f"{len(examples)} examples; {skipped} runs never read the planted text", flush=True)
    started = time.time()
    activations = np.stack([backend.hidden(example.prompt) for example in examples])
    out = Path(args.out)
    save(out, examples, activations)
    meta = {"model": args.model, "examples": len(examples), "never_read": skipped}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {activations.shape} to {out} in {time.time() - started:.0f}s", flush=True)


def _interp_probe(args: argparse.Namespace) -> None:
    import json

    from .interp.dataset import load
    from .interp.experiments import compare, early_warning, layer_sweep

    directory = Path(args.dir)
    examples, acts = load(directory)
    meta = json.loads((directory / "meta.json").read_text())
    result = {**meta, "layers": acts.shape[1], "width": acts.shape[2], "sweep": {}, "compare": {}}
    for label in ("injected", "attempted"):
        sweep = layer_sweep(examples, acts, label)
        result["sweep"][label] = sweep
        # The layer is picked on held-out templates, then judged on the other splits too.
        layer = args.layer or max(sweep, key=lambda row: row["auroc"])["layer"]
        result["compare"][label] = {"layer": layer, "splits": compare(examples, acts, label, layer)}
        print(f"{label}: layer {layer}", flush=True)
    result["early_warning"] = early_warning(examples, acts, result["compare"]["injected"]["layer"])
    result["counts"] = {
        "examples": len(examples),
        "injected": sum(e.injected for e in examples),
        "attempted": sum(e.attempted for e in examples),
        "succeeded": sum(e.attack for e in examples),
    }
    out = RESULTS / "interp" / directory.name / "probes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {out}")


def _interp_direction(args: argparse.Namespace) -> None:
    import numpy as np

    from .bench.cases import attack_cases, sample_cases
    from .interp.dataset import load
    from .interp.experiments import select
    from .interp.probes import direction, random_like

    directory = Path(args.dir)
    examples, acts = load(directory)
    # Fit on cases outside the subset the intervention will be tested on.
    held = {case.id for case in sample_cases(attack_cases(), args.exclude, args.seed)}
    keep = np.array([e.id not in held for e in examples])
    chosen, x, y = select(
        [e for e, k in zip(examples, keep, strict=True) if k], acts[keep], args.label
    )
    vector = direction(x[:, args.layer - 1], y).astype(np.float32)
    name = f"{args.label}-L{args.layer}"
    np.save(directory / f"{name}.npy", vector)
    np.save(directory / f"{name}-random.npy", random_like(vector, seed=args.seed))
    print(
        f"wrote {directory / name}.npy from {len(chosen)} examples "
        f"({int(y.sum())} positive), norm {np.linalg.norm(vector):.2f}"
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
    bench.add_argument(
        "--model-arg", action="append", default=[], help="key=value passed to the model provider"
    )
    bench.add_argument("--label", help="name for this condition in the records")
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

    monitor = commands.add_parser("monitor", help="score recorded runs with a monitor")
    monitor.add_argument("--logs", required=True, help="directory of Inspect logs")
    monitor.add_argument("--set", required=True, help="name for this set of runs")
    monitor.add_argument("--name", required=True, help="name for this monitor in the records")
    monitor.add_argument(
        "--kind", required=True, choices=["llm", "llm+reads", "ungrounded", "tainted"]
    )
    monitor.add_argument("--model", help="the monitor model, for the LLM kinds")
    monitor.set_defaults(run=_monitor)

    interp = commands.add_parser("interp", help="probe a model's activations")
    steps = interp.add_subparsers(dest="step", required=True)
    collect = steps.add_parser("collect", help="read activations at each recorded decision")
    collect.add_argument("--model", required=True, help="mlx/<repo> or torch/<repo>")
    collect.add_argument("--logs", required=True, help="directory of undefended benchmark logs")
    collect.add_argument("--out", required=True)
    collect.set_defaults(run=_interp_collect)
    probe = steps.add_parser("probe", help="fit probes and baselines on held-out groups")
    probe.add_argument("--dir", required=True, help="output directory of collect")
    probe.add_argument("--layer", type=int, help="layer to compare at; default: best in sweep")
    probe.set_defaults(run=_interp_probe)
    direct = steps.add_parser("direction", help="save a difference-of-means direction")
    direct.add_argument("--dir", required=True, help="output directory of collect")
    direct.add_argument("--label", required=True, choices=["injected", "attempted"])
    direct.add_argument("--layer", type=int, required=True, help="layer, counting from 1")
    direct.add_argument("--exclude", type=int, default=168, help="size of the test subset")
    direct.add_argument("--seed", type=int, default=0)
    direct.set_defaults(run=_interp_direction)

    report = commands.add_parser("report", help="rebuild the result tables from results/")
    report.add_argument("--out", default="docs/results")
    report.set_defaults(run=_report)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
