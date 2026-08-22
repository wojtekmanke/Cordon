# cordon

Safety evaluations for language-model agents that do real work with tools,
set in the back office of a simulated online shop.

An agent that reads a customer's message can be told what to do by that
customer. This repository measures how often that works, what stops it, what
stopping it costs, and whether the model's own activations show it happening.
It has three tracks that share one environment:

1. **A prompt-injection benchmark.** 11 jobs, 6 things an attacker wants, 7
   ways of wording the demand, planted in tickets, reviews, emails and product
   data: 504 attack cases, plus 252 cases with harmless text in the same
   places. Eight defences are measured on attack success and on usefulness.
2. **A control evaluation for a fleet of agents.** Four agents with split
   privileges against one agent that holds everything. Does separating
   privileges contain an injection, or does one agent pass the attacker's
   request to another as if it were internal work? Monitors are judged by the
   attacks they catch at a fixed amount of human auditing.
3. **Probes on the model's activations.** At the moment an agent has read
   planted text and not yet acted, can a linear probe tell an attack from a
   harmless note, and predict whether the agent will obey? Probes are scored
   only on wordings, goals and tasks held out of training, against baselines
   that read the text alone.

Everything runs on open models on one machine. No API keys are needed.

RESULTS_PLACEHOLDER

## Run it

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev,mlx,interp]"   # mlx needs Apple Silicon; omit it elsewhere
.venv/bin/pytest -q                             # no model weights needed
```

Try the pipeline with scripted models, which need no GPU. `gullible` obeys
every planted instruction and `oracle` ignores them all, so the numbers are
known in advance:

```sh
.venv/bin/cordon bench --model scripted/gullible --defence none,grounding --cases 60
```

With a real model (any Inspect model id works; these run locally through MLX):

```sh
.venv/bin/cordon bench --model mlx/mlx-community/Qwen2.5-7B-Instruct-4bit --defence none,spotlight --cases 60
.venv/bin/cordon report            # rebuilds docs/results and docs/figures
.venv/bin/inspect view --log-dir logs   # read any transcript
```

The `Makefile` lists the exact commands behind the results above.

## How it is put together

```
src/cordon/
  shop/        the simulated shop: state, seed data, 17 tools
  agent.py     the agent loop, with hooks for defences and provenance tracking
  defences.py  eight defences acting on the prompt, the text or the action
  bench/       tasks, attacker goals, wordings, the case matrix, Inspect tasks
  fleet/       four roles with split privileges, delegation, Inspect tasks
  monitor.py   monitors scored offline over recorded runs
  interp/      probing dataset, probes, baselines, experiments
  models/      Inspect providers: MLX, PyTorch, and scripted models for tests
  stats.py     intervals; report.py and figures.py rebuild everything from records
```

A few decisions shape the rest, and [docs/design.md](docs/design.md) gives the
reasons:

- **Outcomes are read from the shop's final state.** No judge model reads the
  transcript, so an agent cannot talk its way to a pass.
- **Every string the agent reads is tagged by who wrote it**, and every call
  records whether its target came from a trusted source. Defences, monitors
  and later analysis all read the same tags.
- **Headline intervals resample whole tasks.** 504 cases over 11 tasks are not
  504 independent trials.
- **No probe is scored on a random split.** Planted attacks share wording, and
  a random split rewards learning the wording.

## What the results do not show

The attacks are fixed wordings written before the defences. A defence that
stops them has not been shown to stop an attacker who knows the defence and
searches for a way round it; that is the first item on the
[roadmap](docs/roadmap.md). The agents are small open models that make
mistakes with no attacker present. There are eleven tasks, one run per case,
and the shop is simulated. [docs/threat-model.md](docs/threat-model.md) sets
out what is assumed and what is out of scope.

## Licence

MIT.
