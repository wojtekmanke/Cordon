# Roadmap

What exists, what is being worked on, and what comes next. The order is by
what would most change the conclusions, not by what is easiest.

Status as of 2026-10-12, the day the first version was written.

## Done

**Shared ground**

- A simulated shop: state, seed data, 17 tools. Each tool marks which fields
  outsiders write and whether a call is high-risk.
- An agent loop with hooks for defences, and provenance tracking of every
  string the agent has read.
- Model providers for Inspect: MLX and PyTorch for local open models, and
  scripted models so the pipeline runs in CI with no weights.
- Per-case records, reports and figures all rebuilt from those records.

**Track 1: prompt-injection benchmark**

- 11 tasks, each with a reference plan that proves it solvable.
- 6 attacker goals, 7 wordings, 12 injection points: 504 attack cases.
- 252 benign cases: harmless text planted where an attack would go.
- 8 defences acting on the prompt, the text or the action.
- First results on a 7B model. See `results/bench.md`.

**Track 2: fleet control evaluation**

- Four agents with split privileges and a single-agent comparison arm.
- Taint and provenance that follow requests and answers between agents.
- Injection and insider threats.
- Monitors scored offline, judged at a fixed audit rate.
- First results on a 14B model. See `results/fleet.md`.

**Track 3: probing**

- Activations read at the moment an agent has read planted text.
- Probes scored only on held-out wordings, goals and tasks, against
  baselines that read the text alone.
- Steering along a probe direction, run through the benchmark like any other
  defence.
- First results on the 7B model. See `results/interp.md`.

## Next: what would most change the conclusions

### 1. Attacks that adapt to the defence

Every defence result so far is against fixed wordings written before the
defence existed. That measures the defence against a lazy attacker.

- An attacker model that rewrites the injection after seeing whether it
  worked, with a fixed query budget per case.
- For local models, gradient-guided suffix search against the spotlight and
  datamark prompts.
- Attacks written against `grounding` specifically: goals whose target is
  already in trusted data, and two-step attacks that first get the target
  into a trusted field.
- Report every defence at two strengths: fixed wordings, and adaptive with
  budget N.

### 2. Variation between runs, and more tasks

- Several seeds at a non-zero temperature, so intervals include run-to-run
  variation and not only the choice of cases.
- More tasks, written by someone who has not seen the attacks, with a held-out
  set that no prompt or defence is tuned on. Eleven tasks is the main reason
  the headline intervals are wide.
- Injection position (start, middle, end of the field) and length as their own
  axes.

### 3. Stronger and different models

- Hosted models through Inspect's own providers. Nothing in the harness is
  specific to local models; this needs API keys and a budget.
- A second open model family, to see which findings are about Qwen.
- A stronger trusted model, and the same model as its own monitor, to
  separate "the monitor is weak" from "the signal is not there".

## Later

**Control protocols.** Defer to the trusted model when a call is flagged,
instead of blocking. Resample a suspicious step. Split one audit budget across
agents by how much damage each can do. Insider attack policies that try to
look ordinary, and two insiders that cooperate.

**Probing in depth.** Where in the prompt the signal comes from: attention
from the last token to the planted text, and patching activations between an
attacked run and its benign twin. Sparse features in place of single
directions. Whether a probe fitted on one model or one quantisation transfers
to another. Probes at the token where the tool call's arguments are written,
not only where the reply begins.

**The environment.** Real sandboxes for tools that touch files or the
network. Longer jobs with many tickets in one session, where an injection
read early can act late.

## Not planned

- A leaderboard. The numbers here are for comparing defences on one model
  under one threat model, and stop meaning much outside that.
- Attacks on deployed systems. Everything stays inside the simulated shop.
