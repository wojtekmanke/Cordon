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
- 396 negative cases: harmless notes, and near-miss notes that mention what
  attacks mention, planted where an attack would go.
- 8 defences acting on the prompt, the text or the action.
- Results on a 7B model: the baseline on all 504 cases, each defence on a
  fixed 168-case subset. See `results/bench.md`.

**Track 2: fleet control evaluation**

- Four agents with split privileges and a single-agent comparison arm.
- Taint and provenance that follow requests and answers between agents.
- Injection and insider threats.
- Monitors scored offline, judged at a fixed audit rate.
- Results on a 14B model, 72 stratified cases per arm, and monitors scored
  over the 7B and 14B runs. See `results/fleet.md` and `results/monitor.md`.

**Track 3: probing**

- Activations read at the moment an agent has read planted text.
- Probes scored only on held-out wordings, goals and tasks, against text,
  length and label baselines and a shuffled-label control.
- Steering along a probe direction, run through the benchmark like any other
  defence, with random-direction controls.
- Results on the 7B model. See `results/interp.md`.

## What the first results changed

Three items below were added or moved up because of what the first runs
showed.

- The attack-against-note probe is not a hard enough question. Length alone
  separates the classes and a layer-1 probe nearly matches the best layer.
  Item 4 is the fix.
- Steering along the compliance direction was matched by a random direction
  of the same length, which is a much larger push in terms of natural
  variation. Item 5 is the proper experiment.
- In the fleet, the weak point was a rule an agent could not check. Item 6
  follows from that.

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

### 4. Negatives that make the probing question hard

- Notes matched to attacks in length, and notes that are themselves
  instructions the agent should follow because they fit the operator's
  request. Only then does "attack or not" ask about more than vocabulary.
- A sentence-embedding baseline, which is the fair text-only competitor for a
  probe on a model's own activations.

### 5. Steering done properly

- A dose-response curve: several magnitudes, each with random directions
  matched in standard deviations of natural variation, not in length.
- Directions fitted at the token where a tool call's name is produced, where
  the decision to act is made, and not only where the reply begins.

### 6. Provenance for requests between agents

- A request is ungrounded if it names an order, address or code that the
  sending agent only ever saw in outsider-written text. The check exists for
  tool calls; it should exist for messages.
- Carry the origin of a request with it, so the receiving agent is told "this
  came from a ticket" and can apply its own rule.

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
