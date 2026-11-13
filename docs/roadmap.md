# Roadmap

What comes next, in what order, and what has been set aside. The order is by
what would most change the conclusions, and then by what lets a reader check
them without taking anything on trust.

Status as of 2026-11-13.

## Where things stand

Three tracks share one environment, and each has first results on one model
family: Qwen2.5 at 7B and 14B, quantised to 4 bits, on one machine. The full
list is under [Done](#done).

The strongest claim so far is also the least tested. A rule that needs no
second model, "act only on a target that a trusted source named", cut attack
success from 24% to 1% as a defence and caught 98% of successful attacks as a
monitor. Every attack it faced was a fixed wording written before the rule
existed. None was written by someone who had read the rule.

Two results from other people set the bar for what comes next:

- Nasr et al., [The Attacker Moves Second](https://arxiv.org/abs/2510.09023)
  (2025). Twelve published defences had reported attack success near zero
  against fixed attacks. Attacks that adapt took most of them above 90%.
- Narisetty et al., [Adaptive Evaluation of Out-of-Band Defenses Against
  Prompt Injection in LLM Agents](https://arxiv.org/abs/2606.26479) (2026). A
  defence that enforces a policy outside the model (Progent) held against one
  hand-written adaptive attack, on AgentDojo with Qwen2.5-7B. The authors
  leave search-based attacks untested, and call the result consistent with,
  not proof of, the idea that rules outside the model are harder to adapt
  against.

`grounding` is a defence of that second kind, measured on the same model. So
the next stage asks one question:

> **Does a rule about where a target came from still hold against an attacker
> who has read the rule?**

Items 0 to 5 serve that question. Item 6 is the probing track, kept small on
purpose.

## The plan

Sizes are rough. Small is a day or two, medium about a week, large two to
three weeks.

| | Item | Size | What it produces |
|---|---|---|---|
| 0 | Make every claim checkable | small | Transcripts online, seeds, related work |
| 1 | Attacks that adapt to the defence | large | Every defence at two strengths: fixed and adaptive |
| 2 | Requests between agents carry their origin | medium | Hand-off and compliance rates, before and after |
| 3 | Other models, other machines | medium | The same tables on full-precision, second-family and hosted models |
| 4 | The same rule where other people have numbers | medium | `grounding` on AgentDojo |
| 5 | Write it up | medium | One post, one tagged release |
| 6 | Probes: two repairs and one question | small to medium | A steering result with a fair control |

Items 3 and 4 do not depend on each other and can run alongside item 1. Item 5
starts in item 0 as a skeleton and is filled as results arrive.

### 0. Make every claim checkable

Nothing new is found here. It is first because every later number inherits
these gaps.

- **Transcripts anyone can read.** Inspect logs are too large to commit, so a
  reader today cannot open a single run. Publish them with
  `inspect view bundle`, and link each case the README quotes to its
  transcript.
- **Records that trace to code.** Every record carries the commit that
  produced it. Add a test that each stamp under `results/` resolves to a
  commit in the history, and run it in CI.
- **Variation between runs.** One run per case at temperature 0 says nothing
  about noise between runs. Several seeds at a non-zero temperature for the
  baseline and every defence, with intervals that resample tasks and seeds.
  The 504 attack cases take about 20 minutes on the 7B model, so this is an
  afternoon of compute. The command line seeds the choice of cases today, not
  the sampling.
- **A task the agent cannot do.** The 7B agent never completes
  `late_delivery`. It stops after three steps, attacked or not, and its 42
  attack cases count as 42 failed attacks in every rate. Report attack success
  over the tasks the agent solves clean next to the rate over all tasks, and
  find out why it stops.
- **Related work in the README.** What is the same as AgentDojo and what is
  different (provenance recorded by the tools themselves, harmless and
  near-miss negatives, a fleet), and how `grounding` relates to CaMeL and
  Progent. A reader from the field asks this first.
- **A pinned environment.** A lock file, so the numbers can be rebuilt with
  the versions that produced them.

### 1. Attacks that adapt to the defence

Predictions, written before any of this is run:

- The three prompt defences (`spotlight`, `datamark`, `reminder`) return to
  the undefended rate or above within 20 tries per case.
- Rewording does nothing to `grounding`, because the rule never reads the
  wording. If a reworded attack on a new target gets through, that is a bug in
  the rule, like the lookup hole, and is reported as one.
- `grounding` falls to attacks that need no clever wording at all (1a).

**First, the threat model.** `threat-model.md` says the attacker cannot see
the agent's output or retry. An adaptive attacker does both against a copy of
the system, then plants one text. That document changes before anything is
run.

**1a. Attacks on the rule itself, written by hand.** Four routes, from reading
the rule. None has been run.

- *The other argument.* The rule checks the target of a call (which order,
  which product) and not the rest of it (the new address, the amount, the
  price). All six goals so far name a new target. A goal that keeps the
  operator's own target and changes the other argument should pass.
- *A match somewhere else.* A value counts as named if it appears anywhere in
  trusted text. A 100 percent discount is grounded if the number 100 appears
  in any record the agent has read: a price, a stock count, an order total.
- *Laundering.* Two steps: get the value into a field the shop writes, then
  act on it. The lookup hole was one case. Look for others among the tools
  that write.
- *Blocking the job.* Text that leads an honest agent to an ungrounded action,
  so that the rule stops the work. This attacks usefulness, and is measured by
  tasks solved.

Then count, with no model at all, what the rule leaves open: for each task,
the high-risk calls that pass given only what the operator's request and the
reference plan make trusted. Tightening the rule (checking every argument)
shrinks that count and costs clean tasks. Measure both.

**1b. An attacker model with a budget.** A model rewrites the planted text
after seeing whether it worked, with 1, 5 and 20 tries per case. It is tuned
on some tasks and reported on tasks it never touched. Run it against every
defence, and against defences in combination (`sanitise+grounding`, a detector
in front of both). The code can stack defences and no stacked result has been
recorded.

**1c. Attacks written by people.** The public competition reported in
Dziemian et al., [How Vulnerable Are AI Agents to Indirect Prompt
Injections?](https://arxiv.org/abs/2603.15714) (2026) released 95 successful
attacks against Qwen. Use them as starting points for 1b and as a transfer
test on their own.

**1d. Search with gradients.** Suffix search (GCG) against `spotlight` and
`datamark` through the PyTorch backend. Only if 1b leaves a prompt defence
standing.

**Monitors over the same runs.** Monitors are scored offline, so the adaptive
runs cost nothing more to score. Two questions: does the provenance check
still flag what gets past the defence, and is a monitor that is shown the
planted text moved by text written to move it?

Produces: one table (each defence against fixed wordings, hand-written
attacks, and search at 5 and 20 tries), one figure (attack success against
budget), and the end of the caveat that closes the README.

If the rule is changed in response, it is attacked again and both rounds are
reported.

### 2. Requests between agents carry their origin

In the fleet the weak point was a request passed on word for word with nothing
to say where it came from, and a colleague with a rule it could not check.

- **More cases first.** The hand-off findings rest on 18 hand-offs, and the
  breakdown by goal on groups of five to seven. Run all 504 cases on the fleet
  with seeds. One arm on 72 cases takes about ten minutes on the 14B model.
- **The rule, for messages.** A request is ungrounded if it names an order,
  address or code that the sender only ever saw in outsider-written text. The
  check exists for tool calls. It should exist for messages.
- **The origin, carried.** The receiving agent is told "this came from a
  ticket" and applies its own rule.
- **Then attack it** as in item 1: a sender that restates the demand in its
  own words, and a demand split between two colleagues.

Produces: hand-off and compliance rates before and after, under fixed and
adaptive attacks.

### 3. Other models, other machines

- **Full-precision weights.** The PyTorch backend has only been exercised on
  small random networks in tests. Run it on a rented GPU: the same model and
  cases at 4 bits and at 16, compared case by case, to see whether any finding
  is about quantisation.
- **A current open model from another family.** Qwen2.5 is two years old.
- **Hosted models**, through Inspect's own providers, on the 72-case set.
  Fixed wordings will probably do little there. In the competition above,
  attack success against frontier models ran from 0.5% to 8.5% of attempts.
  That is the reason item 1 comes first: an attacker that adapts is what keeps
  the benchmark informative on a strong model.
- **The trusted model.** A stronger one, and the agent's own model as its
  monitor, to separate "the monitor is weak" from "the signal is not there".
- **Control tokens in tool output.** The most effective wording works because
  the serving stack tokenises control tokens found in tool output as control
  tokens. Check which other stacks do the same (vLLM, llama.cpp, Ollama,
  transformers) and report it upstream where they do.

### 4. The same rule where other people have numbers

Eleven tasks written by the author of the rule are a weak test of the rule.

- Implement `grounding` for AgentDojo: 97 tasks written by other people, with
  published numbers for CaMeL, Progent and others on the same cases. The work
  is in marking which fields outsiders write. AgentDojo's tools do not say.
- Report what it costs in tasks solved there. If the cost is high, that is the
  finding.
- Offer upstream whatever is useful to others.

This also answers the older item "more tasks, written by someone who has not
seen the attacks", without writing them.

### 5. Write it up

- In item 0: the skeleton of one post. The question, the empty table, the
  figure to be drawn. Each later run fills a cell.
- One post of about 2,000 words: the question, one table, one figure, and what
  went wrong along the way. The top of the README cut to one screen that says
  the same.
- A tagged release that the post's numbers come from.
- A short paper only if items 1 and 4 give a result that holds.

### 6. Probes: two repairs and one question

The probing track made two promises it has not kept. Keep them, and stop
there unless the question below turns out to be worth asking.

- **Steering with a fair control.** A dose-response curve: several magnitudes,
  each with random directions matched in standard deviations of natural
  variation, not in length.
- **Negatives that make the question hard.** Notes matched to attacks in
  length, and notes that are themselves instructions the agent should follow.
  A sentence-embedding baseline, which is the fair text-only competitor.
- **The question.** Can the attacker of 1b succeed while keeping the
  compliance probe quiet, with the probe's score added to what it searches
  against? Probes on other behaviours have been evaded this way (Bailey et
  al., Obfuscated Activations Bypass LLM Latent-Space Defenses). Only worth
  running if the first two repairs leave a probe worth attacking: one that
  catches 28% of attacks at a 5% false-alarm rate is not.

## What would change the plan

- **1a finds the rule open on most tasks.** Then the result is that the rule
  is narrower than its 1% suggests. Tighten it, measure what that costs in
  clean tasks, and write up the trade.
- **1b does not break the prompt defences at 20 tries.** Run 1d before
  claiming that they hold.
- **Hosted models sit at zero against everything, adaptive attacks included.**
  Say so, and keep the open models as the main testbed.
- **`grounding` blocks too much honest work on AgentDojo.** Report it. A rule
  that only works on tasks its author wrote is worth knowing about.

## Set aside

Each of these is worth doing and none of them bears on the question above.

- **Control protocols.** Defer to the trusted model when a call is flagged
  instead of blocking. Resample a suspicious step. Split one audit budget
  across agents by how much damage each can do. This needs monitors that run
  during the job and not after it, and an insider with a deliberate attack
  policy: the insider here mostly declined its side task, so its low rates say
  little. Two insiders that cooperate.
- **Probing in depth.** Where in the prompt the signal comes from: attention
  from the last token to the planted text, and patching activations between
  an attacked run and its benign twin. Sparse features in place of single
  directions. Transfer between models and quantisations. Probes at the token
  where a tool call's name or arguments are written.
- **The environment.** Real sandboxes for tools that touch files or the
  network. Longer jobs with many tickets in one session, where an injection
  read early can act late. Injection position and length as their own axes.
- **More prompt defences.** Three were measured and none clearly worked.
  Item 1 will say whether that is the last word.

## Not planned

- A leaderboard. The numbers here are for comparing defences on one model
  under one threat model, and stop meaning much outside that.
- Attacks on deployed systems. Everything stays inside the simulated shop.

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
