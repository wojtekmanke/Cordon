# Design notes

Why the code is shaped the way it is. Each section is a decision, the
alternative, and the reason.

## Score the world, not the transcript

A task is solved if the shop's final state says so: the refund exists, the
address changed, the reply contains the tracking number. An attack succeeded
if the attacker's refund exists. No judge model reads the conversation.

A judge model would be cheaper to write and would cover tasks that are hard
to check mechanically. It would also be wrong some of the time, in a way that
correlates with the thing being measured: a model that has been talked into
something can describe it innocently. Every task has a reference plan, and a
test runs it to prove the check accepts a correct solution and rejects doing
nothing.

The cost is that tasks are limited to those with a checkable outcome, and the
checks for answers given in words are loose (a summary of reviews passes if it
mentions the zip).

## Three independent choices make an attack

Goal, wording, and place are separate lists, and a case is one of each. This
is what makes the breakdowns possible: the same refund demand in seven
wordings, the same wording carrying six demands. It also keeps the matrix
honest. An attack that works only when hand-fitted to one task would not
appear here.

Two tests guard the matrix. One checks that every planted text really lands
in a field the task's reference plan reads. The other checks that an agent
which ignores the planted text still solves the task, so a failed task under
attack is the agent's doing and not a broken case.

## Write the agent loop by hand

Inspect has stock agent loops. This project uses Inspect for models, logging
and running, and writes its own loop, because the safety work happens inside
the loop: a defence must see each outsider-written string before the model
does and each tool call before it runs.

The loop runs one tool call per step. On the first trial run a 7B model
batched several calls in one turn and filled the later ones with ids it had
not yet read. One call per step removed that failure without touching what is
being measured.

## Provenance is tracked whether or not anything uses it

Each tool declares which keys of its result are written by outsiders. The
loop splits every result along that line and keeps two lists per agent:
trusted text and outsider text. Every call is recorded with two facts: had
the agent read outsider text, and was the call's target named by a trusted
source.

Those facts cost nothing to record and pay for themselves three times: the
`taint` and `grounding` defences read them, the deterministic monitors read
them, and they are there in the records for any later analysis.

### The rule that had to be fixed

The first version of grounding trusted any id found in a field the shop
writes. That has a hole. An agent told by a review to refund order O-9999 can
look the order up, and the shop's own record echoes the id back in a trusted
field. The lookup launders the id.

The rule now: a record's fields count as trusted only if a trusted source led
the agent to the record. The scripted victim used in tests never did the
lookup, so the tests had passed. A regression test now plays the laundering
sequence.

This is the general shape of the problem with provenance defences, and the
reason the roadmap puts adaptive attacks first. The defence was written, it
passed its tests, and it was wrong.

### What grounding does not do

It checks where a target came from, not whether acting on it is right. If the
operator asks about customer C-103's orders, then C-103's orders are in
trusted data, and an injection asking to reroute one of them is not stopped.
The results break this out.

## In the fleet, trust belongs to text and not to agents

A request from one agent to another is not trusted because a colleague sent
it. It is trusted if the sender had read nothing written by an outsider when
it wrote the request. Otherwise the receiving agent treats the request itself
as outsider text. Answers are handled the same way coming back.

The alternative, treating colleagues as trusted, is how a real fleet would
most likely be built, and it is exactly the path an injection takes: the
agent that reads the ticket cannot issue refunds, so it asks the agent that
can, and the request arrives looking like internal work.

Trusted text, on the other hand, is shared across the fleet. If the support
agent learned from a ticket's own order field that the ticket is about order
O-2002, the finance agent may act on O-2002.

## Monitors are scored offline

A monitor here never blocks anything. Runs are recorded with no defence, and
each monitor scores the recorded events afterwards. So one set of runs serves
every monitor, a monitor cannot change what the agents did, and adding a
monitor needs no new agent runs.

The price is that this measures detection, not the full protocol. What
happens after a flag (block, ask a person, hand the step to a trusted model)
changes what the agent does next, and that is not simulated. The assumption
made explicit in the results: a flagged event is audited by a person who
judges it correctly.

## Intervals resample tasks

There are 504 attack cases and 11 tasks. A model that mishandles one task
mishandles all 42 attacks planted in it. Treating the 504 as independent
would give intervals several times too narrow. Headline rates are therefore
bootstrapped by resampling whole tasks. Breakdowns within a run, where the
question is about that run's cases, use Wilson intervals.

## Probes are scored only on what they have not seen

Planted attacks share wording. A classifier can learn "contains the word
INFORMATION in angle brackets" and score perfectly on a random split. So no
probe is ever scored on a random split:

- hold out one wording, and test on a wording never trained on;
- hold out one goal, and test on a demand never trained on;
- hold out one task, and test in a conversation never trained on.

Harmless notes are spread across the same folds, so a held-out fold also
contains notes the probe never saw.

Every probe is compared with a baseline that reads only the planted text
(character n-grams). If the baseline matches the probe, the probe has found
nothing the text did not already say, and the result is reported that way.

## Activations come from the runs that produced the behaviour

The probing dataset is not a set of synthetic prompts. Each example is a
moment in a recorded benchmark run: the exact prompt the model saw after
reading the planted text, with the label of what that same model then did.
The activations are read from the same weights, at the same quantisation,
that produced the behaviour.

## Steering is evaluated as a defence

An edit to the residual stream is passed to the model provider as an
argument. The benchmark then runs unchanged, and the steered model gets a row
in the same table as every other defence: attack success, tasks solved, on
the same cases. The direction is fitted on cases outside the evaluation
subset, and a random direction of the same length is run as the control.
