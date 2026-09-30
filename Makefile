# The experiments behind docs/results/. Each target writes per-case records
# under results/; `make report` rebuilds every table and figure from them.

PY      ?= .venv/bin/python
CORDON  ?= .venv/bin/cordon
AGENT   ?= mlx/mlx-community/Qwen2.5-7B-Instruct-4bit
BIG     ?= mlx/mlx-community/Qwen2.5-14B-Instruct-4bit
TRUSTED ?= mlx/mlx-community/Qwen2.5-1.5B-Instruct-4bit
SLUG    ?= qwen2.5-7b-instruct-4bit

.PHONY: install test lint report bench defences fleet monitors probes steering

install:
	python3.12 -m venv .venv
	$(PY) -m pip install -e ".[dev,mlx,interp]"

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

report:
	$(CORDON) report

# Track 1. The baseline runs every case; each defence runs a fixed 168-case subset.
bench:
	$(CORDON) bench --model $(AGENT) --defence none --suite clean,attacks,benign,nearmiss --log-dir logs/bench-7b

defences:
	$(CORDON) bench --model $(AGENT) --cases 168 --log-dir logs/bench-7b \
		--defence grounding,spotlight,datamark,reminder,sanitise,least_privilege,taint
	$(CORDON) bench --model $(AGENT) --trusted $(TRUSTED) --cases 168 --log-dir logs/bench-7b \
		--defence detector --suite clean,attacks,nearmiss

# Track 2. One agent against a fleet, on the stratified 72-case subset.
fleet:
	$(CORDON) fleet --model $(BIG) --arm solo,fleet --suite clean,injection,insider --cases 72 --log-dir logs/fleet-14b
	$(CORDON) fleet --model $(BIG) --arm fleet --defence grounding --suite clean,injection --cases 72 --log-dir logs/fleet-14b-defended
	$(CORDON) fleet --model $(BIG) --arm fleet --share-request --suite clean,injection --cases 72 --log-dir logs/fleet-14b-shared

# Monitors score recorded runs; they need `make bench` and `make fleet` first.
monitors:
	for kind in ungrounded tainted; do \
		$(CORDON) monitor --logs logs/bench-7b --set bench-7b --name $$kind --kind $$kind; \
		$(CORDON) monitor --logs logs/fleet-14b --set fleet-14b-injection --only clean,injection --name $$kind --kind $$kind; \
	done
	$(CORDON) monitor --logs logs/bench-7b --set bench-7b --name llm-1.5b --kind llm --model $(TRUSTED)
	$(CORDON) monitor --logs logs/bench-7b --set bench-7b --name llm-1.5b+reads --kind llm+reads --model $(TRUSTED)
	$(CORDON) monitor --logs logs/bench-7b --set bench-7b --name llm-7b --kind llm --model $(AGENT)
	$(CORDON) monitor --logs logs/bench-7b --set bench-7b --name llm-7b+reads --kind llm+reads --model $(AGENT)
	for set in injection insider; do \
		$(CORDON) monitor --logs logs/fleet-14b --set fleet-14b-$$set --only clean,$$set --name llm-7b --kind llm --model $(AGENT); \
		$(CORDON) monitor --logs logs/fleet-14b --set fleet-14b-$$set --only clean,$$set --name llm-7b+reads --kind llm+reads --model $(AGENT); \
	done

# Track 3. Needs `make bench` first: probes are fitted on its recorded runs.
ACTS = artifacts/interp/$(SLUG)
STEER = $(CORDON) bench --model $(AGENT) --defence none --suite clean,attacks --cases 168 --log-dir logs/bench-7b-steer

probes:
	$(CORDON) interp collect --model $(AGENT) --logs logs/bench-7b --out $(ACTS)
	$(CORDON) interp probe --dir $(ACTS)

# Steering runs go to their own log directory so that the probes and monitors,
# which read every undefended log in logs/bench-7b, never see a steered run.
steering:
	$(CORDON) interp direction --dir $(ACTS) --label attempted --layer 17
	$(STEER) --label ablate-comply --model-arg steer=$(ACTS)/attempted-L17.npy --model-arg steer_mode=ablate
	$(STEER) --label ablate-random --model-arg steer=$(ACTS)/attempted-L17-random.npy --model-arg steer_mode=ablate
	$(STEER) --label steer-against-comply --model-arg steer=$(ACTS)/attempted-L17.npy \
		--model-arg steer_mode=add --model-arg steer_layer=16 --model-arg steer_alpha=-1.0
	$(STEER) --label steer-random --model-arg steer=$(ACTS)/attempted-L17-random.npy \
		--model-arg steer_mode=add --model-arg steer_layer=16 --model-arg steer_alpha=-1.0
