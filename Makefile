# The experiments behind docs/results/. Each target writes per-case records
# under results/; `make report` rebuilds every table and figure from them.

PY      ?= .venv/bin/python
CORDON  ?= .venv/bin/cordon
AGENT   ?= mlx/mlx-community/Qwen2.5-7B-Instruct-4bit
BIG     ?= mlx/mlx-community/Qwen2.5-14B-Instruct-4bit
TRUSTED ?= mlx/mlx-community/Qwen2.5-1.5B-Instruct-4bit
SLUG    ?= qwen2.5-7b-instruct-4bit

.PHONY: install test lint report bench defences fleet monitors probes

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
	$(CORDON) bench --model $(AGENT) --defence none --suite clean,attacks,benign --log-dir logs/bench-7b

defences:
	$(CORDON) bench --model $(AGENT) --cases 168 --log-dir logs/bench-7b \
		--defence grounding,spotlight,datamark,reminder,sanitise,least_privilege,taint
	$(CORDON) bench --model $(AGENT) --trusted $(TRUSTED) --cases 168 --log-dir logs/bench-7b \
		--defence detector --suite clean,attacks,benign

# Track 2. One agent against a fleet, on the stratified 72-case subset.
fleet:
	$(CORDON) fleet --model $(BIG) --arm solo,fleet --suite clean,injection,insider --cases 72 --log-dir logs/fleet-14b
	$(CORDON) fleet --model $(BIG) --arm fleet --share-request --suite clean,injection --cases 72 --log-dir logs/fleet-14b

monitors:
	$(CORDON) monitor --logs logs/fleet-14b --set fleet-14b --name ungrounded --kind ungrounded
	$(CORDON) monitor --logs logs/fleet-14b --set fleet-14b --name tainted --kind tainted
	$(CORDON) monitor --logs logs/fleet-14b --set fleet-14b --name llm-1.5b --kind llm --model $(TRUSTED)
	$(CORDON) monitor --logs logs/fleet-14b --set fleet-14b --name llm-1.5b+reads --kind llm+reads --model $(TRUSTED)

# Track 3. Needs `make bench` first: probes are fitted on its recorded runs.
probes:
	$(CORDON) interp collect --model $(AGENT) --logs logs/bench-7b --out artifacts/interp/$(SLUG)
	$(CORDON) interp probe --dir artifacts/interp/$(SLUG)
