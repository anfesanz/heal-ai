PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
STREAMLIT := $(VENV)/bin/streamlit
PORT ?= 8501
N_BOOTSTRAP ?= 300
PIP_FLAGS ?= --disable-pip-version-check --quiet --no-cache-dir
MPLCONFIGDIR ?= $(CURDIR)/.cache/matplotlib
MPLBACKEND ?= Agg
export MPLCONFIGDIR
export MPLBACKEND

.PHONY: help all all-download venv install check analysis analysis-download dashboard clean-outputs clean-cache

help:
	@echo "HEAL-AI Eval commands"
	@echo ""
	@echo "  make all               Install, check, and run analysis"
	@echo "  make all-download      Install, check, download data, and run analysis"
	@echo "  make venv              Create local virtual environment"
	@echo "  make install           Install Python dependencies"
	@echo "  make check             Compile Python files"
	@echo "  make analysis          Run analysis using local/processed data"
	@echo "  make analysis-download Download public NHANES files and run analysis"
	@echo "  make dashboard         Start Streamlit dashboard on first free port"
	@echo "  make clean-outputs     Remove generated figures, tables, models"
	@echo "  make clean-cache       Remove Python cache files"
	@echo ""
	@echo "Variables:"
	@echo "  PORT=8502              Override dashboard port"
	@echo "  N_BOOTSTRAP=500        Override bootstrap resamples"
	@echo "  PIP_FLAGS=             Show full pip install output"

all: install check analysis

all-download: install check analysis-download

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install $(PIP_FLAGS) -r requirements.txt

check:
	$(PY) -m py_compile src/*.py app/streamlit_app.py

analysis:
	mkdir -p $(MPLCONFIGDIR)
	$(PY) -m src.run_analysis --n-bootstrap $(N_BOOTSTRAP)

analysis-download:
	mkdir -p $(MPLCONFIGDIR)
	$(PY) -m src.run_analysis --download --n-bootstrap $(N_BOOTSTRAP)

dashboard:
	mkdir -p .cache
	$(PY) -m src.find_free_port --start $(PORT) > .cache/dashboard_port
	@echo "Starting dashboard at http://localhost:$$(cat .cache/dashboard_port)"
	$(STREAMLIT) run app/streamlit_app.py --server.port $$(cat .cache/dashboard_port)

clean-outputs:
	find outputs/figures outputs/tables outputs/models -type f ! -name ".gitkeep" -delete

clean-cache:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +
