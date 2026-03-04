# Contributing

## Development Setup

```bash
git clone https://github.com/your-org/industrial-vision-sorting.git
cd industrial-vision-sorting
pip install -r requirements.txt
```

## Running Tests

No GPU or hardware required for the unit test suite:

```bash
python3 test_suite.py
```

54 tests cover: models, double_buffer, classical_cv, tracker, predictor,
scheduler, latency_monitor, watchdog, ejector (mock GPIO), camera (mock).

## Code Style

- Follow existing code style — section headers, spacing, docstrings
- All public methods must have docstrings with Args/Returns
- Use `logging` — no `print()` in production code
- New GPU kernels go in `inference/` as `cupy.RawKernel`
- New hardware drivers must implement the relevant interface in `core/interfaces.py`

## Adding a Hardware Driver

1. Add abstract method to `core/interfaces.py` if needed
2. Implement in the relevant `hardware/` or `capture/` file
3. Add mock in `test_suite.py` and write tests
4. Wire up in `main.py`

## Pull Request Checklist

- [ ] `python3 test_suite.py` passes (54/54)
- [ ] No `print()` statements — use `logger`
- [ ] No imports inside functions or methods
- [ ] No per-frame memory allocations (no `cudaMalloc`, `np.zeros`, etc. in hot paths)
- [ ] Docstrings on all public methods
- [ ] `config.yaml` updated if new parameters added
