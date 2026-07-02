COMPREHENSIVE CODEBASE ANALYSIS: GSGP_NN
Repository root: /home/gallobota/GSGP_NN  
Git history (4 commits): 631fb77 Initial commit, 71d249e Initial commit, 5ec9459 Estructurado la organizacion del proyecto, 13f0023 v1 funcional (HEAD)  
Data directory: Does not exist (empty) — no .txt dataset files found.  
Results directory: results/ exists but is empty.  
1. COMPLETE FILE INVENTORY
#	File Path	Lines	Type
1	/home/gallobota/GSGP_NN/main.py	0	Empty entry point
2	/home/gallobota/GSGP_NN/config/config.yaml	9	Hyperparameter config
3	/home/gallobota/GSGP_NN/enviroment.yml	0	Empty dependency file
4	/home/gallobota/GSGP_NN/README.md	1	Empty stub (just title)
5	/home/gallobota/GSGP_NN/.gitignore	221	Standard Python gitignore
6	/home/gallobota/GSGP_NN/legacy/GSGP_NN.py	1177	Legacy monolithic implementation
7	/home/gallobota/GSGP_NN/legacy/Demostracion_GSGPNN.tex	796	LaTeX mathematical derivation
8	/home/gallobota/GSGP_NN/scripts/run_experiment.py	372	New experiment runner CLI
9	/home/gallobota/GSGP_NN/src/__init__.py	9	Package init, exports GSGPNN/GSGPNNLayer
10	/home/gallobota/GSGP_NN/src/data/__init__.py	6	Data sub-package exports
11	/home/gallobota/GSGP_NN/src/data/dataset.py	111	RegressionDataset class
12	/home/gallobota/GSGP_NN/src/data/loader.py	149	Semantic data preparation + config loading
13	/home/gallobota/GSGP_NN/src/evolution/__init__.py	7	Evolution sub-package exports
14	/home/gallobota/GSGP_NN/src/evolution/nodes.py	130	GPNode tree node
15	/home/gallobota/GSGP_NN/src/evolution/semantics.py	99	SemanticEvaluator
16	/home/gallobota/GSGP_NN/src/evolution/generator.py	109	GPGenerator
17	/home/gallobota/GSGP_NN/src/models/__init__.py	5	Models sub-package exports
18	/home/gallobota/GSGP_NN/src/models/gsgp_nn.py	196	GSGPNN + GSGPNNLayer torch modules
19	/home/gallobota/GSGP_NN/src/training/__init__.py	7	Training exports
20	/home/gallobota/GSGP_NN/src/training/trainer.py	167	Training loop
21	/home/gallobota/GSGP_NN/src/training/callbacks.py	87	TrainingLogger + ModelCheckpoint
22	/home/gallobota/GSGP_NN/src/training/early_stopping.py	87	EarlyStopping callback
23	/home/gallobota/GSGP_NN/src/utils/__init__.py	13	Utils exports
24	/home/gallobota/GSGP_NN/src/utils/device.py	65	get_device + estimate_memory_usage_mb
25	/home/gallobota/GSGP_NN/src/utils/metrics.py	90	compute_metrics + compute_metrics_original_scale
26	/home/gallobota/GSGP_NN/src/utils/seed.py	47	set_seed
27	/home/gallobota/GSGP_NN/tests/test_evolution.py	128	GPNode + GPGenerator tests
28	/home/gallobota/GSGP_NN/tests/test_model.py	133	GSGPNNLayer + GSGPNN tests
29	/home/gallobota/GSGP_NN/tests/test_training.py	121	Trainer + EarlyStopping tests
30	/home/gallobota/GSGP_NN/tests/test_utils.py	0	Empty test file
31	/home/gallobota/GSGP_NN/notebooks/run_tests_and_regression.ipynb	849	Test execution + regression demo
32	/home/gallobota/GSGP_NN/explicaciones/guia.md	61	Spanish code guide
33	/home/gallobota/GSGP_NN/explicaciones/guia_codigo.md	327	Bilingual code guide
34	/home/gallobota/GSGP_NN/explicaciones/estructura_pytorch.md	62	Project structure proposal
35	/home/gallobota/GSGP_NN/artifacts/Implementation Plan.md	572	Refactoring plan document
2. COMPLETE SOURCE CODE DETAILS
2.1 src/__init__.py (lines 1-9)
- Exports: GSGPNN, GSGPNNLayer
- Docstring: Brief project description
- No issues
2.2 src/data/dataset.py (lines 1-111)
- Class: RegressionDataset
- __init__(self, path: str, test_size: float = 0.3, seed: int = 0) — initializes scalers, calls _load
- _load(self, path: str, test_size: float, seed: int) — validates file existence with Path(path).exists(), loads via np.loadtxt, fits StandardScaler, stores y_mean/y_std for inverse transform, performs train_test_split
- get_tensors(self, device: torch.device) — returns 4 tensors (X_train, y_train, X_test, y_test) as float32 on device
- Properties: num_features, num_train_patterns
- Type hints: Yes, via from __future__ import annotations
- Docstrings: Detailed Google-style
- Error handling: FileNotFoundError raised for missing dataset
- Strengths: Clean separation of concerns, stores scaler statistics
- Issues:
- Line 52-56: File not found error but no fallback to synthetic data (unlike legacy preparar_datos_concrete_regresion)
- No __repr__ or __len__ methods (not a true PyTorch Dataset subclass)
- No validation of data content (e.g., empty arrays, NaN values)
2.3 src/data/loader.py (lines 1-149)
- Functions:
- load_config(config_path: str = "config/config.yaml") — YAML config loading
- prepare_semantic_data(X, num_nodes, num_layers, population_size, device) — Main bridge between GP (CPU) and NN (GPU):
- Creates GPGenerator with ["+", "-", "*", "/"], max_depth=10
- Generates initial + auxiliary populations of size population_size
- Creates SemanticEvaluator on target device
- Evaluates both populations → semantic matrices
- Builds parent semantics (cycling through initial pop)
- Builds initial semantics (cycling through combined pops)
- Builds route semantics (RT1/RT2 pairs per layer per node, cycling through auxiliary pop)
- Returns dict with all tensors + evaluator + populations
- Strengths: Clear CPU→GPU handoff, reuses evaluator for test data later
- Issues:
- Line 97-99: parent_rows built via Python list comprehension loop over range(num_nodes) — could be done with tensor indexing
- Line 106-110: Same pattern for init_rows
- Line 113-132: Triple-nested loops for route semantics (for _k, for _n, plus manual index tracking) — verbose but necessary for the cycling logic
- No validation that population_size >= 2*N*K (see line 76 of run_experiment.py where int(2*N*K) is computed; but the indexing can cycle safely)
2.4 src/evolution/nodes.py (lines 1-130)
- Class: GPNode with __slots__ = ("value", "is_terminal", "arity", "children")
- __init__(self, value, is_terminal=False, arity=0)
- add_child(self, child) — appends if slots available
- is_complete(self) — checks if len(children) == arity
- evaluate(self, variables: dict[str, float]) -> float:
- Terminals: returns float value or variables.get(self.value, 0.0)
- Functions: recursive evaluation with protected division (abs(b) < 1e-10 → 1.0), result clamped to [-1e6, 1e6]
- IMPORTANT BUG (line 77): math.isfinite — This is wrong! The correct function is math.isfinite() does not exist in Python's math module. The correct function is math.isfinite() — actually, checking: Python's math module has isfinite() since Python 3.2. Let me verify: yes, math.isfinite(x) exists in Python 3.2+. So this is fine.
- Actually, let me double check: math.isfinite was added in Python 3.2. So this is correct for Python 3.10+.
- deep_copy(self) — recursive deep copy
- __repr__(self) — string representation
- Issues:
- Line 77: if not math.isfinite(val) — uses math.isfinite which exists, but the comment says # TODO: Implementar la división protegida... with a C-style comment block (lines 90-96) that references an unrelated implementation detail
- Line 90-96: Dead code / TODO comment block referencing C code about "pushGenes" and "sqrtf" — left over from some reference implementation
- Line 98-99: Protected division returns 1.0 on near-zero, matching legacy. However, the legacy GSGP_NN.py line 157 has the same logic.
- Line 108: Bare except Exception: return 0.0 — broad exception handling that could mask bugs
2.5 src/evolution/generator.py (lines 1-109)
- Module constant: _OPERATORS = {"+": 2, "-": 2, "*": 2, "/": 2}
- Class: GPGenerator
- __init__(self, function_set, terminal_set, max_depth=3) — validates function set against _OPERATORS
- _random_constant() — static method, returns round(random.uniform(-1.0, 1.0), 3)
- _create_random_node(self, force_terminal=False) — 50% terminal/function split, terminals are 70% variable / 30% constant
- generate_tree_grow(self, max_depth) — recursive Grow method
- generate_population(self, size) — cycles depths through [1, 2, 2, 3, 3, 3]
- Issues:
- Line 42: # TODO: Revisar si las constantes tienen que estar restringidas... — unresolved TODO
- Line 47: # TODO: Revisar la proporción 70 % variable, 30 % constante — unresolved TODO
- Line 66: # TODO: Revisar el metodo para generar el arbol (Grow) — unresolved TODO
- Line 89: # TODO: Revisar la lógica de generacion de poblacion y profundidad — unresolved TODO  
- Line 102-103: Large TODO about replacing depth schedule with hyperparameter and using Ramped Half-and-Half
- No Full/Half method — only Grow method is implemented, which is less diverse
- Uses random module directly (not np.random or torch.Generator), so set_seed() must also seed random
2.6 src/evolution/semantics.py (lines 1-99)
- Class: SemanticEvaluator
- __init__(self, variable_names: list[str], device: torch.device)
- evaluate_individual(self, tree: GPNode, X: np.ndarray) -> torch.Tensor:
- Iterates over M rows with Python for i in range(M) loop — this is the main CPU bottleneck (sequential tree evaluation per sample)
- Builds variables dict per row
- Calls tree.evaluate(variables), stores in np.zeros(M, dtype=np.float64)
- Normalizes via _normalize_semantics → converts to torch.tensor(..., dtype=torch.float32, device=self.device)
- create_semantic_matrix(self, population, X) — list comprehension calling evaluate_individual for each tree, then torch.stack
- _normalize_semantics(semantics) — static method: clip to -1e6, 1e6, z-score normalize, clip to -5, 5; if std < 1e-10, returns deterministic ramp mean + 1e-8 * arange(len)
- Performance Issue (lines 44-54): The per-sample Python loop is unavoidable but is the sequential bottleneck. Each of population_size trees must evaluate all M samples one-by-one. For pop_size=400, M=500, that is 200,000 tree evaluations. This is the one part that CANNOT be GPU-accelerated because GP trees are recursive Python objects.
- Line 43: semantics = np.zeros(M, dtype=np.float64) — uses float64, but final tensor is float32. Minor precision loss.
- Line 53: Bare except Exception: semantics[i] = 0.0 — masks errors.
2.7 src/models/gsgp_nn.py (lines 1-196)
- Class: GSGPNNLayer(nn.Module)
- __init__(self, num_nodes: int):
- self.alpha = nn.Parameter(torch.empty(num_nodes + 1, num_nodes)) — shape (N+1, N) where row 0 is parent weight, rows 1..N are prev-layer weights
- self.ms = nn.Parameter(torch.empty(num_nodes)) — mutation scale
- self.beta = nn.Parameter(torch.empty(num_nodes)) — mutation coefficient
- Uses kaiming_normal_ for alpha, normal_(std=0.1) for ms/beta
- forward(self, prev_semantics, parent_semantics, rt1, rt2):
- Inherited term: parent_contrib = alpha[0].unsqueeze(1) * parent_semantics  → (N, M)
- prev_contrib = torch.mm(alpha[1:].t(), prev_semantics)  → (N, M)
- H = parent_contrib + prev_contrib
- Denominator: d = alpha.abs().sum(dim=0).unsqueeze(1).clamp(min=1e-12) → (N, 1)
- inherited = H / d
- Mutation: mutation = beta.unsqueeze(1) * torch.tanh(ms.unsqueeze(1) * (rt1 - rt2))
- Returns inherited + mutation
- Class: GSGPNN(nn.Module)
- __init__(self, num_nodes, num_layers):
- self.layers = nn.ModuleList([GSGPNNLayer(num_nodes) for _ in range(num_layers)])
- self.output_layer = nn.Linear(num_nodes, 1)
- forward(self, initial_semantics, parent_semantics, route_semantics):
- Iterates through layers, feeding current output as prev_semantics
- Final output: self.output_layer(current.t()).squeeze(1) → (M,)
- count_parameters(self) — returns total trainable params
- summary(self) — human-readable model architecture string
- Strengths:
- Fully batched tensor operations — NO Python loops in forward pass
- Correct handling of denominator normalization
- Proper use of nn.ModuleList
- Issues:
- Line 109: d = self.alpha.abs().sum(dim=0).unsqueeze(1) — sum(dim=0) sums the (N+1) rows into (N,) then unsqueezes to (N, 1). This is equivalent to the denominator Σ_j |α[j,n]|. CORRECT.
- Line 62: self.alpha has shape (N+1, N) — row 0 is for parent connection, rows 1..N are for previous-layer connections. But the legacy code uses alpha[(0,n,k)] for the parent of node n at layer k, and alpha[(i,n,k)] for the previous-layer node i connecting to node n. In the tensor version, alpha[0, n] is the weight from parent to node n, and alpha[i, n] (i=1..N) is the weight from previous-layer node i to node n. This transposition is correct.
- Missing bias in output layer: nn.Linear(num_nodes, 1) includes bias by default (bias=True). The legacy code has self.b_output which starts at 0.0 and is trainable. This is consistent.
2.8 src/training/trainer.py (lines 1-167)
- Class: Trainer
- __init__(self, model, learning_rate=0.005, weight_decay=0.0, grad_clip_norm=5.0, early_stopping=None, logger=None, checkpoint=None):
- Uses torch.optim.Adam (replacing manual AdamOptimizerCorregido)
- Accepts callbacks via dependency injection
- fit(self, semantic_data, y_train, epochs=200):
- Standard loop: forward → F.mse_loss → optimizer.zero_grad() → loss.backward() → clip_grad_norm_ → optimizer.step()
- Replaces ~150 lines of manual gradient computation from legacy
- Reports timing and history
- Returns dict with loss_history, total_time, epoch_times, epochs_trained, final_loss, converged
- predict(self, semantic_data) — @torch.no_grad() inference
- Strengths:
- Clean, standard PyTorch pattern
- Proper gradient clipping via clip_grad_norm_ (replaces per-parameter np.clip)
- Good use of dependency injection for callbacks
- Issues:
- Line 54: weight_decay=weight_decay is passed to Adam but is always 0.0 (no L2 regularization in config)
- Line 128: self.early_stopping(loss_val) — calls EarlyStopping.__call__ which returns bool. Works correctly.
- No validation loop — evaluation is done externally after training
2.9 src/training/callbacks.py (lines 1-87)
- Class: TrainingLogger
- __init__(self, print_every=1, window_size=25)
- on_epoch_start(self) — records start time
- on_epoch_end(self, epoch, loss) — prints formatted log line with improvement percentage
- Issue: on_epoch_start() is never called in Trainer.fit() — epoch time is measured directly in the fit method, not via this callback. The logger's _epoch_start is never set, so elapsed = time.time() - self._epoch_start uses the initial value 0.0, resulting in huge epoch times (as seen in the notebook: [1782978706.626s]).
- Class: ModelCheckpoint
- __init__(self, save_dir="results", filename="best_model.pt")
- on_epoch_end(self, loss, model) — saves model.state_dict() when loss improves
- Issue: Saves to results/best_model.pt, but each training run overwrites the same file. No seed/run/dataset in filename.
2.10 src/training/early_stopping.py (lines 1-87)
- Class: EarlyStopping — clean implementation with __call__ pattern
- Implements warmup period, patience, min_delta
- reset(), state_dict(), load_state_dict() for checkpointing
- Strengths: Full API, serializable state
- Issue: state_dict() and load_state_dict() are defined but never used (Trainer has no checkpoint save/restore for early stopping state)
2.11 src/utils/device.py (lines 1-65)
- get_device(prefer_cuda=True) — detects CUDA, prints GPU info
- estimate_memory_usage_mb(num_nodes, num_layers, num_patterns) — rough memory estimate
- Issues:
- Line 9-10 comment says Use CUDA when available but the docstring has typo prefer_cuda
- Line 58: total bytes calculation may underestimate activation memory in autograd graph
- estimate_memory_usage_mb is called in run_experiment.py line 79 but is a rough estimate
2.12 src/utils/metrics.py (lines 1-90)
- compute_metrics(y_true, y_pred) — GPU-native MAE, MSE, RMSE, R²
- compute_metrics_original_scale(y_true_scaled, y_pred_scaled, y_mean, y_std) — manual inverse transform
- Includes if __name__ == "__main__": test block for self-testing
- Issues:
- Line 28: ss_tot = ((y_true - y_true.mean()) ** 2).sum() — uses biased variance formula. R² is still correct but numerically slightly different from sklearn.
2.13 src/utils/seed.py (lines 1-47)
- set_seed(seed) — seeds random, numpy, torch, CUDA, sets deterministic flags
- Issue: Line 32: torch.use_deterministic_algorithms(True) — this can cause runtime errors with some PyTorch operations (e.g., torch.nn.functional.interpolate). Wrapped in try/except, but could silently fail.
- Includes self-test in if __name__ == "__main__":
3. CONFIGURATION
config/config.yaml (lines 1-9)
num_corridas: 30
configuraciones:
  - { path: "yatch.txt", K: 2, N: 200, lr: 0.005, epochs: 200 }
  - { path: "eheating.txt", K: 2, N: 200, lr: 0.005, epochs: 200 }
  - { path: "ecooling.txt", K: 2, N: 200, lr: 0.005, epochs: 200 }
  - { path: "housing.txt", K: 2, N: 50, lr: 0.005, epochs: 200 }
  - { path: "ConcreteData.txt", K: 2, N: 200, lr: 0.005, epochs: 200 }
  - { path: "tower.txt", K: 2, N: 200, lr: 0.005, epochs: 200 }
- Issues:
- path values reference .txt files that are NOT present in the data/ directory (it does not exist)
- Typo: yatch.txt (should be yacht.txt)
- No weight_decay, grad_clip, device, or mixed-precision fields
- housing.txt uses N=50 instead of N=200 — inconsistent with others
4. DEPENDENCIES
- enviroment.yml — EMPTY (0 bytes)
- requirements.txt — DOES NOT EXIST
- setup.py — DOES NOT EXIST
- pyproject.toml — DOES NOT EXIST
- No explicit dependency specification at all
From import analysis, the required packages are:
- numpy
- pytorch (torch)
- scikit-learn (sklearn)
- pandas
- matplotlib
- pyyaml (yaml)
- pytest
- jupyter (for the notebook)
The notebook output shows the conda env is named gsgp_nn with Python 3.10.20, pytest 9.1.1.
5. TEST ANALYSIS
5.1 tests/test_evolution.py (15 tests, all PASS)
- TestGPNode (11 tests): terminal constant, terminal variable, missing variable, addition, subtraction, multiplication, protected division normal, protected division by zero, result clamped, deep copy, repr
- TestGPGenerator (4 tests): generate single tree, tree evaluates, generate population, population diversity
- Coverage gaps: No test for GPNode with nested deep trees (>2 depth), no test for SemanticEvaluator at all, no test for normalization edge cases, no test for large constant generation
5.2 tests/test_model.py (10 tests, all PASS)
- TestGSGPNNLayer (4 tests): output shape, output finite, gradients flow, parameter count
- TestGSGPNN (6 tests): output shape, output finite, gradients flow through all layers, count parameters, summary returns string, deterministic output
- Coverage gaps: 
- No test with K=0 or N=0 edge cases
- No test with requires_grad=False inputs
- No test for the denominator clamping path (when sum of abs alphas < 1e-12)
- No test for numerical equivalence with legacy code
- No CUDA device tests (all use CPU fixture)
5.3 tests/test_training.py (8 tests, all PASS)
- TestTrainer (4 tests): fit runs, loss decreases, predict shape, early stopping triggers
- TestEarlyStopping (4 tests): no stop during warmup, stops after patience, resets on improvement, reset method
- Coverage gaps:
- No test for logger
- No test for checkpoint
- No test for gradient clipping behavior
- No test for compute_metrics or compute_metrics_original_scale
- No test for RegressionDataset
- No test for prepare_semantic_data
- No test with NaN/Inf in loss
- No integration test comparing legacy vs new outputs
5.4 tests/test_utils.py — EMPTY (0 lines)
Overall test coverage: 33 passing, many critical gaps:
- No SemanticEvaluator tests
- No RegressionDataset tests
- No prepare_semantic_data tests
- No load_config tests
- No metrics tests
- No device/seed tests
- No integration/regression tests against legacy
6. PERFORMANCE ANALYSIS
6.1 Sequential Bottlenecks (CPU-bound)
Location	File:Line	Description	Impact
Tree evaluation	semantics.py:45-54	for i in range(M) loop over all samples per tree	O(pop_size × M) Python iterations
Population evaluation	semantics.py:75	List comprehension calling evaluate_individual per tree	O(pop_size) sequential calls
Tree generation	generator.py:105-108	for i in range(size) creating trees	O(pop_size) recursive tree builds
Test semantics	run_experiment.py:170-171	Same per-sample loop repeated for test data	Full re-evaluation of all pop trees on test set
These are inherent to the GP approach and cannot be fully vectorized. The GP trees are recursive Python objects and cannot be evaluated as batched tensor operations. However:
6.2 Potential Optimizations for GP Evaluation
1. Numba JIT compilation for evaluate_individual loop over samples
2. Tree flattening / stack-based evaluation to reduce Python recursion overhead
3. Multiprocessing via concurrent.futures for population evaluation (embarrassingly parallel)
4. Cython for tree evaluation inner loop
6.3 GPU-accelerated Components (Well-Optimized)
Component	File	Method	Status
Forward pass	gsgp_nn.py:76-119	Batched matrix ops	Optimal
Layer loop	gsgp_nn.py:165-171	K layer iterations	Minimized
Backward pass	trainer.py:104-112	loss.backward() + optimizer.step()	Optimal (autograd)
6.4 GPU/CPU Data Transfer
- One-time transfer at initialization: prepare_semantic_data outputs tensors directly on device. Good.
- Test evaluation: _prepare_test_semantics (run_experiment.py:154-199) re-evaluates GP on CPU then transfers to GPU. This is the only significant transfer.
- Metrics: compute_metrics uses torch.no_grad() and .item() to return Python floats — no large transfers.
- No batch training: Full-batch gradient descent (M samples processed at once). For large M (e.g., >10,000), memory could be an issue. Consider mini-batch training.
6.5 Memory Analysis
From device.py:30-59:
- Parameters: (N+1)*N + 2*N per layer × K + N + 1 for output
- Semantic tensors: (2 + 2*K) * N * M floats
- For N=200, K=2, M=500: ~2.3 MB model params + 1.2 MB tensors + 2 MB gradients ≈ 5.5 MB total
- Well within GPU memory constraints
6.6 Redundant Computations
1. Normalization in evaluate_individual (semantics.py:56): _normalize_semantics is called per-tree. It computes mean/std of the M-length vector. This is necessary but each call does O(M) work.
2. Test semantics regeneration (run_experiment.py:154-199): Entirely duplicates the logic of prepare_semantic_data with the same indexing patterns. This is a code duplication issue.
3. Cyclic index arithmetic in prepare_semantic_data (loader.py:97-99, 106-110, 113-132) and _prepare_test_semantics (run_experiment.py:174-193): Repeated modulo indexing could be pre-computed.
7. NOTEBOOK ANALYSIS
File: notebooks/run_tests_and_regression.ipynb (849 lines)
Content Summary:
1. Cell 1 (exec_count=13): Runs all 33 pytest tests — ALL PASS
2. Cells 2-5 (exec_count=14-17): Full GSGP-NN training demo on sklearn Diabetes dataset:
- Loads Diabetes data (442 samples, 10 features)
- Uses N=10, K=3, pop_size=40
- Trains with lr=0.01, 200 epochs, early stopping (patience=20, warmup=10)
- BUG: TrainingLogger epoch times are absurd (1782978706.626s) because on_epoch_start() is never called
- Final loss: 0.435, Training time: 2.09s (10.4ms/epoch)
3. Cell 6 (exec_count=18): Evaluation plots loss curve (MSE=3337.58, R²=0.37)
4. Cells 7-9 (exec_count=19-20): SimpleNN baseline with 441 params
- Final loss: 0.124, better than GSGP-NN's 0.435
- Test MSE: 5270.82, R²: 0.005 — GSGP-NN outperforms SimpleNN on test set (R² 0.37 vs 0.005)
Issues:
- Line 42-46: history["loss_history"] not history["loss"] (corrected in code)
- The epoch time bug (on_epoch_start never called) makes timing data useless
- Uses sklearn's mean_squared_error and r2_score instead of the project's own compute_metrics
- The notebook demonstrates GSGP-NN significantly outperforms a standard MLP on the Diabetes dataset despite having fewer parameters
8. LEGACY CODE COMPARISON
File: legacy/GSGP_NN.py (1177 lines)
Architecture Differences:
Component	Legacy (CPU)	New (PyTorch+GPU)
Parameters	self.alpha[(i,n,k)] dict	layer.alpha[i,n] tensor
Forward	Per-node Python loop	Batched torch.mm
Backward	~150 lines manual np.dot	loss.backward() autograd
Optimizer	AdamOptimizerCorregido manual	torch.optim.Adam
Population size	int(2*N*K)	int(2*N*K)
Data loading	preparar_datos_concrete_regresion	RegressionDataset
Experiment runner	Inline if __name__=="__main__"	scripts/run_experiment.py CLI
Numerical Differences:
- Legacy uses np.float64 for intermediate computations; new uses torch.float32
- Legacy clips gradients per-parameter with np.clip(grad, -5, 5); new uses clip_grad_norm_(..., 5.0) (norm-based clipping)
- Legacy clips parameter values after update; new does NOT clip parameters (relies on gradient clipping + Adam)
- Potential numerical divergence: The absence of parameter clipping in the new version could lead to training instability not seen in legacy
Key Functions in Legacy Not Ported:
- generar_estadisticas_robustez — statistics summary (not needed, done in _print_summary)
- visualizar_resultados_robustez — matplotlib visualization (delegated to notebook/user)
- guardar_resultados — CSV/Text saving (done in _save_results)
9. SECURITY AUDIT
Hardcoded Values:
Location	Value	Severity
config/config.yaml:4-9	Dataset paths (no data present)	Medium — breaks immediately
config/config.yaml:1	num_corridas: 30	Low — configurable
loader.py:76	max_depth=10	Low — reasonable default
generator.py:45	random.uniform(-1.0, 1.0) constant range	Low
generator.py:52	0.7 variable vs constant ratio	Low
generator.py:102	Depth schedule [1,2,2,3,3,3]	Low
semantics.py:93	std < 1e-10 threshold	Low
gsgp_nn.py:110	1e-12 denominator clamp	Low
callbacks.py:48	1e-10 improvement threshold	Low
metrics.py:29	1e-12 R² denominator clamp	Low
Secrets/Passwords: NONE FOUND
Path Traversal Risks:
- dataset.py:52-55: Path(path).resolve() is used after confirmation — low risk
- loader.py:30-32: Config path validation — safe
- callbacks.py:71: Path(save_dir) / filename — safe
Input Validation Gaps:
- RegressionDataset._load(): No validation that data is finite or non-empty
- prepare_semantic_data(): No validation that population_size > 0
- Trainer.fit(): No validation that semantic_data has required keys
- GSGPNN.forward(): No shape validation on inputs
- SemanticEvaluator.evaluate_individual(): Bare except Exception: semantics[i] = 0.0 — could hide memory errors or type errors
General Security:
- No environment variables used
- No network calls
- No file writes outside results/ directory
- No user input in production code (except legacy's interactive menu)
- matplotlib interactive plotting may cause issues in headless environments (no plt.switch_backend('Agg'))
10. COMPLETE ISSUE REGISTER
Critical:
1. enviroment.yml is empty — no way to reproduce the environment
2. No data files present — data/ directory does not exist, config references files that are absent
3. TrainingLogger._epoch_start is never set — Trainer.fit() never calls logger.on_epoch_start(), so epoch times are meaningless (line 38 of callbacks.py reports time.time() - 0.0)
High:
4. tests/test_utils.py is empty — no tests for metrics, device, or seed utilities
5. No numerical equivalence test between legacy and PyTorch versions
6. GP tree evaluation is strictly CPU-bound with no parallelization — the O(pop_size × M) Python loop in semantics.py:45-54 is the ultimate bottleneck
7. No requirements.txt or setup.py — project cannot be installed via pip
8. main.py is empty — no entry point
Medium:
 9. Four unresolved TODOs in generator.py (lines 42, 47, 66, 89, 102)
10. Dead code comment block in nodes.py:90-96 referencing unrelated C implementation
11. Parameter clipping removed — legacy clipped alpha ∈ [-5,5], ms/beta ∈ [-2,2], w_output ∈ [-10,10] after each update; new version only uses gradient norm clipping
12. ModelCheckpoint overwrites same file — no run/seed/dataset differentiation
13. EarlyStopping.state_dict/load_state_dict are defined but never called
14. node.py:108 — bare except Exception: return 0.0 hides errors
15. semantics.py:53 — bare except Exception: semantics[i] = 0.0 hides errors
Low:
16. Typo in config: yatch.txt → yacht.txt
17. readme.md has only title — no documentation
18. metric.py:28 — biased variance for R² (minor numerical difference from sklearn)
19. seed.py:32 — torch.use_deterministic_algorithms(True) may fail with some ops
20. No mini-batch support — full-batch training only, may hit GPU memory for large datasets
21. loader.py — repeated cyclic indexing patterns could be pre-computed
22. run_experiment.py:154-199 — _prepare_test_semantics duplicates logic from prepare_semantic_data
11. STRENGTHS
 1. Excellent modularization — clean separation of concerns across evolution/, models/, training/, data/, utils/
 2. Full GPU parallelization — all neural computations are batched tensor operations
 3. Standard PyTorch patterns — nn.Module, loss.backward(), torch.optim.Adam replace fragile manual implementations
 4. Comprehensive type hints — all functions have type annotations (Python 3.10+ style with from __future__ import annotations)
 5. Detailed docstrings — Google-style with Args/Returns for all public APIs
 6. Strong numerical safeguards — protected division, clipping, epsilon values throughout
 7. Callback architecture — EarlyStopping, TrainingLogger, ModelCheckpoint follow clean separation of concerns
 8. Deterministic seeds — set_seed() covers Python, NumPy, PyTorch, CUDA
 9. Memory estimation — estimate_memory_usage_mb() helps predict GPU requirements
10. GPU info printing — device selection prints name, VRAM, CUDA capability
11. Bilingual documentation — Spanish guides for original developers, English for new contributors
12. Successful test suite — all 33 tests pass
13. Notebook demonstrates clear advantage — GSGP-NN (R²=0.37) significantly outperforms standard MLP (R²=0.005) on Diabetes dataset
14. Mathematical derivation in LaTeX — legacy/Demostracion_GSGPNN.tex provides complete formal derivation
15. Legacy code preserved — legacy/GSGP_NN.py enables comparison and regression testing
12. COMPLETE DELIVERABLES MAP
Every file has been read completely. The project is a well-structured PyTorch implementation of a novel GSGP-NN hybrid architecture. It is functionally complete but has critical gaps in environment specification, data files, documentation, and test coverage. The GPU-accelerated forward pass is the core achievement, replacing 700+ lines of legacy CPU-bound code with ~200 lines of efficient tensor operations. The main remaining performance bottleneck is the GP tree evaluation (CPU), which cannot be easily GPU-accelerated due to its recursive tree structure.