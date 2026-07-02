GSGP-NN: Refactorización Completa a PyTorch con CUDA
Contexto
El archivo monolítico 
GSGP_NN.py
 (1178 líneas, ~48KB) implementa un modelo híbrido de Programación Genética Geométrica Semántica con Redes Neuronales (GSGP-NN). Actualmente opera completamente en CPU con NumPy usando loops Python explícitos, lo que lo hace extremadamente lento para las configuraciones con N=200 nodos y K=2 capas.

Problemas del código actual
Todo en un solo archivo — imposible de mantener, testear o extender
Sin paralelización — loops for n in range(1, N+1) para N=200 nodos, ejecutados secuencialmente
Optimizador Adam manual — reimplementa algo que PyTorch ya tiene optimizado
Backpropagation manual — calcula gradientes a mano con np.dot, propenso a errores
Diccionarios Python como almacenamiento de parámetros — self.alpha[(i, n, k)] es 1000x más lento que un tensor
Nombres en español — dificulta colaboración y publicación
Ganancia esperada con PyTorch + CUDA
El forward pass actual itera K × N × N = 2×200×200 = 80,000 operaciones secuencialmente
Con tensores en GPU, esto se reduce a K multiplicaciones matriciales ejecutadas en paralelo en miles de CUDA cores
Estimación conservadora: 50-100x speedup en el entrenamiento
User Review Required
IMPORTANT

Cambio de nombres a inglés: Todas las funciones, clases, variables y docstrings se escribirán en inglés como solicitaste. Los comentarios explicativos críticos tendrán versión bilingüe donde sea útil.

IMPORTANT

Autograd reemplaza backpropagation manual: PyTorch calculará todos los gradientes automáticamente. Las ~150 líneas de calcular_todos_los_gradientes() y calcular_gradiente_alpha() se eliminan completamente. Esto es más preciso y mantenible.

WARNING

Cambio de API: El código legacy usa modelo.entrenar(X, y) y modelo.predecir(X). La nueva API será trainer.fit(model, train_loader, val_loader) y model(X_tensor). Si hay notebooks o scripts que dependen de la API vieja, necesitarán actualizarse.

Open Questions
IMPORTANT

1. Árboles GP vs compilación a funciones: Los árboles GP actualmente se evalúan nodo por nodo con recursión Python (NodoGP.evaluar()). Hay dos caminos:

Opción A (Recomendada): Compilar cada árbol GP a una función vectorizada de PyTorch al inicio, y luego almacenar solo su semántica como tensor. El árbol solo se necesita una vez.
Opción B: Mantener los árboles GP como están (en CPU) y solo mover la parte neural a GPU. Yo recomiendo Opción A porque la semántica GP es fija durante el entrenamiento y no necesita recomputarse.
IMPORTANT

2. Mixed Precision (float16): ¿Quieres activar torch.amp para mixed precision training? Puede duplicar la velocidad en GPUs con Tensor Cores (RTX 3060+, A100, etc.) con pérdida mínima de precisión. Recomiendo sí incluirlo como opción configurable.

IMPORTANT

3. Multi-GPU: ¿Piensas usar más de una GPU? Si es así puedo incluir torch.nn.DataParallel o DistributedDataParallel. Si solo tienes una GPU, no lo incluyo para no agregar complejidad innecesaria.

Proposed Changes
Estructura Final del Proyecto
text

GSGP_NN/
├── config/
│   └── config.yaml              # [EXISTING] Hiperparámetros por dataset
├── data/                         # [EXISTING] Archivos .txt de datasets
├── legacy/
│   └── GSGP_NN.py               # [EXISTING] Código original (referencia)
├── src/
│   ├── __init__.py               # [MODIFY] Exports principales
│   ├── data/
│   │   ├── __init__.py           # [MODIFY]
│   │   ├── dataset.py            # [NEW] RegressionDataset (torch Dataset)
│   │   └── loader.py             # [NEW] Funciones para crear DataLoaders
│   ├── evolution/
│   │   ├── __init__.py           # [MODIFY]
│   │   ├── nodes.py              # [NEW] GPNode tree structure
│   │   ├── generator.py          # [NEW] GPGenerator (tree generation)
│   │   └── semantics.py          # [NEW] SemanticEvaluator → tensor output
│   ├── models/
│   │   ├── __init__.py           # [MODIFY]
│   │   └── gsgp_nn.py            # [NEW] GSGPNN(nn.Module) — the core model
│   ├── training/                 # [FIX TYPO] renamed from "traininig"
│   │   ├── __init__.py           # [MODIFY]
│   │   ├── trainer.py            # [NEW] Training loop with CUDA
│   │   ├── early_stopping.py     # [NEW] EarlyStopping callback
│   │   └── callbacks.py          # [NEW] Logging, checkpointing
│   └── utils/
│       ├── __init__.py           # [MODIFY]
│       ├── metrics.py            # [NEW] Evaluation metrics
│       ├── device.py             # [NEW] CUDA device management
│       └── seed.py               # [NEW] Reproducibility utilities
├── tests/
│   ├── test_model.py             # [NEW] Unit tests for GSGPNN model
│   ├── test_evolution.py         # [NEW] Unit tests for GP components
│   └── test_training.py          # [NEW] Integration tests
├── scripts/
│   └── run_experiment.py         # [NEW] Main entry point (replaces __main__)
├── explicaciones/
│   └── guia_codigo.md            # [NEW] Guía para entender el código
├── results/                      # [NEW] Output directory for CSVs, plots
├── enviroment.yml                # [MODIFY] Add PyTorch + CUDA dependencies
└── README.md                     # [MODIFY] Project documentation
Component: src/evolution/ — GP Tree Engine (CPU)
This component runs on CPU. GP trees are evaluated once at initialization to produce semantic tensors that are then moved to GPU.

[NEW] 
nodes.py
Extracted from legacy lines 120-177. The NodoGP class becomes GPNode:

python

class GPNode:
    """A node in a Genetic Programming expression tree."""
    
    def __init__(self, value: str | float, is_terminal: bool = False, arity: int = 0):
        self.value = value
        self.is_terminal = is_terminal
        self.arity = arity
        self.children: list['GPNode'] = []
    
    def evaluate(self, variables: dict[str, float]) -> float:
        """Evaluate this node given a variable assignment."""
        ...
    
    def deep_copy(self) -> 'GPNode':
        """Create a deep copy of this subtree."""
        ...
Key changes:

All names to English (valor → value, hijos → children, evaluar → evaluate)
Type hints added
Same mathematical logic preserved
[NEW] 
generator.py
Extracted from legacy lines 179-225. GeneradorGP → GPGenerator:

python

class GPGenerator:
    """Generates random GP expression trees using the Grow method."""
    
    def __init__(self, function_set: list[str], terminal_set: list[str], max_depth: int = 3):
        ...
    
    def generate_tree_grow(self, max_depth: int) -> GPNode:
        ...
    
    def generate_population(self, size: int) -> list[GPNode]:
        ...
[NEW] 
semantics.py
Extracted from legacy lines 227-270. EvaluadorSemántico → SemanticEvaluator:

python

class SemanticEvaluator:
    """Evaluates GP trees to produce semantic vectors as PyTorch tensors."""
    
    def __init__(self, variable_names: list[str], device: torch.device):
        ...
    
    def evaluate_individual(self, tree: GPNode, X: np.ndarray) -> torch.Tensor:
        """Evaluate a single tree on all data points. Returns tensor on device."""
        ...
    
    def create_semantic_matrix(self, population: list[GPNode], X: np.ndarray) -> torch.Tensor:
        """Evaluate entire population → (pop_size, M) tensor on device."""
        ...
Key optimization: The output is directly a torch.Tensor placed on the target device. No intermediate NumPy step needed after this point.

Component: src/models/ — Neural Network (GPU)
[NEW] 
gsgp_nn.py
This is the core refactoring. The entire GSGP_NN_Final class (legacy lines 272-713) becomes a proper torch.nn.Module. This is where the massive parallelization happens.

python

import torch
import torch.nn as nn
class GSGPNNLayer(nn.Module):
    """A single GSGP-NN semantic layer.
    
    Replaces the per-node loop in calcular_semantica_nodo() with 
    batched matrix operations on GPU.
    
    Legacy: for n in range(1, N+1): ... → single matrix multiply
    """
    
    def __init__(self, num_nodes: int, num_patterns: int):
        super().__init__()
        self.N = num_nodes
        self.M = num_patterns
        
        # Alpha weights: (N+1, N) — row 0 is parent weight, rows 1..N are layer inputs
        # Replaces self.alpha[(i, n, k)] dictionary
        self.alpha = nn.Parameter(torch.empty(num_nodes + 1, num_nodes))
        nn.init.kaiming_normal_(self.alpha)
        
        # Mutation coefficients: replaces self.ms[(n,k)] and self.beta[(n,k)]
        self.ms = nn.Parameter(torch.empty(num_nodes))
        self.beta = nn.Parameter(torch.empty(num_nodes))
        nn.init.normal_(self.ms, std=0.1)
        nn.init.normal_(self.beta, std=0.1)
    
    def forward(
        self, 
        prev_semantics: torch.Tensor,   # (N, M) — previous layer output
        parent_semantics: torch.Tensor,  # (N, M) — fixed parent semantics
        rt1: torch.Tensor,              # (N, M) — random tree 1 semantics
        rt2: torch.Tensor,              # (N, M) — random tree 2 semantics
    ) -> torch.Tensor:
        """
        Compute all N node semantics in parallel.
        
        Legacy equivalent (sequential, N iterations):
            for n in range(1, N+1):
                d_nk = abs(alpha[0,n,k]) + sum(abs(alpha[i,n,k]) for i in 1..N)
                H_nk = alpha[0,n,k]*parent[n] + sum(alpha[i,n,k]*prev[i])
                inherited = H_nk / d_nk
                mutation = beta[n,k] * tanh(ms[n,k] * (RT1[n] - RT2[n]))
                result[n] = inherited + mutation
        
        PyTorch equivalent (parallel, 1 operation):
        """
        # === INHERITED TERM (fully parallel) ===
        # Stack inputs: (N+1, M) — parent semantics + previous layer
        stacked_inputs = torch.cat([
            parent_semantics.unsqueeze(0),  # (1, N_parent, M) → need (1, M) per node... 
            prev_semantics                   # (N, M)
        ], dim=0)  # → but we need to handle parent per-node indexing
        
        # Actually, parent_semantics[n] is different per node, so:
        # all_inputs shape: (N+1, M) where row 0 = parent[n], rows 1..N = prev[1..N]
        # But parent[n] varies per output node n...
        
        # Correct approach: compute numerator H as matrix multiply
        # H[n] = alpha[0,n] * parent[n] + sum_i(alpha[i,n] * prev[i])
        # = alpha[0,n]*parent[n] + alpha[1:,n]^T @ prev
        
        # parent contribution: (N,) * (N, M) → element-wise per node
        parent_contrib = self.alpha[0].unsqueeze(1) * parent_semantics  # (N, M)
        
        # prev layer contribution: alpha[1:] is (N_in, N_out), prev is (N_in, M)
        # Result: (N_out, M)  
        prev_contrib = torch.mm(self.alpha[1:].t(), prev_semantics)  # (N, M)
        
        H = parent_contrib + prev_contrib  # (N, M) — numerator
        
        # Denominator: sum of absolute alphas per output node
        d = self.alpha.abs().sum(dim=0, keepdim=True).t()  # (N, 1)
        d = d.clamp(min=1e-12)
        
        inherited = H / d  # (N, M) — broadcasting
        
        # === MUTATION TERM (fully parallel) ===
        rt_diff = rt1 - rt2                                    # (N, M)
        mutation_base = self.ms.unsqueeze(1) * rt_diff          # (N, M)
        mutation = self.beta.unsqueeze(1) * torch.tanh(mutation_base)  # (N, M)
        
        return inherited + mutation  # (N, M)
class GSGPNN(nn.Module):
    """
    Geometric Semantic Genetic Programming Neural Network.
    
    Full model with K semantic layers + linear output layer.
    All operations are tensorized for GPU execution.
    """
    
    def __init__(self, num_nodes: int, num_layers: int, num_patterns: int):
        super().__init__()
        self.N = num_nodes
        self.K = num_layers
        self.M = num_patterns
        
        # K semantic layers
        self.layers = nn.ModuleList([
            GSGPNNLayer(num_nodes, num_patterns) for _ in range(num_layers)
        ])
        
        # Output layer: linear combination of final semantics
        self.output_linear = nn.Linear(num_nodes, 1)
    
    def forward(
        self,
        initial_semantics: torch.Tensor,  # (N, M)
        parent_semantics: torch.Tensor,   # (N, M) 
        route_semantics: list[dict],       # K dicts with 'rt1', 'rt2' tensors
    ) -> torch.Tensor:
        """
        Forward pass through all K layers.
        Returns predictions of shape (M,).
        """
        current = initial_semantics  # (N, M)
        
        for k, layer in enumerate(self.layers):
            current = layer(
                prev_semantics=current,
                parent_semantics=parent_semantics,
                rt1=route_semantics[k]['rt1'],
                rt2=route_semantics[k]['rt2'],
            )
        
        # Output: linear combination of final node semantics
        # current is (N, M), we need (M, N) for linear layer
        output = self.output_linear(current.t())  # (M, 1)
        return output.squeeze(1)  # (M,)
Parallelization gains summary:

Operation	Legacy (CPU)	PyTorch (GPU)
H_nk numerator	N² scalar multiplies in Python loop	1 matrix multiply torch.mm()
Denominator d_nk	N sums per node, N nodes	1 abs().sum() across dim
Mutation	N iterations, each with element-wise ops	1 batched tanh() call
Output prediction	N scalar-vector multiplies	1 nn.Linear forward
Gradients	150 lines of manual calculus	Automatic via loss.backward()
Component: src/training/ — Training Pipeline (GPU)
[NEW] 
trainer.py
python

class Trainer:
    """Handles training loop, optimizer, and device management."""
    
    def __init__(
        self,
        model: GSGPNN,
        optimizer: torch.optim.Optimizer,  # Use PyTorch's Adam directly
        device: torch.device,
        early_stopping: EarlyStopping | None = None,
        callbacks: list[Callback] | None = None,
    ):
        ...
    
    def fit(
        self,
        train_data: dict[str, torch.Tensor],  # semantics already on device
        y_train: torch.Tensor,
        val_data: dict[str, torch.Tensor] | None = None,
        y_val: torch.Tensor | None = None,
        epochs: int = 200,
    ) -> dict:
        """
        Main training loop.
        
        Key difference from legacy:
        - No manual gradient computation
        - Uses torch.optim.Adam (battle-tested, GPU-accelerated)
        - loss.backward() computes ALL gradients automatically
        - optimizer.step() applies updates
        """
        self.model.train()
        
        for epoch in range(epochs):
            # Forward
            y_pred = self.model(
                train_data['initial_semantics'],
                train_data['parent_semantics'],
                train_data['route_semantics'],
            )
            
            # Loss
            loss = F.mse_loss(y_pred, y_train)
            
            # Backward — replaces 150 lines of manual gradient code
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (replaces manual np.clip)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            # Update — replaces manual Adam implementation
            self.optimizer.step()
            
            # Early stopping check
            if self.early_stopping and self.early_stopping(loss.item()):
                break
        
        return history
[NEW] 
early_stopping.py
Extracted from legacy lines 86-117. EarlyStoppingRobusto → EarlyStopping:

python

class EarlyStopping:
    """Early stopping with warmup period and best model restoration."""
    
    def __init__(self, patience: int = 100, min_delta: float = 1e-6, 
                 warmup_epochs: int = 50):
        ...
    
    def __call__(self, current_loss: float) -> bool:
        """Returns True if training should stop."""
        ...
    
    def state_dict(self) -> dict:
        """For checkpointing."""
        ...
[NEW] 
callbacks.py
python

class TrainingLogger:
    """Logs metrics to console with formatting."""
    ...
class ModelCheckpoint:
    """Saves best model weights to disk."""
    ...
class TensorBoardLogger:
    """Optional: logs to TensorBoard for visualization."""
    ...
Component: src/data/ — Data Pipeline
[NEW] 
dataset.py
python

class RegressionDataset:
    """Loads a regression dataset from a text file.
    
    Handles scaling and train/test split.
    """
    
    def __init__(self, path: str, test_size: float = 0.3, seed: int = 0):
        ...
    
    def get_tensors(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        """Returns (X_train, y_train, X_test, y_test) as tensors on device."""
        ...
    
    def inverse_transform_y(self, y_scaled: torch.Tensor) -> torch.Tensor:
        """Convert predictions back to original scale."""
        ...
[NEW] 
loader.py
python

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load experiment configuration from YAML."""
    ...
def prepare_semantic_data(
    X: torch.Tensor,
    gp_generator: GPGenerator, 
    evaluator: SemanticEvaluator,
    num_nodes: int,
    num_layers: int,
    population_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Generate all GP populations, evaluate semantics, and organize
    into the tensor structure needed by GSGPNN.forward().
    
    Returns dict with 'initial_semantics', 'parent_semantics', 'route_semantics'.
    All tensors are on the specified device.
    """
    ...
Component: src/utils/ — Utilities
[NEW] 
device.py
python

def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device with diagnostic info."""
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device
def estimate_memory_usage(N: int, K: int, M: int) -> float:
    """Estimate GPU memory needed in MB for given configuration."""
    ...
[NEW] 
seed.py
python

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
[NEW] 
metrics.py
python

def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    """Compute MAE, MSE, RMSE, R² on GPU tensors."""
    ...
def compute_metrics_original_scale(
    y_true_scaled: torch.Tensor, y_pred_scaled: torch.Tensor,
    scaler_y, device: torch.device,
) -> dict[str, float]:
    """Compute metrics after inverse-transforming to original scale."""
    ...
Component: scripts/ — Entry Points
[NEW] 
run_experiment.py
Replaces the legacy if __name__ == "__main__" block (lines 1079-1178):

python

"""
Main experiment runner for GSGP-NN.
Usage:
    python scripts/run_experiment.py                    # Run all datasets
    python scripts/run_experiment.py --dataset yacht     # Single dataset
    python scripts/run_experiment.py --seed 42 --epochs 500  # Custom params
"""
import argparse
import yaml
from src.utils.device import get_device
from src.utils.seed import set_seed
from src.data.dataset import RegressionDataset
from src.data.loader import prepare_semantic_data
from src.evolution.generator import GPGenerator
from src.evolution.semantics import SemanticEvaluator
from src.models.gsgp_nn import GSGPNN
from src.training.trainer import Trainer
def run_single_experiment(config: dict, seed: int, device: torch.device) -> dict:
    """Run a single training experiment and return metrics."""
    ...
def run_robustness_analysis(config: dict, num_runs: int, device: torch.device) -> pd.DataFrame:
    """Run multiple experiments for robustness analysis."""
    ...
def main():
    parser = argparse.ArgumentParser(description="GSGP-NN Experiment Runner")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--num-runs", type=int, default=30)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    ...
Component: Environment & Dependencies
[MODIFY] 
enviroment.yml
yaml

name: gsgp_nn
channels:
  - pytorch
  - nvidia
  - defaults
dependencies:
  - python=3.10
  - pytorch>=2.0
  - pytorch-cuda=12.1  # Match your CUDA version
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - pyyaml
  - pip:
    - tensorboard  # optional, for logging
Verification Plan
Automated Tests
bash

# Unit tests for model components
python -m pytest tests/test_model.py -v
# Unit tests for GP evolution
python -m pytest tests/test_evolution.py -v
# Integration test: compare PyTorch output vs legacy NumPy output
python -m pytest tests/test_training.py -v
# Numerical equivalence test (critical)
python -m pytest tests/test_training.py::test_numerical_equivalence -v
Manual Verification
Numerical Equivalence: Run both legacy and PyTorch versions on the same dataset with the same seed and verify that the loss curves match within float32 tolerance (~1e-6)
GPU Utilization: Monitor nvidia-smi during training to verify the GPU is actually being used
Speedup Measurement: Compare wall-clock time per epoch between legacy (CPU) and new (GPU) on the tower.txt dataset with N=200
Full Robustness Run: Run the 30-run analysis on one dataset and compare statistical results (mean R², std) with legacy results
Performance Benchmarks
bash

# Single run timing comparison
python scripts/run_experiment.py --dataset tower --num-runs 1 --device cuda
python scripts/run_experiment.py --dataset tower --num-runs 1 --device cpu
# Full robustness analysis
python scripts/run_experiment.py --num-runs 30
