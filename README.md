# Rubik’s Cube Reinforcement Learning

A Python project for studying reinforcement learning and heuristic search on a
3×3 Rubik’s Cube. It combines a geometry-based simulator, tabular Q-learning,
Deep Q-Networks (DQN), and value-guided weighted A* search.

The project provides a common environment for comparing policies under controlled
scramble depths and action budgets. It is an experimental learning framework,
rather than a general-purpose optimal cube solver.

## Implemented Methods

- Geometry-derived sticker permutations and 12 quarter-turn actions.
- An environment with `reset()` and `step()`, terminal rewards, step penalties,
  timeout penalties, and configurable episode limits.
- Discrete states for tabular methods and 324-component one-hot observations
  for neural models.
- A random-policy baseline and shared evaluation metrics.
- Tabular Q-learning with epsilon-greedy exploration.
- DQN with experience replay, a target network, Huber loss, gradient clipping,
  and a curriculum over increasing scramble depths.
- Value estimation through scramble-length supervision and Bellman refinement.
- Weighted A* with duplicate-state handling, inverse-action pruning, batched
  heuristic evaluation, and state and elapsed-time limits.
- Model checkpoints, CSV training metrics, and training plots.

Conceptual explanations and a suggested study sequence are available in
[LEARNING_GUIDE.md](LEARNING_GUIDE.md), currently written in Ukrainian.

## Environment and Installation

The development environment uses Python 3.12. The pinned dependencies in
`requirements.txt` target a CUDA 13.0 build of PyTorch; they are not a universal
CPU-only installation specification. Neural components select CUDA when available
and otherwise use the CPU.

From the repository root, using Windows PowerShell:

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
~~~

If activation is unavailable, invoke the environment interpreter directly:

~~~powershell
.\.venv\Scripts\python.exe main.py
~~~

Inspect CUDA availability without assuming a GPU is present:

~~~powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
~~~

## Configuration and Execution

Set `RUN_MODE` in the `SETTINGS` section of `main.py`, then run:

~~~powershell
python main.py
~~~

| Mode | Description |
| --- | --- |
| `demo` | Display the cube and apply a predefined sequence of moves. |
| `random` | Evaluate random policies at the configured scramble depth. |
| `train_tabular` | Train and evaluate a tabular Q-learning agent. |
| `train_dqn` | Train a DQN using the scramble-depth curriculum. |
| `evaluate` | Evaluate a saved DQN checkpoint without exploration. |
| `train_value` | Train a value network for heuristic search. |
| `solve` | Run weighted A* using a saved value-network checkpoint. |

The settings control random seeds, scramble depth, evaluation episodes, training
budgets, checkpoint paths, heuristic weight, and search limits. Additional
parameters are defined in `TabularTrainingConfig`, `DQNTrainingConfig`, and
`ValueTrainingConfig`.

Start with shallow scrambles and small training budgets when checking a new
configuration. The `evaluate` and `solve` modes require the corresponding
checkpoints to exist.

## Evaluation and Limitations

Policy evaluation reports solved episodes, success rate, average reward, and
average episode length. Search additionally reports the solution sequence,
expanded and generated states, elapsed time, and termination reason.

For comparisons, keep scramble seeds, evaluation depth, episode count, and action
limits consistent. Training and evaluation routines accept explicit seeds.
However, the `random` entry-point mode generates policy seeds from an unseeded
generator; use `RandomAgent` and `evaluate_agent` directly with explicit seeds
for reproducible baseline comparisons.

A scramble’s length is not necessarily its optimal solution distance. Different
random seeds do not guarantee disjoint training and evaluation states, especially
at shallow depths. Success on short scrambles does not establish performance on
arbitrary cube configurations.

The learned heuristic is not guaranteed to be admissible or consistent, and this
weighted A* implementation does not guarantee a shortest solution. Search time
limits are checked between expansions rather than interrupting an inference call.
Configured training depths are targets, not evidence of achieved performance.

## Tests

~~~powershell
python -m unittest discover -s tests -v
~~~

The suite covers cube transformations, environment behavior, tabular updates,
search, neural components, and checkpoint serialization. Small neural training
tests verify basic operation rather than deep-scramble performance.

## Repository Structure

~~~text
main.py                       Configuration and execution modes
rubiks_rl/
    cube.py                   Cube state and move interface
    geometry.py               Sticker geometry and move permutations
    env.py                    Reinforcement learning environment
    agents.py                 Random and tabular Q-learning agents
    tabular_training.py       Tabular training loop
    evaluation.py             Shared policy evaluation
    dqn.py                    DQN and curriculum training
    value_learning.py         Value estimation and training
    search.py                 Weighted A* and solver evaluation
    metrics.py                Metric export and plotting
tests/                        Unit and smoke tests
checkpoints/                  Generated model checkpoints
outputs/                      Generated metrics and plots
~~~

Output directories are created as needed by the corresponding routines.
