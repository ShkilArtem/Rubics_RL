# Rubik's Cube RL

Навчальний проєкт із reinforcement learning: від випадкового агента й
табличного Q-learning до DQN, curriculum learning та neural-guided search.

## Що реалізовано

- фізично коректний 3×3 кубик на основі 3D-геометрії стікерів;
- 12 quarter-turn дій і вивід кубика хрестом;
- RL-середовище з `reset()`/`step()`, sparse reward і one-hot observation;
- random baseline та спільні метрики;
- табличний Q-learning;
- PyTorch DQN із replay buffer, target network і curriculum;
- checkpoints, CSV-метрики та графіки;
- value network і weighted A* для складніших станів;
- детерміновані тести симулятора, RL та neural-компонентів.

Детальне пояснення понять і рекомендований порядок роботи є у
[`LEARNING_GUIDE.md`](LEARNING_GUIDE.md).

## Оточення

Проєкт налаштовано на Python 3.12.10 та CUDA-збірку PyTorch для RTX 4070.

Активуй уже створене оточення:

```powershell
.\.venv\Scripts\Activate.ps1
```

Якщо PowerShell забороняє activation script, запускай Python напряму:

```powershell
.\.venv\Scripts\python.exe main.py
```

Відтворення оточення з нуля:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Перевірка GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Запуск

У верхній частині `main.py` зміни `RUN_MODE`:

| Режим | Що робить |
|---|---|
| `demo` | Показує кубик і задані вручну рухи |
| `random` | Вимірює випадкового агента |
| `train_tabular` | Навчає табличний Q-learning |
| `train_dqn` | Запускає DQN curriculum |
| `evaluate` | Перевіряє збережений DQN checkpoint |
| `train_value` | Навчає value network |
| `solve` | Запускає weighted A* із value network |

Після цього одна команда запускає вибраний режим:

```powershell
python main.py
```

`train_dqn` і `train_value` можуть працювати довго. Спочатку залишай depth 1
і невеликі training limits.

## Тести

```powershell
python -m unittest discover -s tests -v
```

## Основні файли

```text
main.py                       # Налаштування та всі режими запуску
rubiks_rl/cube.py             # Стан кубика і публічні операції
rubiks_rl/geometry.py         # Фізична 3D-модель поворотів
rubiks_rl/env.py              # RL-середовище
rubiks_rl/agents.py           # Random і tabular агенти
rubiks_rl/dqn.py              # DQN та curriculum training
rubiks_rl/value_learning.py   # Value network
rubiks_rl/search.py           # Weighted A*
tests/                        # Unit та smoke tests
```
