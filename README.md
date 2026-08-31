# CGT-AI Framework

A Python AI framework for Combinatorial Game Theory with Numba acceleration support.


## Overview

**CGT-AI** is a Python framework designed for building AI agents that play combinatorial games. It provides a clean, composable architecture centered around immutable game states, pluggable value functions, and search algorithms—all accelerated with Numba JIT compilation for high performance.

Unlike existing CGT libraries that focus on mathematical game analysis, CGT-AI is built for **practical AI development**: defining games, evaluating positions, and searching for optimal moves at scale.


## Core Features

- **Immutable Data Objects** — All game states, steps, and values are immutable and serializable to Unicode strings (pickle + base64 by default, with `ast.literal_eval` support for literal objects).
- **Numba Acceleration** — JIT-compile hot paths like legal move generation and evaluation functions for near-native performance.
- **Composable Value Functions** — Chain multiple evaluation functions with fallback semantics (`ValueFunction >> ValueFunction`).
- **Pluggable Searchers** — MCTS, Minimax, Alpha-Beta, and custom search algorithms with a unified interface.
- **Engine Wrappers** — Built-in boilerplate for XBoard, UCI, and boardgame.io (WebAssembly via monkey-patching).


## Architecture

### DataObject

The foundation of the framework. Every `DataObject` is:

- **Immutable** — Once created, it cannot be modified.
- **Serializable** — Can be saved and loaded as a Unicode string.

```python
from cgt_ai import DataObject

class MyData(DataObject):
    def __init__(self, value):
        self._value = value

obj = MyData(42)
serialized = obj.dumps()   # base64-encoded pickle
restored = MyData.loads(serialized)
```

For literal objects (built-in types, tuples, lists, dicts, etc.), use `LiteralDataObject` which uses `ast.literal_eval` for safe serialization:

```python
from cgt_ai import LiteralDataObject

state = LiteralDataObject({"position": (3, 4), "score": 100})
```

### CGTStep

A `CGTStep` represents a single move or action in a game. It inherits from `DataObject` and contains the source code definition of the `CGTGame` it is compatible with.

```python
from cgt_ai import CGTStep

class MoveStep(CGTStep):
    def __init__(self, x, y):
        self.x = x
        self.y = y
```

### CGTGame

A `CGTGame` defines the rules of a combinatorial game. The key interface is `legal_moves()`, which generates compatible `CGTStep` objects:

```python
from cgt_ai import CGTGame

class MyGame(CGTGame):
    def legal_moves(self):
        # May be infinite — returns a generator
        for x in range(self.width):
            for y in range(self.height):
                if self._is_legal(x, y):
                    yield MoveStep(x, y)
```

### ValueFunction

A `ValueFunction` encapsulates evaluation logic. It contains source code defining an `evaluate` variable, inspectable via `inspect.getsignature`. The function accepts a `CGTGame` and returns a `CGTValue` (a comparable `DataObject` implementing `__lt__` and `__eq__`).

```python
from cgt_ai import ValueFunction, TotalValueFunction

@ValueFunction
def material_score(game):
    """Simple material count evaluation."""
    return CGTValue(game.material_advantage())

@TotalValueFunction
def winning_determination(game):
    """Total evaluation — always returns a value."""
    if game.is_terminal():
        return CGTValue(float('inf') if game.is_win() else float('-inf'))
    return CGTValue(0.0)
```

Value functions can be chained with the `>>` operator. The chain runs functions in order, falling back to the next if a `ValueError` is raised:

```python
evaluator = material_score >> winning_determination
# If material_score raises ValueError, winning_determination is used.
```

### Searcher

Searchers implement search algorithms. A searcher is an inspectable callable that accepts a `CGTGame` and an optional `datetime.datetime` deadline, returning `(CGTValue, CGTStep, CGTValue)` — the value before the move, the best move, and the value after the move.

```python
from cgt_ai import mcts, minimax, alphabeta

# Basic usage
value_before, best_move, value_after = mcts(game, deadline=datetime.now() + timedelta(seconds=5))

# Optional: get the principal variation
moves = mcts.mainline(game, depth=10)

# Optional: get multiple variations
variations = mcts.principal_variations(game, pv=5, depth=10)
```

**Built-in searchers:**

| Searcher | Description |
|----------|-------------|
| `MCTS` | Monte Carlo Tree Search with UCB1 |
| `Minimax` | Classic minimax with configurable depth |
| `AlphaBeta` | Minimax with alpha-beta pruning |
| `Negamax` | Negamax variant with transposition tables |

### EngineWrapper

`EngineWrapper` provides boilerplate for integrating with external frameworks:

```python
from cgt_ai import EngineWrapper, XBoardWrapper, UCIWrapper

# XBoard/Fairychess compatibility
xboard_engine = XBoardWrapper(game_class=MyGame, searcher=mcts)

# UCI compatibility
uci_engine = UCIWrapper(game_class=MyGame, searcher=alphabeta)

# boardgame.io → WebAssembly (via monkey-patching numba's LLVM backend)
wasm_engine = boardgameio_wrapper(game_class=MyGame, searcher=mcts)
```


## Numba Acceleration

The framework leverages Numba to JIT-compile performance-critical sections:

- **Legal move generation** — `@numba.jit` on `legal_moves()` loops
- **Evaluation functions** — JIT-compiled value function bodies
- **Search internals** — Core search loops and node expansions

Example of accelerating a game:

```python
from numba import jit
from cgt_ai import CGTGame, accelerate

@accelerate
class FastGame(CGTGame):
    @jit(nopython=True)
    def legal_moves(self):
        # Hot loop — compiled to native code
        ...
```

For GPU acceleration, the framework supports `numba.cuda` for parallel MCTS.


## Related Work

| Project | Focus | Approach |
|---------|-------|----------|
| **EasyAI** | Two-player abstract games | Pure Python, Negamax with alpha-beta |
| **cgt-tools** | CGT mathematical toolkit | Rust + Python thin wrapper |
| **Ludology** | Python CGT package | Mathematical game analysis |
| **python-chess** | Chess engine communication | UCI/XBoard protocols |
| **MCTS-NC** | GPU-parallelized MCTS | Numba CUDA implementation |

**What makes CGT-AI different:**

- **Unified abstraction** — Same interface for any combinatorial game
- **Numba-first** — Acceleration is a core design principle, not an afterthought
- **Serialization by design** — Every object is persistable, enabling saving/loading game states, evaluation functions, and search results
- **Value function chaining** — Composable evaluation with graceful fallback
- **Cross-framework compatibility** — XBoard, UCI, boardgame.io out of the box


## Quick Example

```python
from cgt_ai import CGTGame, CGTStep, mcts, ValueFunction
from datetime import datetime, timedelta

# Define a simple Nim game
class NimStep(CGTStep):
    def __init__(self, pile, count):
        self.pile = pile
        self.count = count

class NimGame(CGTGame):
    def __init__(self, piles):
        self.piles = list(piles)
    
    def legal_moves(self):
        for i, count in enumerate(self.piles):
            for take in range(1, count + 1):
                yield NimStep(i, take)
    
    def apply(self, step):
        new_piles = self.piles.copy()
        new_piles[step.pile] -= step.count
        return NimGame(new_piles)

# Define a value function
@ValueFunction
def nim_value(game):
    # Sprague-Grundy for Nim
    xor_sum = 0
    for p in game.piles:
        xor_sum ^= p
    return CGTValue(0 if xor_sum == 0 else 1)

# Search for the best move
searcher = mcts(iterations=1000)
value_before, best_move, value_after = searcher(
    NimGame([3, 4, 5]), 
    deadline=datetime.now() + timedelta(seconds=1)
)

print(f"Best move: take {best_move.count} from pile {best_move.pile}")
```


## Installation

```bash
pip install cgt-ai
# With Numba acceleration
pip install cgt-ai[numba]
# With GPU support
pip install cgt-ai[cuda]
```


## License

[AGPLv3](LICENSE)
