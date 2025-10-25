# Scaling Reinforcement Learning at Test-Time using Model Predictions: An attempt at playing chess phenomenally slowly

_TLDR: Might work. Chess is hard though. Jump to [installation](#installation)|[usage](#usage)._

### Motivation

Combine MPC & RL - get RL to hypothesise & plan and use information from this planning to improve its policy.

Generalisation is a myth. It assumes we can't change the model. But we've seen LLMs scaled at test-time.

From [previous work with Transfer Learning](https://arxiv.org/abs/2508.21615), fine tuning a general model provides better performance for a specific task, so why not do this for control in real time?
> Unfortunately for this project the best pre-trained model I could achieve is still pretty bad, so combining its (broadly unhelpful) knowledge with the observations is limited. In theory, if you could train a good chess model with strong strategy fundamentals, and then fine-tune it for the current position it could work well.

### Goal

1. To demonstrate this idea.

2. Use minimal hardware (_Just my laptop - I don't have a server or good GPU_)

3. Perform compute within 'real time' constraints. (_For chess this is about 2-3 mins per move to mimic 90min format classical chess_)

I chose chess to test this because:
1. rules are simple
2. existing simulators (quick development)
3. obvious test of performance (elo, i.e. can it win games?)

_And it did deliver on these. Dev was pretty quick, and testing was straightforward._

However, it was probably a bad choice because:
- chess is really hard - creating a reasonable chess engine with a trainable NN is challenging
- it's not clear how to get 'reward' from the environment in a way that doesn't feel like cheating

_The headache of trying to train a decent chess engine with a beefy GPU took up most of the project time._

### Approach & Model

... predict out possible futures (estimates of how game will play out) for each available move; evaluate those futures using external scorer; tune model using this data (combine existing knowledge with environment info) ...

This strategy is somewhat similar to the Monte Carlo Tree Search (MCTS) used by [AlphaZero](https://nikcheerla.github.io/deeplearningschool/2018/01/01/AlphaZero-Explained/) (it's kind of exploring the future in a similar way, i.e. on-policy-ish, but maybe having access to an opponent engine is the difference c.f. the need to self-play)

The physical analogy to this scheme is you having a grand master sitting next to you who you can explain your plan for the next few turns to, and for each rollout they'll give you feedback on how good the plan is on a scale of 1 to 10, and you use this info to help you plan your current turn.

... because of use of prediction scoring, I took a value regression approach, and to make it tunable I used neural nets - both CNN (Alphazero-like) and transformer architectures implemented ...
> Small and simple chess engines can be [super effective](https://github.com/SebLague/Tiny-Chess-Bot-Challenge-Results), but I needed something tunable

... basically I didn't want to have to play out full games to get reward signals (this is super sparse, and for other systems, e.g. energy you have much more dense rewards) ... but taking this approach sort of requires cheating (having an external scorer that already knows chess) ...

### Results so far & Issues

... base model can kinda play some chess against a rubbish stockfish - MAE is about 150 [centipawns](https://www.reddit.com/r/chess/comments/mtzfaj/what_is_centipawn/), so not brilliant, but it can make some progress (it can get into winning positions) ... but can't actually win (it hasn't seen many checkmate positions and doesn't understand what checkmate is) ...

... there are lots of issues ... the NN isn't good enough to provide useful trajectories, and isn't flexible enough to take up the environment info properly ... even when trained using 0 length trajectories (i.e. perfect position evaluations which can beat a GM), it still can't win games, it's just not able to pick up on the new information well enough, and the existing knowledge is sufficiently bad that it's counterproductive ...

[This repo](https://github.com/undera/chess-engine-nn) had some similar issues with the score regression approach to playing chess. Most people getting good results, e.g. [this repo](https://github.com/dogeystamp/chess_inator), seem to use self-play.

### Possible solutions

... use someone else's pre-trained model, e.g. [Leela Chess Zero](https://lczero.org/) or [Chessformer](https://arxiv.org/abs/2409.12272v2) ... problem is the ones that actually work are either not (easily) tunable, or are huge models that are too good to benefit from tuning ...

<br/>

## Installation

### Requirements

Install the required python packages from the requirements file:
```bash
pip3 install -r requirements.txt
```

Tensorflow is also needed. See installation instructions [here](https://www.tensorflow.org/install/pip).

### Module installation

To use the codebase, install the module in editable mode:
```bash
pip install -e .
```
from the root directory.

### Mac installation

Installing Tensorflow with GPU support on Apple Silicon is pretty fiddly.

After lots of searching, the only stable installation I could achieve on an M3 Pro device using `conda` is as follows:
```bash
conda create -n tf-macos python=3.11.11
pip install tensorflow==2.17
pip install tensorflow-metal==1.1
pip install tf_keras==2.17.0
conda install -c conda-forge cairo
pip install -r requirements.txt
```

Creds to [this thread](https://stackoverflow.com/questions/78845096/tensorflow-metal-not-installable-on-m2-macbook-and-github-page-is-down) for the solution.

### Stockfish

Download the specific [stockfish binary](https://stockfishchess.org/download/) for your platform. A potentially out-of-date script for downloading it was provided by the original repo author:

```bash
cd res
chmod +x get_stockfish.sh
./get_stockfish.sh linux   #or "mac", depending of your platform. 
```

If you download it manually, place it in the `resources` directory.


## Usage

**ToDo.**

1. Generate games
```bash
python scripts/generate_game_data.py resources/stockfish-17-macos-m1-apple-silicon data/games/games.json --games 2500 --workers 12
```

2. Convert games to positions
```bash
python scripts/games_to_positions.py data/games/games.json data/positions/positions.json 50000 resources/stockfish-17-macos-m1-apple-silicon
```

3. Train neural network position evaluation model
```bash
python scripts/train_model.py data/models/Cmodel data/positions/positions.json --epochs 50 --bs 256 --vs 0.1
```
Training can be monitored with Tensorboard, e.g.:

```bash
tensorboard --logdir data/models/Cmodel/train
```

4. Watch chess model play
```bash
python scripts/benchmark.py data/models/Cmodel resources/stockfish-17-macos-m1-apple-silicon --plot
```

5. Watch chess model using test-time scaling (_takes waaay longer_)
```bash
python scripts/benchmark.py data/models/Cmodel resources/stockfish-17-macos-m1-apple-silicon --plot --use_ttmp
```

## Dev notes

### Stockfish engine processes
Connections to the Stockfish engine objects need to be closed explicitly in order for them to be properly cleaned up and prevent the program from hanging. This is done by calling the `close()` method on the relevant object which owns the engine connection.
