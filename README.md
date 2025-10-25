# Scaling Reinforcement Learning at Test-Time using Model Predictions: An attempt at playing chess phenomenally slowly

_TLDR: Might work. Chess is hard though. Jump to [installation](#installation)|[usage](#usage)._

### Motivation

Model Predictive Control (MPC) and Reinforcement Learning (RL) are SOTA approaches for controlling lots of systems (e.g. energy systems, inventory management, stock trading, etc.)

RL is excellent at learning complex patterns directly from data and learning long-term strategies outside the reach of MPC schemes. Whereas MPC is great at using our existing knowledge of systems (e.g. physics) and performing in unseen situations (where potentially we can forecast the conditions we haven't seen before).

So the question is: **Can we combine MPC and RL to get both complexity and flexibility?**
The idea is to use the predictive model to allow the RL agent to hypothesise, plan, and evaluate its actions. And then use this information to tune the policy to make it better at controlling the system in its current condition, rather than the data distribution it was trained on.
_This might for example allow us to perform better in physics regimes we can simulate but haven't got in our training dataset._
> The need for model generalisation is a bit of a myth. It assumes we can't change the model. But we've seen LLMs scaled at test-time very effectively. It just takes more compute.

[Previous work with Transfer Learning](https://arxiv.org/abs/2508.21615) has shown that fine tuning a general model can provide better performance for a specific task, combining strong general strategies with task specific info.
So why not do this for control in real time?
> Unfortunately for this project the best pre-trained model I could achieve is still pretty bad, so combining its (broadly unhelpful) knowledge with the observations is limited. In theory, if you could train a good chess model with strong strategy fundamentals, and then fine-tune it for the current position it could work well.

### Goals

1. Demonstrate this idea.

2. Use minimal hardware (_just my laptop, I don't have a server or chunky GPU_)

3. Perform compute within 'real time' (_for chess this is about 2-3 mins per move to mimic 90min format classical chess_)

I chose chess to test this because:
1. The rules are simple
2. There are great existing simulators (_so hopefully development is quick_)
3. Testing performance is obvious (_use [ELO](https://www.chess.com/terms/elo-rating-chess), i.e. can it win games?_)

_And tbf it did deliver on these. Dev was pretty quick, and testing was straightforward._

However, it was probably a bad choice because:
- Chess is really hard (_shock, I know_); creating a reasonable chess engine with a tunable ML model is extremely challenging
- It's not clear how to get 'reward' from the environment in a way that doesn't feel like cheating (whether the model wins the game is far too parse a signal for the approach I wanted to take)

_The headache of trying to train a decent chess engine without a beefy GPU took up most of the project time._

### Approach & Model

#### Strategy
1. Predict possible game realisation for each available move (_predict how game will play out using the model and a simulated opponent_)
2. Evaluate those realisation using a position scorer
3. Tune the model using this data (_combine the model's existing knowledge with this position specific information from the environment_)

This strategy is somewhat similar to the Monte Carlo Tree Search (MCTS) used by [AlphaZero](https://nikcheerla.github.io/deeplearningschool/2018/01/01/AlphaZero-Explained/) (it's kind of exploring the future in a similar way, i.e. on-policy-ish, but maybe having access to an opponent engine is the difference c.f. the need to self-play)

The physical analogy to this scheme is having a grand master sitting next to you who you can explain your plan for the next few turns to, and for each rollout they'll give you feedback on how good the plan is on a scale of 1 to 10, and you use this info to help you plan your current turn.

Because the information I'm getting from the environment is estimates (scores) of how good the rollout following on from each move will be, I needed to use a value regression model to estimate position scores. And to make it tunable I used neural nets for the regression - both CNN (Alphazero-like) and Transformer architectures.
> Small and simple chess engines can be [super effective](https://github.com/SebLague/Tiny-Chess-Bot-Challenge-Results), but I needed something tunable

I used the position scores because I didn't want to have to play out full games to get reward signals (this is super sparse, and for other systems, e.g. energy you have much more dense rewards). But, taking this approach sort of requires cheating (having a scoring model that already knows chess - just using this scoring model to make the moves would beat Magnus Carlsen).

### Results so far & Issues

The base model can kinda play some chess against a weak stockfish opponent.
The Mean Absolute Error (MAE) and the validation set is about 150 [centipawns](https://www.reddit.com/r/chess/comments/mtzfaj/what_is_centipawn/), so not brilliant. But it can make some progress (it's able to get itself into winning positions against weak opponents in the early game). Unfortunately, it can't actually win (it hasn't seen many checkmate positions and doesn't understand what checkmate is).

There are two key issues currently:
1. The model isn't good enough to provide useful trajectories (it loses very quickly from most starting positions)
2. It isn't flexible enough to take up the environment reward properly (even when trained using perfect position evaluations it still can't win games, it's just not able to pick up on the new information well enough, and its existing knowledge is sufficiently bad that it's counterproductive)

[This repo](https://github.com/undera/chess-engine-nn) had some similar issues with the score regression approach to playing chess.
Most people getting good results, e.g. [this repo](https://github.com/dogeystamp/chess_inator), seem to train using self-play (unsurprisingly it seems to good folks from DeepMind know what they're doing).

### Possible solutions

- Use someone else's pre-trained model, e.g. [Leela Chess Zero](https://lczero.org/) or [Chessformer](https://arxiv.org/abs/2409.12272v2).
The issue with this is the ones that actually work are either not (easily) tunable, or are huge models that are too good to need any tuning

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

<br/>

## Usage

**ToDo.**

1. Generate example games by playing Stockfish against itself
```bash
python scripts/generate_game_data.py resources/stockfish-17-macos-m1-apple-silicon data/games/games.json --games 2500 --workers 12
```
_Low ELO Stockfish instances are used to bias the games towards positions the model is more likely to encounter during play._

2. Convert dataset of example games to dataset of scored positions
```bash
python scripts/games_to_positions.py data/games/games.json data/positions/positions.json 50000 resources/stockfish-17-macos-m1-apple-silicon
```

3. Train neural network position evaluation model
```bash
python scripts/train_model.py data/models/Cmodel data/positions/positions.json --epochs 50 --bs 256 --vs 0.1
```
Training can be monitored with Tensorboard

```bash
tensorboard --logdir data/models/Cmodel/train
```

4. Watch the chess model play Stockfish
```bash
python scripts/benchmark.py data/models/Cmodel resources/stockfish-17-macos-m1-apple-silicon --plot
```

5. Watch the chess model play using test-time scaling (_takes waaay longer_)
```bash
python scripts/benchmark.py data/models/Cmodel resources/stockfish-17-macos-m1-apple-silicon --plot --use_ttmp
```

<br/>

## Dev notes

### Stockfish engine processes
Connections to the Stockfish engine objects need to be closed explicitly in order for them to be properly cleaned up and prevent the program from hanging. This is done by calling the `close()` method on the relevant object which owns the engine connection.
