# Neural chess <br> Reinforcement Learning based chess engine.

**TODO** Update project description and credit previous author.

Personal project to build a chess engine based using reinforcement learning.

The idea is to some sort replicate the system built by DeepMind with AlphaZero. I'm
aware that the computational resources to achieve their results is huge, but my aim
it's simply to reach an amateur chess level performance (about 1200-1400 Elo), not
state of the art.

At this moment, the approach I'm using is based on pre-training a model using self-play data of a Stockfish algorithm. Later, the idea is to put two models to play against each other and make selection/merges of weights (RL part).

If you want to reuse this code on your project, and have any doubt [here](https://github.com/AIRLegend/ChessRL/blob/master/DOCS.md) you will find some explanation about the most important classes. Also, feel free to open an issue on this repo to ask.

*Work in progress*

## Notes

- Using `pip install -e .` to install the code as a module and allow `src` to be accessed from anywhere

- Current base model settings (`modela`):
   - ToDo

- introduced `move_quality` parameter to Stockfish player to allow it to play worse moves (to simulate lower Elo opponents) - need to add some stochasticity for the predictions to avoid all the simulated games being the same
- I think if the original policy is crap, you don't really get any information out of the model predictions. It's also possible this approach isn't super powerful for chess because the state space is so unbelievably huge, so we can't explore a meaningful portion of the possible futures
- when predicting games, because the policy is deterministic, the first move is always the same, which means all the training data has the same opening move, which is bad, how can I introduce a bit of exploration to this first move? we also want a bit of randomness in stockfish as well to avoid all games being the same, maybe the training_mode flag is good enough?
- what if we accumulated training over the game? I.e. didn't reset the agent after each move

- starting chess games is suuuuper hard, openings are crazy, it might be good to start test games using an opening book to get into a reasonable position and not get absolutely destroyed in the first 10 moves

## Open issues

- Work out how to evaluate game performance, i.e. how good the play of a model is in a way other than naively playing another model
- Implement the test-time MPC fine-tuning strategy (use new class and overload `predict` method) - aim for 2-3 mins per move to mimic 90min format classical chess
- Think about how this strategy differs from the MCTS used by AlphaZero (it's kind of exploring the future in a similar way, i.e. on-policy-ish, but maybe having access to an opponent engine is the difference c.f. the need to self-play)

### Closing engine connections

Note that connections to the Stockfish engine objects need to be closed explicitly in order for them to be properly cleaned up and prevent the program from hanging. This is done by calling the `close()` method on the relevant object which owns the engine connection.

## Requirements
The necessary python packages (ATM) are listed in the requirements file.
You can install them with

```bash
pip3 install -r requirements.txt
```

Tensorflow is also needed, but you must install either `tensorflow` or `tensorflow-gpu` (for the development I used >= TF 2.0).

### Mac installation

Installing Tensorflow with GPU support on Apple Silicon is very fiddly.

After lots of searching the only stable install I could achieve on an M3 Pro device using `conda` is as follows:
```bash
conda create -n tf-macos python=3.11.11
pip install tensorflow==2.17
pip install tensorflow-metal==1.1
pip install tf_keras==2.17.0
conda install -c conda-forge cairo
pip install -r requirements.txt
```

Creds to [this thread](https://stackoverflow.com/questions/78845096/tensorflow-metal-not-installable-on-m2-macbook-and-github-page-is-down).

### Stockfish

Also, you need to download the specific 
[stockfish binary](https://stockfishchess.org/download/) for your platform,
for automating this made a script to automatically download it.

```bash
cd res
chmod +x get_stockfish.sh
./get_stockfish.sh linux   #or "mac", depending of your platform. 
```
If you want to download it manually, you have to put the stockfish executable/binary under `res/stockfish-17-macos-m1-apple-silicon` path, in order to the training script to detect it.


## Training
> **DISCLAIMER:** This is under development and can still contains bugs or  inefficiencies and modifications are being made.

> **DISCLAIMER 2:** As soon as I get acceptable results, I will also share weights/datasets with this code.

For training an agent in a supervised way you will need a saved dataset of games. The script `gen_data_stockfish.py` is made for generating a JSON with this. This script will play (and record) several games using two Stockfish instances. Execute it first to create this dataset (take a look at it's possible arguments).

The main purpose of this part is to pre-train the model to make the policy head of the network to reliably predict the game outcome. This will be useful during the self-play phase as the MCTS will make better move policies (reducing the training time).

```bash
cd src/chessrl
python gen_data_stockfish.py ../../res/stockfish-17-macos-m1-apple-silicon ../../data/dataset_stockfish.json --games 100
```

Once we have a training dataset (generated or your own adapted), start the supervised training with:

```bash
cd src/chessrl
python supervised.py ../../data/models/model1 ../../data/dataset_stockfish.json --epochs 2 --bs 4
```

Once we have a pre-trained model, we can move to the self-play phase. The incharged of this process is the `selfplay.py` script, which will fire up a instance of the model which play against itself and after each one, makes a training round (saving the model and the results). Please, take a look at the possible arguments. However, here you have an example. (Keep in mind that this is an expensive process which takes a considerable amount of time per move).

```bash
cd src/chessrl
python selfplay.py ../../data/models/model1-superv --games 100
```


## How do I view the progress?

The neural network training evolution can be monitored with Tensorboard, simply:

```bash
tensorboard --logdir data/models/model1/train
```
(And set the "Horizontal axis" to "WALL" for viewing all the different runs.)

Also, in the same model directory you will find a `gameplays.json` file which
contains the recorded training games of the model. With this, we can study its
behaviour over time.

## Can I play against the agent?

Yes. Under `src/webplayer` you will find a Flask app which deploys a web interface to play against the trained agent. There is another README with more information.


## Literature

1. Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning
   Algorithm, Silver, D. et al. https://arxiv.org/pdf/1712.01815.pdf
2. Mastering the game of Go without human knowledge. Silver, D. et al. https://www.nature.com/articles/nature24270.epdf
