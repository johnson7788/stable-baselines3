<img src="docs/\_static/img/logo.png" align="right" width="40%"/>

[![pipeline status](https://gitlab.com/araffin/stable-baselines3/badges/master/pipeline.svg)](https://gitlab.com/araffin/stable-baselines3/-/commits/master) [![Documentation Status](https://readthedocs.org/projects/stable-baselines/badge/?version=master)](https://stable-baselines3.readthedocs.io/en/master/?badge=master) [![coverage report](https://gitlab.com/araffin/stable-baselines3/badges/master/coverage.svg)](https://gitlab.com/araffin/stable-baselines3/-/commits/master)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# Stable Baselines3

Stable Baselines3（SB3）是PyTorch中强化学习算法的一套可靠实现。它是[Stable Baselines](https://github.com/hill-a/stable-baselines)的下一个主要版本。

您可以阅读Stable Baselines3的详细介绍  [v1.0 blog post](https://araffin.github.io/post/sb3/).

这些算法将使研究界和工业界更容易复制、完善和确定新的想法，并将创造良好的基线，在此基础上建立项目。我们希望这些工具将被用作一个基础，围绕它可以添加新的想法，并作为一个工具，将新方法与现有的方法进行比较。我们也希望这些工具的简单性能够让初学者尝试更高级的工具集，而不会被埋没在实施细节中。

**注意：尽管使用简单，Stable Baselines3（SB3）假定你有一些关于强化学习（RL）的知识。**你不应该在没有实践的情况下利用这个库。在这种情况下，我们在[文档](https://stable-baselines3.readthedocs.io/en/master/guide/rl.html)中提供了很好的资源来开始使用RL。

## Main Features

**对每种算法的性能进行了测试**（见其各自页面中的*结果*部分）。
you can take a look at the issues [#48](https://github.com/DLR-RM/stable-baselines3/issues/48) and [#49](https://github.com/DLR-RM/stable-baselines3/issues/49) for more details.


| **Features**                | **Stable-Baselines3** |
| --------------------------- | ----------------------|
| State of the art RL methods | :heavy_check_mark: |
| Documentation               | :heavy_check_mark: |
| Custom environments         | :heavy_check_mark: |
| Custom policies             | :heavy_check_mark: |
| Common interface            | :heavy_check_mark: |
| Ipython / Notebook friendly | :heavy_check_mark: |
| Tensorboard support         | :heavy_check_mark: |
| PEP8 code style             | :heavy_check_mark: |
| Custom callback             | :heavy_check_mark: |
| High code coverage          | :heavy_check_mark: |
| Type hints                  | :heavy_check_mark: |


### Planned features

Please take a look at the [Roadmap](https://github.com/DLR-RM/stable-baselines3/issues/1) and [Milestones](https://github.com/DLR-RM/stable-baselines3/milestones).

## Migration guide: from Stable-Baselines (SB2) to Stable-Baselines3 (SB3)

A migration guide from SB2 to SB3 can be found in the [documentation](https://stable-baselines3.readthedocs.io/en/master/guide/migration.html).

## Documentation

Documentation is available online: [https://stable-baselines3.readthedocs.io/](https://stable-baselines3.readthedocs.io/)

## RL Baselines3 Zoo: A Training Framework for Stable Baselines3 Reinforcement Learning Agents

[RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) is a training framework for Reinforcement Learning (RL).

它提供了用于训练、评估agent、调整超参数、绘制结果和录制视频的脚本。
此外，它还包括常见环境和RL算法的调整超参数集合，以及用这些设置训练的agent。

本repository的目标。

1. 提供一个简单的界面来训练和享受RL agent
2. 为不同的强化学习算法制定基准
3. 为每个环境和RL算法提供调整的超参数
4. 与训练过的agent一起享受乐趣!

Github repo: https://github.com/DLR-RM/rl-baselines3-zoo

Documentation: https://stable-baselines3.readthedocs.io/en/master/guide/rl_zoo.html

## SB3-Contrib: Experimental RL Features

We implement experimental features in a separate contrib repository: [SB3-Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)

This allows SB3 to maintain a stable and compact core, while still providing the latest features, like Truncated Quantile Critics (TQC) or Quantile Regression DQN (QR-DQN).

Documentation is available online: [https://sb3-contrib.readthedocs.io/](https://sb3-contrib.readthedocs.io/)


## Installation

**Note:** Stable-Baselines3 supports PyTorch 1.4+.

### Prerequisites
Stable Baselines3 requires python 3.6+.

#### Windows 10

To install stable-baselines on Windows, please look at the [documentation](https://stable-baselines3.readthedocs.io/en/master/guide/install.html#prerequisites).


### Install using pip
Install the Stable Baselines3 package:
```
pip install stable-baselines3[extra]
```
**Note:** Some shells such as Zsh require quotation marks around brackets, i.e. `pip install 'stable-baselines3[extra]'` ([More Info](https://stackoverflow.com/a/30539963)).

This includes an optional dependencies like Tensorboard, OpenCV or `atari-py` to train on atari games. If you do not need those, you can use:
```
pip install stable-baselines3
```

Please read the [documentation](https://stable-baselines3.readthedocs.io/) for more details and alternatives (from source, using docker).


## Example

Most of the library tries to follow a sklearn-like syntax for the Reinforcement Learning algorithms.

Here is a quick example of how to train and run PPO on a cartpole environment:
```python
import gym

from stable_baselines3 import PPO

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()
```

Or just train a model with a one liner if [the environment is registered in Gym](https://github.com/openai/gym/wiki/Environments) and if [the policy is registered](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html):

```python
from stable_baselines3 import PPO

model = PPO('MlpPolicy', 'CartPole-v1').learn(10000)
```

Please read the [documentation](https://stable-baselines3.readthedocs.io/) for more examples.


## Try it online with Colab Notebooks !

All the following examples can be executed online using Google colab notebooks:

- [Full Tutorial](https://github.com/araffin/rl-tutorial-jnrr19)
- [All Notebooks](https://github.com/Stable-Baselines-Team/rl-colab-notebooks/tree/sb3)
- [Getting Started](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_getting_started.ipynb)
- [Training, Saving, Loading](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/saving_loading_dqn.ipynb)
- [Multiprocessing](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/multiprocessing_rl.ipynb)
- [Monitor Training and Plotting](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/monitor_training.ipynb)
- [Atari Games](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/atari_games.ipynb)
- [RL Baselines Zoo](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/rl-baselines-zoo.ipynb)
- [PyBullet](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pybullet.ipynb)


## Implemented Algorithms

| **Name**         | **Recurrent**      | `Box`          | `Discrete`     | `MultiDiscrete` | `MultiBinary`  | **Multi Processing**              |
| ------------------- | ------------------ | ------------------ | ------------------ | ------------------- | ------------------ | --------------------------------- |
| A2C   | :x: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:                |
| DDPG  | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :x:                               |
| DQN   | :x: | :x: | :heavy_check_mark: | :x:                 | :x:                | :x:                               |
| HER   | :x: | :heavy_check_mark: | :heavy_check_mark: | :x:                 | :x:                | :x:                               |
| PPO   | :x: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:                |
| SAC   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :x:                               |
| TD3   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :x:                               |


Actions `gym.spaces`:
 * `Box`: A N-dimensional box that containes every point in the action space.
 * `Discrete`: A list of possible actions, where each timestep only one of the actions can be used.
 * `MultiDiscrete`: A list of possible actions, where each timestep only one action of each discrete set can be used.
 * `MultiBinary`: A list of possible actions, where each timestep any of the actions can be used in any combination.



## Testing the installation
All unit tests in stable baselines3 can be run using `pytest` runner:
```
pip install pytest pytest-cov
make pytest
```

You can also do a static type check using `pytype`:
```
pip install pytype
make type
```

Codestyle check with `flake8`:
```
pip install flake8
make lint
```

## 使用 Stable-Baselines3的项目

我们试图维护一个使用SB3的项目清单，在 [documentation](https://stable-baselines3.readthedocs.io/en/master/misc/projects.html),
please tell us when if you want your project to appear on this page ;)

## Citing the Project

To cite this repository in publications:

```bibtex
@misc{stable-baselines3,
  author = {Raffin, Antonin and Hill, Ashley and Ernestus, Maximilian and Gleave, Adam and Kanervisto, Anssi and Dormann, Noah},
  title = {Stable Baselines3},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DLR-RM/stable-baselines3}},
}
```

## Maintainers

Stable-Baselines3 is currently maintained by [Ashley Hill](https://github.com/hill-a) (aka @hill-a), [Antonin Raffin](https://araffin.github.io/) (aka [@araffin](https://github.com/araffin)), [Maximilian Ernestus](https://github.com/ernestum) (aka @ernestum), [Adam Gleave](https://github.com/adamgleave) (@AdamGleave) and [Anssi Kanervisto](https://github.com/Miffyli) (@Miffyli).

**Important Note: We do not do technical support, nor consulting** and don't answer personal questions per email.
Please post your question on the [RL Discord](https://discord.com/invite/xhfNqQv), [Reddit](https://www.reddit.com/r/reinforcementlearning/) or [Stack Overflow](https://stackoverflow.com/) in that case.


## How To Contribute

To any interested in making the baselines better, there is still some documentation that needs to be done.
If you want to contribute, please read [**CONTRIBUTING.md**](./CONTRIBUTING.md) guide first.

## Acknowledgments

The initial work to develop Stable Baselines3 was partially funded by the project *Reduced Complexity Models* from the *Helmholtz-Gemeinschaft Deutscher Forschungszentren*.

The original version, Stable Baselines, was created in the [robotics lab U2IS](http://u2is.ensta-paristech.fr/index.php?lang=en) ([INRIA Flowers](https://flowers.inria.fr/) team) at [ENSTA ParisTech](http://www.ensta-paristech.fr/en).


Logo credits: [L.M. Tenkes](https://www.instagram.com/lucillehue/)
