# 🐦 Flappy Bird AI — Deep Q-Network (DQN)

A Reinforcement Learning project that trains an AI agent to play **Flappy Bird** using a **Deep Q-Network (DQN)** with Experience Replay and a Target Network — built with PyTorch and the `flappy_bird_gymnasium` environment.

---

## 📌 Overview

This project implements a classic DQN agent (inspired by DeepMind's Atari paper) on the Flappy Bird environment. The agent learns purely from trial and error — no hard-coded rules, just rewards. It starts off crashing immediately and, after training, navigates pipes indefinitely.

---

## 🧠 How It Works

The agent uses the **Deep Q-Learning** algorithm:

1. **Observe** the current game state (bird position, velocity, pipe distances, etc.)
2. **Choose** an action — flap or do nothing — via an **ε-greedy policy**
3. **Store** the experience `(state, action, next_state, reward, done)` in a **Replay Memory**
4. **Sample** a random mini-batch from memory and **optimize** the policy network
5. **Periodically sync** the target network to stabilize training

---

## 🗂️ Project Structure

```
Flappy-Bird-AI/
│
├── agent.py              # DQN Agent — training loop, epsilon decay, model saving
├── dqn.py                # Neural network architecture (Linear → ReLU → Linear)
├── experience_replay.py  # Replay Memory using collections.deque
├── flappy_bird_game.py   # Game environment wrapper
└── parameters.yaml       # Hyperparameter configuration
```

---

## 🏗️ Architecture

### Neural Network (`dqn.py`)

A simple fully-connected feed-forward network:

```
Input (state_dim)  →  Linear  →  ReLU  →  Linear  →  Output (action_dim)
                         Hidden: 256 units
```

### Replay Memory (`experience_replay.py`)

A fixed-size circular buffer (`deque`) that stores past transitions. Random sampling from this buffer breaks temporal correlations and stabilizes training.

### Agent (`agent.py`)

- **Policy Network** — the network being trained
- **Target Network** — a periodically-synced copy used to compute stable Q-targets
- **Optimizer** — Adam
- **Loss** — Mean Squared Error (MSE) between predicted Q-values and Bellman targets

---

## ⚙️ Hyperparameters

Defined in `parameters.yaml`:

| Parameter              | Value       | Description                                          |
|------------------------|-------------|------------------------------------------------------|
| `epsilon_init`         | `1.0`       | Starting exploration rate (100% random)              |
| `epsilon_min`          | `0.05`      | Minimum exploration rate                             |
| `epsilon_decay`        | `0.9995`    | Multiplicative decay per episode                     |
| `replay_memory_size`   | **`100000`** | Max transitions stored in replay buffer *(updated)*  |
| `min_batch_size`       | `32`        | Mini-batch size for gradient updates                 |
| `network_sync_rate`    | `0.001`     | Frequency of target network sync                     |
| `alpha` (lr)           | `0.001`     | Adam learning rate                                   |
| `gamma`                | `0.99`      | Discount factor for future rewards                   |
| `reward_threshold`     | `100000`    | Max reward per episode before termination            |

> ✅ **Trained on a replay memory size of 10,000** — enough to provide diverse experience batches while keeping memory usage efficient.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/sriKritarth/Flappy-Bird-AI.git
cd Flappy-Bird-AI
```

### 2. Install Dependencies

```bash
pip install torch gymnasium flappy-bird-gymnasium pyyaml
```

### 3. Train the Agent

```bash
python agent.py flappybirdv0 --train
```

### 4. Watch the Trained Agent Play

```bash
python agent.py flappybirdv0
```

> The trained model is automatically saved to `runs/flappybirdv0.pt` whenever a new best reward is achieved during training.

---

## 📊 Training Details

- **Exploration** starts at 100% random actions and decays toward 5% as the agent gains experience.
- **Experience Replay** with a buffer of **10,000** transitions prevents the agent from overfitting to recent events.
- **Target Network** syncing prevents oscillating Q-value targets, a common instability in DQN training.
- **Best model checkpointing** — the `.pt` model file is only overwritten when the agent achieves a new personal best reward.
- Training logs are written to `runs/flappybirdv0.log`.

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **PyTorch** — neural network & optimization
- **Gymnasium** — RL environment interface
- **flappy-bird-gymnasium** — Flappy Bird environment
- **PyYAML** — hyperparameter management

---

## 📁 Outputs

| File                        | Description                          |
|-----------------------------|--------------------------------------|
| `runs/flappybirdv0.pt`      | Saved model weights (best episode)   |
| `runs/flappybirdv0.log`     | Training log with best reward history|

---

## 🤝 Contributing

Pull requests are welcome! If you'd like to experiment with different architectures, reward shaping, or environments, feel free to fork and open a PR.

---

## 📄 License

This project is open source. See the repository for license details.

---

*Built with ❤️ using Deep Reinforcement Learning*
