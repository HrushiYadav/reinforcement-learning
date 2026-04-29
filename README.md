# Reinforcement Learning and Neural Networks

implementations of core deep learning and reinforcement learning algorithms using PyTorch and TensorFlow.

## projects

### neural networks

| project | framework | task | dataset | result |
|---------|-----------|------|---------|--------|
| **ANN** | TensorFlow/Keras | bank customer churn prediction | Churn_Modelling (10k rows) | 86.4% accuracy |
| **CNN** | TensorFlow/Keras | image classification (cat vs dog) | 8000 train + 2000 test images | ~80% val accuracy, 96% train |

### reinforcement learning

| project | algorithm | environment | framework | result |
|---------|-----------|-------------|-----------|--------|
| **Lunar Landing** | Deep Q-Learning (DQN) | LunarLander-v3 (Gymnasium) | PyTorch | solved in 385 episodes (avg reward 200+) |
| **Pac-Man** | Deep Convolutional Q-Learning (DCQN) | MsPacmanNoFrameskip-v0 | PyTorch | solved in 159 episodes (avg score 500+) |
| **Kung Fu** | A3C (Asynchronous Advantage Actor-Critic) | KungFuMasterNoFrameskip-v0 | PyTorch | avg reward 1190 after 3000 steps |

## what's inside

### ANN (`ann.ipynb`)
- binary classification with fully connected network
- data preprocessing: label encoding, one-hot encoding, feature scaling
- 2 hidden layers (6 units each), sigmoid output
- trained 100 epochs, evaluated with confusion matrix

### CNN (`cnn.ipynb`)
- image classification with convolutional layers
- data augmentation (shear, zoom, horizontal flip)
- 2 conv layers + pooling + dense layers
- 30 epochs training with validation

### Deep Q-Learning for Lunar Landing (`Lunar_Landing.ipynb`)
- custom DQN agent with experience replay buffer
- epsilon-greedy exploration with decay
- soft update between local and target networks
- solved OpenAI Gymnasium LunarLander in 385 episodes

### Deep Convolutional Q-Learning for Pac-Man (`Deep_Convolutional_Q_Learning_for_Pac_Man.ipynb`)
- 4-layer CNN (with batch normalization) as Q-network
- frame preprocessing: resize to 128x128, normalize
- experience replay with deque-based memory
- solved Ms. Pac-Man in 159 episodes

### A3C for Kung Fu (`A3C_for_Kung_Fu.ipynb`)
- actor-critic architecture with shared CNN backbone
- 3 conv layers + separate actor (policy) and critic (value) heads
- parallel environment training (10 environments simultaneously)
- custom Atari preprocessing: grayscale, resize to 42x42, frame stacking (4 frames)
- entropy regularization for exploration

## tech stack

- **PyTorch** for RL agents (DQN, DCQN, A3C)
- **TensorFlow/Keras** for supervised learning (ANN, CNN)
- **Gymnasium** (OpenAI) for RL environments
- **ALE** (Arcade Learning Environment) for Atari games
- **NumPy, Pandas, scikit-learn** for data processing
