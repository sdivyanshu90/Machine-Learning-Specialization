# %% [markdown]
# Source notebook: State-action value function example.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # State Action Value Function Example
# 
# In this Jupyter notebook, you can modify the mars rover example to see how the values of Q(s,a) will change depending on the rewards and discount factor changing.

# %%
# Notebook cell 2
import numpy as np
from utils import *

# %%
# Notebook cell 3
# Do not modify
num_states = 6
num_actions = 2

# %%
# Notebook cell 4
terminal_left_reward = 100
terminal_right_reward = 40
each_step_reward = 0

# Discount factor
gamma = 0.5

# Probability of going in the wrong direction
misstep_prob = 0.4

# %%
# Notebook cell 5
generate_visualization(terminal_left_reward, terminal_right_reward, each_step_reward, gamma, misstep_prob)

# %%
# Notebook cell 6

