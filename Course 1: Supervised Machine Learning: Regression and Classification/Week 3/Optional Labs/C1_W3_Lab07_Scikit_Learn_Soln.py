# %% [markdown]
# Source notebook: C1_W3_Lab07_Scikit_Learn_Soln.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Ungraded Lab:  Logistic Regression using Scikit-Learn
# 
# 

# %% [markdown]
# ## Goals
# In this lab you will:
# -  Train a logistic regression model using scikit-learn.

# %% [markdown]
# ## Dataset 
# Let's start with the same dataset as before.

# %%
# Notebook cell 4
import numpy as np

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

# %% [markdown]
# ## Fit the model
# 
# The code below imports the [logistic regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression) from scikit-learn. You can fit this model on the training data by calling `fit` function.

# %%
# Notebook cell 6
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X, y)

# %% [markdown]
# ## Make Predictions
# 
# You can see the predictions made by this model by calling the `predict` function.

# %%
# Notebook cell 8
y_pred = lr_model.predict(X)

print("Prediction on training set:", y_pred)

# %% [markdown]
# ## Calculate accuracy
# 
# You can calculate this accuracy of this model by calling the `score` function.

# %%
# Notebook cell 10
print("Accuracy on training set:", lr_model.score(X, y))
