# %% [markdown]
# Source notebook: C1_W2_Lab05_Sklearn_GD_Soln.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Optional Lab: Linear Regression using Scikit-Learn

# %% [markdown]
# There is an open-source, commercially usable machine learning toolkit called [scikit-learn](https://scikit-learn.org/stable/index.html). This toolkit contains implementations of many of the algorithms that you will work with in this course.
# 

# %% [markdown]
# ## Goals
# In this lab you will:
# - Utilize  scikit-learn to implement linear regression using Gradient Descent

# %% [markdown]
# ## Tools
# You will utilize functions from scikit-learn as well as matplotlib and NumPy. 

# %%
# Notebook cell 5
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from lab_utils_multi import  load_house_data
from lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')

# %% [markdown]
# # Gradient Descent
# Scikit-learn has a gradient descent regression model [sklearn.linear_model.SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#examples-using-sklearn-linear-model-sgdregressor).  Like your previous implementation of gradient descent, this model performs best with normalized inputs. [sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) will perform z-score normalization as in a previous lab. Here it is referred to as 'standard score'.

# %% [markdown]
# ### Load the data set

# %%
# Notebook cell 8
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

# %% [markdown]
# ### Scale/normalize the training data

# %%
# Notebook cell 10
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

# %% [markdown]
# ### Create and fit the regression model

# %%
# Notebook cell 12
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

# %% [markdown]
# ### View parameters
# Note, the parameters are associated with the *normalized* input data. The fit parameters are very close to those found in the previous lab with this data.

# %%
# Notebook cell 14
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")
print( "model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")

# %% [markdown]
# ### Make predictions
# Predict the targets of the training data. Use both the `predict` routine and compute using $w$ and $b$.

# %%
# Notebook cell 16
# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)
# make a prediction using w,b. 
y_pred = np.dot(X_norm, w_norm) + b_norm  
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")

# %% [markdown]
# ### Plot Results
# Let's plot the predictions versus the target values.

# %%
# Notebook cell 18
# plot predictions and targets vs original features    
fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],y_pred,color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()

# %% [markdown]
# ## Congratulations!
# In this lab you:
# - utilized an open-source machine learning toolkit, scikit-learn
# - implemented linear regression using gradient descent and feature normalization from that toolkit

# %%
# Notebook cell 20

