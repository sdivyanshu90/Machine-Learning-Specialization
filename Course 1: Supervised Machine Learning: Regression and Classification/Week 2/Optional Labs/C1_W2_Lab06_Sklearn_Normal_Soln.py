# %% [markdown]
# Source notebook: C1_W2_Lab06_Sklearn_Normal_Soln.ipynb
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
# - Utilize  scikit-learn to implement linear regression using a close form solution based on the normal equation

# %% [markdown]
# ## Tools
# You will utilize functions from scikit-learn as well as matplotlib and NumPy. 

# %%
# Notebook cell 5
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from lab_utils_multi import load_house_data
plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)

# %% [markdown]
# <a name="toc_40291_2"></a>
# # Linear Regression, closed-form solution
# Scikit-learn has the [linear regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression) which implements a closed-form linear regression.
# 
# Let's use the data from the early labs - a house with 1000 square feet sold for \\$300,000 and a house with 2000 square feet sold for \\$500,000.
# 
# | Size (1000 sqft)     | Price (1000s of dollars) |
# | ----------------| ------------------------ |
# | 1               | 300                      |
# | 2               | 500                      |

# %% [markdown]
# ### Load the data set

# %%
# Notebook cell 8
X_train = np.array([1.0, 2.0])   #features
y_train = np.array([300, 500])   #target value

# %% [markdown]
# ### Create and fit the model
# The code below performs regression using scikit-learn. 
# The first step creates a regression object.  
# The second step utilizes one of the methods associated with the object, `fit`. This performs regression, fitting the parameters to the input data. The toolkit expects a two-dimensional X matrix.

# %%
# Notebook cell 10
linear_model = LinearRegression()
#X must be a 2-D Matrix
linear_model.fit(X_train.reshape(-1, 1), y_train)

# %% [markdown]
# ### View Parameters 
# The $\mathbf{w}$ and $\mathbf{b}$ parameters are referred to as 'coefficients' and 'intercept' in scikit-learn.

# %%
# Notebook cell 12
b = linear_model.intercept_
w = linear_model.coef_
print(f"w = {w:}, b = {b:0.2f}")
print(f"'manual' prediction: f_wb = wx+b : {1200*w + b}")

# %% [markdown]
# ### Make Predictions
# 
# Calling the `predict` function generates predictions.

# %%
# Notebook cell 14
y_pred = linear_model.predict(X_train.reshape(-1, 1))

print("Prediction on training set:", y_pred)

X_test = np.array([[1200]])
print(f"Prediction for 1200 sqft house: ${linear_model.predict(X_test)[0]:0.2f}")

# %% [markdown]
# ## Second Example
# The second example is from an earlier lab with multiple features. The final parameter values and predictions are very close to the results from the un-normalized 'long-run' from that lab. That un-normalized run took hours to produce results, while this is nearly instantaneous. The closed-form solution work well on smaller data sets such as these but can be computationally demanding on larger data sets. 
# >The closed-form solution does not require normalization.

# %%
# Notebook cell 16
# load the dataset
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

# %%
# Notebook cell 17
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# %%
# Notebook cell 18
b = linear_model.intercept_
w = linear_model.coef_
print(f"w = {w:}, b = {b:0.2f}")

# %%
# Notebook cell 19
print(f"Prediction on training set:\n {linear_model.predict(X_train)[:4]}" )
print(f"prediction using w,b:\n {(X_train @ w + b)[:4]}")
print(f"Target values \n {y_train[:4]}")

x_house = np.array([1200, 3,1, 40]).reshape(-1,4)
x_house_predict = linear_model.predict(x_house)[0]
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.2f}")

# %% [markdown]
# ## Congratulations!
# In this lab you:
# - utilized an open-source machine learning toolkit, scikit-learn
# - implemented linear regression using a close-form solution from that toolkit

# %%
# Notebook cell 21

