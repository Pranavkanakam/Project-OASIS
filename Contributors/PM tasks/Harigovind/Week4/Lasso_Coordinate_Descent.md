# Lasso Regression using Coordinate Descent

Unlike Ridge Regression, Lasso Regression does not have a closed-form analytical solution because of the presence of the **L₁ regularization term**.

The objective function minimized by Lasso Regression is:

$$
J(\beta)=||y-X\beta||^2+\lambda\sum_{j=1}^{p}|\beta_j|
$$

The absolute value term is not differentiable at zero, making it impossible to directly solve for the coefficients by setting the derivative to zero. Therefore, the coefficients are estimated iteratively using the **Coordinate Descent algorithm**.

---

## Coordinate Descent

In Coordinate Descent, one coefficient is updated at a time while keeping all other coefficients fixed.

Suppose the algorithm is updating the coefficient $\beta_j$. The contribution of this feature is temporarily removed from the prediction, and the residual is calculated as:

$$
r = y-(X\beta-x_j\beta_j)
$$

where $x_j$ represents the $j^{th}$ feature column.

This residual represents the part of the target that is not explained by the remaining features, allowing the current coefficient to be optimized independently.

To simplify the optimization, two quantities are introduced:

### Correlation with residual

$$
\rho = x_j^T r
$$

This measures how strongly the current feature is correlated with the remaining residual.

### Feature magnitude

$$
z=x_j^Tx_j
$$

This represents the squared Euclidean norm of the feature column.

Using these quantities, the optimization problem for a single coefficient becomes:

$$
J(\beta_j)=z\beta_j^2-2\rho\beta_j+\lambda|\beta_j|
$$

---

## Soft Thresholding Update

Since the L₁ penalty contains an absolute value term, the optimization is solved by considering three cases.

### Case 1: $\beta_j > 0$

The optimal update is:

$$
\beta_j=\frac{2\rho-\lambda}{2z}
$$


### Case 2: $\beta_j < 0$

The optimal update is:

$$
\beta_j=\frac{2\rho+\lambda}{2z}
$$


### Case 3: Minimum occurs at zero

If neither of the above conditions is satisfied:

$$
\beta_j=0
$$

These three cases together form the **soft-thresholding operator**, which is the key component of Lasso Regression.

---

## Sparsity and Feature Selection

Unlike Ridge Regression, which only shrinks coefficients towards zero, Lasso can force coefficients to become **exactly zero**.

This creates a sparse coefficient vector and enables automatic feature selection by removing insignificant predictors while retaining the important ones.

For example, during the experiment, increasing $\lambda$ caused some coefficients to become zero, demonstrating the feature selection capability of Lasso.

---

## Convergence

The coordinate descent algorithm repeatedly updates each coefficient using the soft-thresholding rule.

After each complete iteration, the change in the coefficient vector is measured:

$$
||\beta^{(new)}-\beta^{(old)}||
$$

If this change becomes smaller than a predefined tolerance value, the algorithm is considered converged and the iterations are terminated.

The final sparse coefficient vector obtained after convergence is then used for prediction.
