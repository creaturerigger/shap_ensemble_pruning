import numpy as np
from scipy.linalg import cholesky
from scipy.optimize import minimize

def nearest_spd(A):
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = V.T @ np.diag(s) @ V
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not is_spd(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return A3

def is_spd(A):
    try:
        _ = cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

def objective(x, L, q, lambda_, u):
    return x[-1] + lambda_ * u

def constraint_inf_norm(x):
    return 1.0 - np.max(np.abs(x))

def constraint_linear(x, L):
    return (-2 * L @ x[:-1]).min()

def sip_pruning(model_preds, lambda_=0.1, u=1.0, epsilon=1e-5):
    n_models = model_preds.shape[0]

    # Clean and standardize model_preds
    model_preds = np.nan_to_num(model_preds, nan=0.0, posinf=1e6, neginf=-1e6)
    model_preds = np.clip(model_preds, -1e3, 1e3)

    # Prepare I matrix for Q = I.T @ I
    reshaped_preds = model_preds.transpose(0, 2, 1).reshape(n_models, -1)
    I = reshaped_preds.T

    # Compute SPD Q matrix safely
    Q = I.T @ I + np.eye(I.shape[1]) * epsilon
    Q = nearest_spd(Q)

    try:
        L = cholesky(Q + np.eye(Q.shape[0]) * epsilon)
    except np.linalg.LinAlgError:
        print("⚠️ Cholesky decomposition failed. Using fallback uniform weights.")
        return np.ones(n_models) / n_models

    z = model_preds.reshape(n_models, -1).T
    q = np.mean(z, axis=0)

    # Initial guess for weights + slack variable
    x0 = np.ones(n_models + 1) / (n_models + 1)

    # Constraints
    constraints = [
        {"type": "ineq", "fun": constraint_inf_norm},
        {"type": "ineq", "fun": lambda x: constraint_linear(x, L)}
    ]

    result = minimize(
        objective,
        x0,
        args=(L, q, lambda_, u),
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-6}
    )

    if not result.success:
        print(f"⚠️ Optimization warning: {result.message}")
        return np.ones(n_models) / n_models

    weights = np.clip(result.x[:-1], 0, None)
    total = weights.sum()
    if total == 0 or not np.isfinite(total):
        return np.ones(n_models) / n_models
    nonzero_weights = np.count_nonzero(weights > 1e-6)
    print(f"✅ SIP used {nonzero_weights} out of {n_models} models.")
    return weights / total
