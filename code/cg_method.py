import numpy as np
import matplotlib.pyplot as plt


def f(x, A):
    """
    Quadratic function: f(x) = 0.5 * x^T A x, with b=0 for simplicity.
    """
    return 0.5 * (x @ (A @ x))


def grad_f(x, A):
    """
    Gradient of the above function: grad f(x) = A x (since b=0).
    This is also the negative residual (-r) when b=0.
    """
    return A @ x


def gradient_descent_fixed_step(A, x0, alpha=0.2, max_iter=150, tol=1e-14):
    """
    Gradient Descent using a fixed step size alpha.
    Returns an array of all iterates [x0, x1, x2, ..., x_max_iter].
    """
    x = x0.copy()
    points = [x.copy()]
    for _ in range(max_iter):
        g = grad_f(x, A)
        # Optional: Stop if gradient is small (equivalent to residual small)
        if np.linalg.norm(g) < tol:
             break
        x = x - alpha * g # Update uses negative gradient = residual
        points.append(x.copy())
    return np.array(points)


def steepest_descent_exact_line_search(A, x0, max_iter=150, tol=1e-14):
    """
    Method of Steepest Descent using an exact line search.
    For f(x)=0.5 * x^T A x, b=0 => gradient g=A x, search direction p = -g.
    alpha_k = (g_k^T g_k) / (g_k^T A g_k) --- using gradient g notation

    Returns an array of all iterates [x0, x1, x2, ...].
    """
    x = x0.copy()
    points = [x.copy()]
    g = grad_f(x, A) # Initial gradient

    for _ in range(max_iter):
        if np.linalg.norm(g) < tol:
             break

        Ag = A @ g
        alpha = (g @ g) / (g @ Ag) # Optimal step size for this quadratic
        x = x - alpha * g         # Update step in negative gradient direction
        points.append(x.copy())

        g = grad_f(x, A) # Calculate new gradient for next iteration

    return np.array(points)


def conjugate_gradient_correct(A, x0, max_iter=150, tol=1e-14):
    """
    Conjugate Gradient for f(x)=0.5 * x^T A x, b=0 => negative gradient is r=-A x.
    *IMPORTANT*: Make sure r_next = r - alpha * (A p).

    Returns an array of all iterates [x0, x1, x2, ...].
    """
    x = x0.copy()
    points = [x.copy()]

    # initial residual = negative gradient
    r = -grad_f(x, A) # r = -Ax
    p = r.copy()

    # Check initial residual norm before loop
    if np.linalg.norm(r) < tol:
        return np.array(points)

    # Calculate initial rTr to avoid redundant calculation inside loop
    rTr = r @ r

    for _ in range(max_iter):
        Ap = A @ p
        alpha = rTr / (p @ Ap) # Use stored rTr
        x = x + alpha * p
        points.append(x.copy())

        r_next = r - alpha * Ap  # Update residual efficiently
        if np.linalg.norm(r_next) < tol:
            break

        rTr_next = r_next @ r_next
        beta = rTr_next / rTr # Use stored rTr
        p = r_next + beta * p
        r = r_next
        rTr = rTr_next # Update rTr for next iteration

    return np.array(points)


import numpy as np
import matplotlib.pyplot as plt

# ... (Keep your f, grad_f, and optimization functions as they are) ...

def plot_iterations_on_own_figure(A, points, title, x_lim_orig, y_lim_orig, filename=None):
    """
    Plots iterations, ensuring an equal aspect ratio without cutting off data.
    x_lim_orig and y_lim_orig are the initial desired bounding box.
    """
    x_min_orig, x_max_orig = x_lim_orig
    y_min_orig, y_max_orig = y_lim_orig

    # Calculate the range and center of the original limits
    dx = x_max_orig - x_min_orig
    dy = y_max_orig - y_min_orig
    center_x = (x_min_orig + x_max_orig) / 2
    center_y = (y_min_orig + y_max_orig) / 2

    # Determine the maximum range needed, add a margin (e.g., 10-15%)
    max_range = max(dx, dy) * 1.15 # Added 15% margin

    # Calculate new limits that define a square region centered on the data
    x_min_adj = center_x - max_range / 2
    x_max_adj = center_x + max_range / 2
    y_min_adj = center_y - max_range / 2
    y_max_adj = center_y + max_range / 2

    # Build a meshgrid over the adjusted square region.
    grid_size = 200
    X, Y = np.meshgrid(
        np.linspace(x_min_adj, x_max_adj, grid_size),
        np.linspace(y_min_adj, y_max_adj, grid_size)
    )

    # Compute f(x, A) over the grid.
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            v = np.array([X[i, j], Y[i, j]])
            Z[i, j] = f(v, A)

    # Create the figure; plot contours and iteration path.
    plt.figure(figsize=(7, 7)) # Use a square figure for better results with equal aspect
    plt.contour(X, Y, Z, levels=30, colors='gray', linestyles='dashed')
    plt.plot(points[:, 0], points[:, 1], 'ko-', lw=2, markersize=5)
    plt.plot(points[0, 0], points[0, 1], 'bo', markersize=8, label='Start')
    plt.plot(points[-1, 0], points[-1, 1], color='gray', marker='X', markersize=10, label='End')

    # Apply settings
    plt.xticks([])
    plt.yticks([])
    # plt.title(title) # Keep title optional/commented if desired
    plt.grid(True, linestyle=':')

    # Set the aspect ratio to equal FIRST
    plt.axis('equal')

    # THEN set the calculated square limits
    plt.xlim(x_min_adj, x_max_adj)
    plt.ylim(y_min_adj, y_max_adj)

    # plt.legend() # Optional

    plt.tight_layout() # Adjust layout
    if filename is not None:
        plt.savefig(filename, dpi=300)

    plt.show()
    plt.close()


def main():
    # Problem Setup:
    A = np.array([
        [5.0, 3.5],
        [3.5, 5.0]
    ], dtype=float)
    x0 = np.array([2.0, -1.0])
    max_iters = 150
    tolerance = 1e-6

    # Run the methods
    print("Running Gradient Descent (fixed step)...")
    gd_points = gradient_descent_fixed_step(A, x0, alpha=0.1, max_iter=max_iters, tol=tolerance)
    print("Running Steepest Descent (exact line search)...")
    sd_els_points = steepest_descent_exact_line_search(A, x0, max_iter=max_iters, tol=tolerance)
    print("Running Conjugate Gradient...")
    cg_points = conjugate_gradient_correct(A, x0, max_iter=max_iters, tol=tolerance)

    # Calculate iterations (number of steps = number of points - 1)
    gd_iters = len(gd_points) - 1
    sd_els_iters = len(sd_els_points) - 1
    cg_iters = len(cg_points) - 1

    print("\n--- Iterations ---")
    print(f"Gradient Descent (fixed step):         {gd_iters} (stopped by {'max_iter' if gd_iters == max_iters else 'tolerance'})")
    print(f"Steepest Descent (exact line search): {sd_els_iters} (stopped by {'max_iter' if sd_els_iters == max_iters else 'tolerance'})")
    print(f"Conjugate Gradient:                    {cg_iters} (stopped by {'max_iter' if cg_iters == max_iters else 'tolerance'})")
    print("--------------------\n")

    # Compute a global bounding box from all iteration sets.
    # Include x0 and the origin (0,0) which is the minimum for this function
    all_points_for_bounds = np.vstack((gd_points, sd_els_points, cg_points, x0.reshape(1,-1), np.array([[0.0, 0.0]])))
    x_min_global = all_points_for_bounds[:, 0].min()
    x_max_global = all_points_for_bounds[:, 0].max()
    y_min_global = all_points_for_bounds[:, 1].min()
    y_max_global = all_points_for_bounds[:, 1].max()

    # No margin needed here, it will be added in the plot function
    x_lim_global = (x_min_global, x_max_global)
    y_lim_global = (y_min_global, y_max_global)

    # Pass the original global limits to the plotting function
    plot_iterations_on_own_figure(A, gd_points, title=f"Gradient Descent (Fixed Step α=0.1)\n{gd_iters} Iterations", x_lim_orig=x_lim_global, y_lim_orig=y_lim_global, filename="gradient_descent.png")
    plot_iterations_on_own_figure(A, sd_els_points, title=f"Steepest Descent (Exact Line Search)\n{sd_els_iters} Iterations", x_lim_orig=x_lim_global, y_lim_orig=y_lim_global, filename="steepest_descent.png")
    plot_iterations_on_own_figure(A, cg_points, title=f"Conjugate Gradient\n{cg_iters} Iterations", x_lim_orig=x_lim_global, y_lim_orig=y_lim_global, filename="conjugate_gradient.png")


if __name__ == "__main__":
    main()
