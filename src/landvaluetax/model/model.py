import matplotlib.pyplot as plt
import numpy as np

###############################
# ONE PERIOD MODEL FUNCTIONS (model 4)
###############################


def add_build(site_build_matrix_original, J):
    """Parameters
    ----------
    site_build_matrix_original : np.ndarray
        2‑D array of 0/1 indicating whether each cell is developed.
    J : int | tuple[int, int]
        Index of the cell you want to flip to 1.
        • If the grid is flat (shape = (N²,)), pass an int.
        • If it’s 2‑D (shape = (N,N)), pass a tuple (row, col).

    Returns:
    -------
    new_matrix : np.ndarray
        A copy of the original matrix with the entry at J set to 1.
    """
    new_matrix = site_build_matrix_original.copy()  # leave original untouched
    new_matrix[J] = 1  # flip the chosen cell
    return new_matrix


def calculate_site_values(n):
    """S_i = n_i + (1/|M(i)|) * sum_{j in M(i)} n_j   (Moore neighbors)"""
    I, J = n.shape
    S = np.zeros_like(n, dtype=float)
    for i in range(I):
        for j in range(J):
            # grab the 3x3 block around (i,j)
            block = n[max(i - 1, 0) : min(i + 2, I), max(j - 1, 0) : min(j + 2, J)]
            neigh_sum = block.sum() - n[i, j]
            neigh_cnt = block.size - 1
            S[i, j] = neigh_sum / neigh_cnt if neigh_cnt else 0.0
    return S


def calculate_rents(people_distribution_matrix, wage):
    """Compute rent matrix R and common utility level U.

    Parameters
    ----------
    n : np.ndarray (float, 2‑D)
        People distribution.  n.sum() must equal 1.
    w : float
        Upper bound on rent each household can pay (the exogenous wage).

    Returns:
    -------
    R : np.ndarray (float, same shape as n)
        Rent in each cell.  Satisfies 0 ≤ R_i ≤ w.
    U : float
        Common flow utility enjoyed by all occupied cells.
    """
    # --- site values from current occupancy ----------------------------------
    S = calculate_site_values(people_distribution_matrix)

    # --- which cells are actually occupied? ----------------------------------
    occ = people_distribution_matrix > 0
    if not np.any(occ):  # nobody lives anywhere → all rents zero
        return np.zeros_like(people_distribution_matrix, dtype=float), 0.0

    S_occ = S[occ]

    # Feasibility interval for U:  max_i (S_i - w) ≤ U ≤ min_i S_i  (occupied)
    lower_bound = np.max(S_occ - wage)
    upper_bound = np.min(S_occ)
    if lower_bound > upper_bound:
        raise ValueError("Wage ceiling w is too low to support current occupancy.")

    # Choose any U in [lower_bound, upper_bound]
    U = lower_bound

    # --- rents ---------------------------------------------------------------
    R = np.zeros_like(S)
    R[occ] = S[occ] - U

    if np.any(R < 0):
        raise ValueError("Negative rent encountered in calculate_rents.")
    # For occupied cells we should have exactly S_i - R_i = U (check):
    np.testing.assert_allclose(S[occ] - R[occ], U)

    return R


def calculate_people_distribution(site_matrix_built):
    """Parameters
    ----------
    site_matrix_built : np.ndarray (2‑D, dtype int/bool)
        1  → site has been developed and is available for occupancy
        0  → still vacant (or outside the choice set)

    Returns:
    -------
    n : np.ndarray (float, same shape)
        Household distribution.  Sum of all entries is exactly 1.
        Vacant cells get 0.
    """
    built_mask = site_matrix_built > 0
    built_count = built_mask.sum()

    # No developed site?  Everyone stays in the outside option.
    if built_count == 0:
        return np.zeros_like(site_matrix_built, dtype=float)

    # Uniform allocation across built sites
    n = np.zeros_like(site_matrix_built, dtype=float)
    n[built_mask] = 1.0 / built_count

    return n


def decision_owner(
    owner_cell_index,
    site_value_matrix_original,
    site_build_matrix_original,
    site_value_matrix_new,
    parameters,
):
    """Decide whether a land‑owner builds.

    Parameters
    ----------
    owner_cell_index : int | tuple[int, int]
        Index of the cell (flat or 2‑D).
    site_value_matrix : np.ndarray
        Current site values S_i  (same shape as the grid).
    rent_matrix : np.ndarray
        Current equilibrium rents R_i for each cell (same shape).
    parameters : dict

    Returns:
    -------
    int
        1 → build, 0 → keep vacant.
    """
    p0_land = site_value_matrix_original[owner_cell_index]  # P⁰ᵛ = S_i

    # Case 1: I do not build:

    p1_land_vacant = site_value_matrix_new[owner_cell_index]
    return_vacant = (
        parameters["beta"] * p1_land_vacant * (1 - parameters["tau"]) / p0_land
    )
    # print(f"Return vacant: {return_vacant}, P0 land: {p0_land}, P1 land vacant: {p1_land_vacant} \n ")

    # Case 2: I build:
    new_site_build_matrix = add_build(site_build_matrix_original, owner_cell_index)
    new_pop_matrix = calculate_people_distribution(new_site_build_matrix)
    p1_land_build = calculate_site_values(new_pop_matrix)[owner_cell_index]
    rent = calculate_rents(new_pop_matrix, parameters["wage"])[owner_cell_index]

    return_build = (
        parameters["beta"]
        * (
            p1_land_build * (1 - parameters["tau"])
            + rent
            - parameters["delta"] * parameters["capex"]
        )
        / (p0_land + parameters["capex"])
    )

    # print(f"Return build: {return_build}, P1 land build: {p1_land_build}, Rent: {rent}, Capex: {parameters['capex']} \n \n")

    return int(return_build >= return_vacant)


def find_equilibrium(parameters, init_site_value):
    """Monotone best‑response loop for the 1‑period land‑development game.

    Parameters
    ----------
    grid_shape      : tuple[int, int]
        (rows, cols) of the lattice, e.g. (10, 10).
    parameters      : dict
        Model parameters; must contain keys
           "beta", "capex", "delta", "wage".
    init_site_value : np.ndarray
        Site‑value matrix *before* any building takes place (P⁰ᵛ = S_i at t=0).
        Shape must equal grid_shape.

    Returns:
    -------
    built_matrix    : np.ndarray (int, 0/1)
        Equilibrium set of developed cells (1 = built).
    people_matrix   : np.ndarray (float)
        Final household distribution (sums to 1).
    rent_matrix     : np.ndarray (float)
        Equilibrium rents.
    site_values     : np.ndarray (float)
        Site values under the final occupancy.
    """
    # --- INITIALISE ----------------------------------------------------------
    built_matrix = np.zeros((parameters["G"], parameters["G"]), dtype=int)  # B⁽⁰⁾ = ∅

    # Pre‑compute a flat index list for easy iteration
    cell_indices = [
        (i, j) for i in range(parameters["G"]) for j in range(parameters["G"])
    ]

    i = 0
    while True:
        i += 1
        build_this_round = False
        # ---- STAGE 0c: household allocation and rents ----------------------

        people_matrix = calculate_people_distribution(built_matrix)
        site_values = calculate_site_values(people_matrix)
        rent_matrix = calculate_rents(people_matrix, parameters["wage"])

        # ---- PROFIT TEST for still‑vacant owners ---------------------------
        newly_profitable = []

        for idx in cell_indices:
            # percent_checked = (cell_indices.index(idx) + 1) / len(cell_indices) * 100
            # print(f"Checking cell {idx}: {percent_checked:.1f}%")

            if built_matrix[idx] == 1:  # already built → skip
                continue
            build_flag = decision_owner(
                owner_cell_index=idx,
                site_value_matrix_original=init_site_value,
                site_build_matrix_original=built_matrix,
                site_value_matrix_new=site_values,
                parameters=parameters,
            )

            if build_flag == 1:
                newly_profitable.append(idx)
                build_this_round = True
                break

        # ---- MONOTONE UPDATE + TERMINATION CHECK ---------------------------
        if not newly_profitable:  # no switchers → equilibrium
            break

        for idx in newly_profitable:  # update B
            built_matrix[idx] = 1

    # one last refresh to return consistent outputs
    people_matrix = calculate_people_distribution(built_matrix)
    site_values = calculate_site_values(people_matrix)
    rent_matrix = calculate_rents(people_matrix, parameters["wage"])

    return built_matrix, people_matrix, rent_matrix, site_values


###############################
# DYNAMIC MODEL II FUNCTIONS
###############################


def init_population_matrix(shape=(10, 10), seed=42):
    """Return a non-negative matrix of given shape whose entries sum to 1."""
    rng = np.random.default_rng(seed)
    A = rng.random(shape)
    return A / A.sum()


def plot_pop_matrix(M, title="Population heatmap"):
    plt.imshow(M, cmap="viridis", origin="lower")
    plt.colorbar(label="Share of mass")
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.show()


def relocate(n_prev, U_t, w, cap=1.0):
    """Given last period distribution (for path-dependence if any),
    target utility U_t, and wage ceiling w, return (n_t, r_t, S_t).
    """
    # 1. compute S_t from n_prev (or from current n if contemporaneous)
    S = calculate_site_values(n_prev)  # use your function
    # 2. feasible rent = min{S - U_t, w}, but rent cannot be negative
    r = np.clip(S - U_t, 0.0, w)
    # 3. allocate mass: fill cells in any order that satisfies r >= 0 until sum=1
    # simplest: proportional to max(r, tiny) subject to cap
    weights = r.clip(min=1e-12)
    raw = weights / weights.sum()
    n = np.minimum(raw, cap)
    n = n / n.sum()  # re-normalize after caps
    return (
        n,
        r,
        S,
    )  ## TODO: Understand this algorithm (obviously, the solution here is not unique!)


def owner_optimality_residuals(
    n_t, r_t, S_t, B_t, P_t, P_tp1, c, kappa, tau, eps=1e-10
):
    """Returns a vector of residuals enforcing:
        if n>0  =>  Π_house >= Π_vacant   (diff <= 0)
        if n=0  =>  Π_house <= Π_vacant   (diff >= 0)
    Implemented as complementarity residuals.
    """
    # profits
    pi_house = r_t * n_t - c * n_t - kappa * B_t - tau * S_t + (P_tp1 - P_t)
    pi_vacant = -kappa * B_t - tau * S_t + (P_tp1 - P_t)

    diff = pi_house - pi_vacant  # = (r_t - c) * n_t

    occ = n_t > eps
    vac = ~occ

    # complementarity residuals (both should be ~0)
    res_occ = np.maximum(diff[occ], 0.0)  # if occ, diff should be <= 0
    res_vac = np.maximum(-diff[vac], 0.0)  # if vac, diff should be >= 0

    return np.concatenate([res_occ.ravel(), res_vac.ravel()])


def equilibrium_residuals(U, n0, B0, par):
    """U : (T,) array  -> one utility level per period
    n0,B0,P0 : (G,G) arrays (initial states)
    par : dict with keys beta, tau, c, kappa, delta_B, w, T
    returns: 1D residual vector
    """
    T = par["T"]
    beta = par["beta"]
    tau = par["tau"]
    c = par["c"]
    kappa = par["kappa"]
    dB = par["delta_B"]
    w = par["w"]
    mass = 1.0

    Gshape = n0.shape

    # storage
    n_path = np.empty((T + 1, *Gshape))
    B_path = np.empty((T + 1, *Gshape))
    S_path = np.empty((T, *Gshape))
    r_path = np.empty((T, *Gshape))
    P_path = np.empty((T + 1, *Gshape))

    n_path[0] = n0
    B_path[0] = B0
    # P_path[0] = P0  # only used if you enforce an initial-price residual

    # forward
    n_prev, B_prev = n0, B0
    for t in range(T):
        n_t, r_t, S_t = relocate(n_prev, U[t], w, cap=1.0)
        B_t1 = (1 - dB) * B_prev + c * n_t

        n_path[t + 1] = n_t
        r_path[t] = r_t
        S_path[t] = S_t
        B_path[t + 1] = B_t1

        n_prev, B_prev = n_t, B_t1

    # backward prices
    P_path[T] = np.zeros_like(n0)  # scrap value
    for t in range(T - 1, -1, -1):
        P_path[t] = beta * (
            P_path[t + 1]
            + (1 - tau) * S_path[t]
            - c * n_path[t + 1]
            - kappa * B_path[t + 1]
        )

    # residuals
    res = []

    # HH and owner optimality
    for t in range(T):
        n_t = n_path[t + 1]
        r_t = r_path[t]
        S_t = S_path[t]
        B_t = B_path[t]
        P_t = P_path[t]
        P_tp1 = P_path[t + 1]

        ## 1. Household indifference on occupied cells
        occ = n_path[t + 1] > 1e-10
        res.extend((S_path[t][occ] - r_path[t][occ] - U[t]).ravel())

        # 2. Owner optimality (new)
        res.extend(
            owner_optimality_residuals(
                n_t, r_t, S_t, B_t, P_t, P_tp1, c=c, kappa=kappa, tau=tau, eps=1e-10
            )
        )

    # mass constraint each period
    for t in range(1, T + 1):
        res.append(n_path[t].sum() - mass)

    # price recursion residuals
    for t in range(T):
        lhs = P_path[t]
        rhs = beta * (
            P_path[t + 1]
            + (1 - tau) * S_path[t]
            - c * n_path[t + 1]
            - kappa * B_path[t + 1]
        )
        res.extend((lhs - rhs).ravel())

    # optional: enforce given initial price
    # res.extend((P_path[0] - P0).ravel())

    return np.asarray(res)


def simulate_from_U(U, n0, B0, par, P_terminal=None):
    """Given U (length T), simulate forward n,r,S,B and backward P.
    Returns dict with full paths.
    """
    T = par["T"]
    beta = par["beta"]
    tau = par["tau"]
    c = par["c"]
    kappa = par["kappa"]
    dB = par["delta_B"]
    w = par["w"]

    Gshape = n0.shape
    n_path = np.empty((T + 1, *Gshape))
    B_path = np.empty((T + 1, *Gshape))
    S_path = np.empty((T, *Gshape))
    r_path = np.empty((T, *Gshape))
    P_path = np.empty((T + 1, *Gshape))

    # initial states
    n_path[0] = n0
    B_path[0] = B0

    # ---------- forward ----------
    n_prev, B_prev = n0, B0
    for t in range(T):
        n_t, r_t, S_t = relocate(n_prev, U[t], w, cap=1.0)
        B_t1 = (1 - dB) * B_prev + c * n_t

        n_path[t + 1] = n_t
        r_path[t] = r_t
        S_path[t] = S_t
        B_path[t + 1] = B_t1

        n_prev, B_prev = n_t, B_t1

    # ---------- backward prices ----------
    if P_terminal is None:
        P_path[T] = np.zeros_like(n0)  # scrap value
    else:
        P_path[T] = P_terminal

    for t in range(T - 1, -1, -1):
        P_path[t] = beta * (
            P_path[t + 1]
            + (1 - tau) * S_path[t]
            - c * n_path[t + 1]
            - kappa * B_path[t + 1]
        )

    return dict(n=n_path, r=r_path, S=S_path, B=B_path, P=P_path)


def summarize_paths(paths, eps=1e-10):
    n_path, r_path, B_path = paths["n"], paths["r"], paths["B"]
    T = r_path.shape[0]

    vacant_share = (n_path[1:] <= eps).mean(axis=(1, 2))
    capital_sum = B_path[1:].sum(axis=(1, 2))
    rent_mean = np.array(
        [
            r_path[t][n_path[t + 1] > eps].mean()
            if (n_path[t + 1] > eps).any()
            else 0.0
            for t in range(T)
        ]
    )
    return dict(vacant_share=vacant_share, capital_sum=capital_sum, rent_mean=rent_mean)


def plot_series(d):
    for k, y in d.items():
        plt.figure()
        plt.plot(y)
        plt.title(k.replace("_", " ").title())
        plt.xlabel("t")
        plt.ylabel(k)
        plt.tight_layout()
        plt.show()


def plot_population_heatmap(n_path, t, title=None, cmap="viridis"):
    """Heatmap of population shares at period t.

    Parameters
    ----------
    n_path : array with shape (T+1, G, G)
        Population path from simulate_from_U.
    t : int
        Period index to plot (0 ... T).
    title : str or None
        Plot title. If None, a default is used.
    cmap : str
        Matplotlib colormap name.
    """
    M = n_path[t]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, origin="lower", cmap=cmap, aspect="equal")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Population share")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(title or f"Population shares at t = {t}")
    plt.tight_layout()
    plt.show()


###############################
# DYNAMIC MODEL FUNCTIONS
###############################


def neighbors_8(idx, G):
    """Return the Moore (8-direction) neighbors for a cell 'idx' on a GxG grid.
    We work in linear indices, then convert to (i,j) and back.
    """
    i, j = divmod(idx, G)
    neigh = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < G and 0 <= nj < G:
                neigh.append(ni * G + nj)
    return neigh


###########################
##SIMPLE MODEL FUNCTIONS
###########################


def random_grid_inhabitants(grid_size=10, total_mass=1.0, seed=42):
    rng = np.random.default_rng(seed)
    n = rng.dirichlet(np.ones(grid_size**2)) * total_mass
    n_grid = n.reshape(grid_size, grid_size)
    return n_grid


def neighbor_avg(arr, i, j, grid_size=10):
    neighbors = []
    # von Neumann neighbors: up, down, left, right
    if i > 0:
        neighbors.append(arr[i - 1, j])
    if i < grid_size - 1:
        neighbors.append(arr[i + 1, j])
    if j > 0:
        neighbors.append(arr[i, j - 1])
    if j < grid_size - 1:
        neighbors.append(arr[i, j + 1])
    if len(neighbors) == 0:
        return 0.0
    return np.mean(neighbors)


def rents_benchmark_case(n_grid, S_grid):
    """Calculate equilibrium rents in the simple case.
    n_grid: population mass per cell
    S_grid: land supply per cell
    Returns: r_grid, equilibrium rents per cell
    """
    occupied = n_grid > 0
    u_star = S_grid[occupied].min()
    r_grid = S_grid - u_star
    r_grid[r_grid < 0] = 0  # Ensure non-negative rents
    return r_grid
