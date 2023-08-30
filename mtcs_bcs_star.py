"""
Multitask Compressive Sensing: Shihao Ji, David Dunson†, and Lawrence Carin

https://www.researchgate.net/publication/224514217_Multitask_Compressive_Sensing
"""

from .bayes_utils import *


# process results
def process_results_MAP(alpha, Phi, Z_measure, i=0):
    M, _ = Z_measure.shape
    A = vector_to_diagonal_matrix(alpha)
    Sigma = np.linalg.pinv(Phi.T @ Phi + A)
    MAP_thetas = {i: Sigma @ Phi.T @ Z_measure[i, :] for i in range(M)}
    return MAP_thetas


class MT_CS_BCS_star:
    def fit(
        self,
        X,
        dictionary="dct",
        measure_matrix="gaussian",
        proportion=0.5,
        MAX_ITER=2000,
        EPS=1e-2,
        type="iterative",
    ):
        # X must be a numpy array
        check_numpy_array(X)

        # importing matrix_utils module
        try:
            imported_module = importlib.import_module("matrix_utils")
            print(f"Successfully imported utils module.")

        except ModuleNotFoundError:
            print(
                f"Module '{dictionary}' not found. Please check the module name and try again."
            )

        # importing the dictionary
        try:
            generate_dict = getattr(imported_module, f"{dictionary}_dictionary")

        except NameError:
            print(f"{dictionary} not found. Please check the name and try again.")

        # importing the measure matrix
        try:
            generate_measure = getattr(imported_module, f"{measure_matrix}_matrix")
        except NameError:
            print(f"{measure_matrix} not found. Please check the name and try again.")

        # BCS algorithm

        Z = X.copy().T
        M, n = Z.shape
        m = int(n * proportion)

        # measures vectors
        Phi = generate_measure(m, n)
        Z_measure = Z @ Phi.T

        # initialization
        alpha_0 = np.random.random()
        alpha = np.ones((n, 1))
        k = 0
        mu = dict()
        Sigma = dict()

        # store alpha_zeros
        alpha_zeros = list()

        # generates a diagonal matrix of alpha
        A = vector_to_diagonal_matrix(alpha)

        if type == "iterative":
            a = b = 1e-1
            Sigma = np.linalg.pinv(Phi.T @ Phi + A)

            def compute_mu(Sigma, Phi, i):
                return Sigma @ Phi.T @ Z_measure[i, :]

            def compute_B(alpha, Phi, i=0):
                A_inv = vector_to_diagonal_matrix(1 / alpha)
                return np.eye(m) + Phi @ A_inv @ Phi.T

            B_i_inv = np.linalg.pinv(compute_B(alpha, Phi, i=0))
            for j in tqdm(range(n)):
                sum_denom = 0
                for i in range(M):
                    mu_i = compute_mu(Sigma, Phi, i)
                    sum_denom += (
                        (mu_i[j] ** 2)
                        * (n + 2 * a)
                        / (Z_measure[i, :].T @ B_i_inv @ Z_measure[i, :] + 2 * b)
                    )
                alpha[j] = M / sum_denom

            # computing the mean/mode sparse approximation from the Multivariate Student
            MAP_thetas = process_results_MAP(alpha, Phi, Z_measure)

            # Dictionary
            D = generate_dict(n, n)
            MAP_errors = {
                i: compute_relative_error(Z[i, :], D @ MAP_thetas[i]) for i in range(M)
            }

            return MAP_thetas, MAP_errors
        elif type == "fast":
            """only updates alpha_0
            prev_alpha_0 = np.inf
            while k < MAX_ITER and np.linalg.norm(prev_alpha_0 - alpha_0) > EPS:
                prev_alpha_0 = alpha_0

                # Sigma & mu
                Sigma = {i: alpha_0 * Phi.T @ Phi + A for i in range(M)}
                mu = {i: alpha_0 * Sigma[i] @ Phi.T @ Z_measure[i, :] for i in range(M)}

                # update alpha_0
                sum_denom = 0
                for i in range(M):
                    sum_denom += np.linalg.norm(Z_measure[i, :] - Phi @ mu[i]) ** 2
                alpha_0 = m * M / sum_denom"""

            # computes C_i_{-j}, each signal is assumed to have its own measure matrix Phi_i
            def compute_C_j(Phi, alpha, alpha_0, j, i=0):
                m, n = Phi.shape
                I = np.eye(m)

                C_j = 1 / alpha_0 * np.eye(m)
                for k in range(n):
                    C_j += 1 / alpha[k] * Phi[:, k] @ Phi[:, k].T if i != j else 0
                return C_j

            def compute_s_ij(Phi, C_j, j, i=0):
                C_ij_inv = np.linalg.pinv(C_j)
                return Phi[:, j].T @ C_ij_inv @ Phi[:, j]

            def compute_q_ij(Ph_i, C_j, j, i):
                C_ij_inv = np.linalg.pinv(C_j)
                # print(Phi.T[:, j].shape, C_ij_inv.shape, Z_measure[i, :].shape)
                return Phi[:, j].T @ C_ij_inv @ Z_measure[i, :]

            # initializes alpha (?)
            alpha = np.ones((n, 1))

            # updates alpha
            for j in tqdm(range(n)):
                C_j = compute_C_j(Phi, alpha, alpha_0, j)
                sum_denom = 0
                for k in range(M):
                    sum_denom += (
                        compute_q_ij(Phi, C_j, j, k) ** 2 - compute_s_ij(Phi, C_j, j)
                    ) / compute_s_ij(Phi, C_j, j) ** 2
                alpha[j] = M / sum_denom

            MAP_thetas = process_results_MAP(alpha, Phi, Z_measure)

            # Dictionary
            D = generate_dict(n, n)
            MAP_errors = {
                i: compute_relative_error(Z[i, :], D @ MAP_thetas[i]) for i in range(M)
            }

            return MAP_thetas, MAP_errors
        else:
            raise ValueError(f"Argument {type} doesn't exist.")
