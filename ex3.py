import numpy as np
from matplotlib import pyplot as plt
from ex3_utils import BayesianLinearRegression, polynomial_basis_functions, load_prior


def log_evidence(model: BayesianLinearRegression, X, y):
    """
    Calculate the log-evidence of some data under a given Bayesian linear regression model
    :param model: the BLR model whose evidence should be calculated
    :param X: the observed x values
    :param y: the observed responses (y values)
    :return: the log-evidence of the model on the observed data
    """
    # extract the variables of the prior distribution
    mu = model.mu
    cov = model.cov
    sigma = model.sig

    # extract the variables of the posterior distribution
    model.fit(X, y)
    map_mu = model.fit_mu
    map_cov = model.fit_cov
    H = model.h(X)

    # calculate the log-evidence
    N = len(X)
    p = get_p(H)
    det_map_cov, det_cov = np.linalg.det(map_cov), np.linalg.det(cov)
    delta_map = get_delta(map_mu, mu, cov)
    delta_y = get_sigma_norm(y, H, map_mu, sigma)
    p_y_1 = 0.5 * np.log(det_map_cov / det_cov)
    p_y_2 = 0.5 * (delta_map + delta_y + (N * np.log(sigma)))
    p_y_3 = 0.5 * p * np.log(2 * np.pi)
    return p_y_1 - p_y_2 - p_y_3


def get_delta(mu_x, mu, cov):
    mu_delta = mu_x - mu
    cov_inv = np.linalg.inv(cov)
    return np.transpose(mu_delta) @ cov_inv @ mu_delta


def get_sigma_norm(y, H, mu_x, sigma):
    inv_sigma = 1 / sigma
    y_Hmu = y - (H @ mu_x)
    y_Hmu_x_norm = (np.linalg.norm(y_Hmu)) ** 2
    return inv_sigma * y_Hmu_x_norm


def get_p(H):
    return len(H)


def plot_evidence_per_noise(var_noise, evs):
    max_evidence_index, max_evidence = np.argmax(evs), np.max(evs)
    max_evidence_noise = var_noise[max_evidence_index]

    plt.figure()
    plt.plot(var_noise, evs, color="darkseagreen", label="evidence")
    plt.xlabel('noise of model')
    plt.ylabel('log-evidence')
    plt.title('log-evidence as a function of noise of the model')
    plt.scatter(max_evidence_noise, max_evidence, label=f'noise with max evidence: {round(max_evidence_noise, 3)}',
                color="green",
                alpha=.5)
    plt.legend()
    plt.show()
    plt.savefig('log_evidence(noise).png')


def plot_evidence_per_d(degrees, ev_d, i):
    best_model_i, max_evidence = np.argmax(ev_d), np.max(ev_d)
    worst_model_i, min_evidence = np.argmin(ev_d), np.min(ev_d)
    best_model_deg, worst_model_deg = degrees[best_model_i], degrees[worst_model_i]

    plt.figure()
    plt.plot(degrees, ev_d, color="lightseagreen")
    plt.xlabel('degree')
    plt.ylabel('log-evidence')
    plt.plot(best_model_deg, max_evidence, color="cadetblue")
    plt.title(f'Function {i}, log-evidence, best degree: {best_model_deg}')
    plt.savefig(f'Function_{i}_log_evidence_deg_{best_model_deg}.png')
    plt.show()

    return best_model_deg, worst_model_deg


def get_model_by_degree(d, noise_var, alpha, x, y):
    pbf = polynomial_basis_functions(d)
    mean, cov = np.zeros(d + 1), np.eye(d + 1) * alpha
    model = BayesianLinearRegression(mean, cov, noise_var, pbf)
    return model.fit(x, y)


def plot_models(best_model, worst_model, x, y, i):
    pred_best, std_best = best_model.predict(x), best_model.predict_std(x)
    pred_worst, std_worst = worst_model.predict(x), worst_model.predict_std(x)

    plt.figure()
    plt.plot(x, pred_best, color="mediumseagreen", label="best")
    plt.plot(x, pred_worst, color="darkmagenta", label="worst")
    plt.title(f'Function {i}: best and worst models')
    plt.fill_between(x, pred_best - std_best, pred_best + std_best, alpha=.5, color="mediumseagreen",
                     label="confidence best")
    plt.fill_between(x, pred_worst - std_worst, pred_worst + std_worst, alpha=.5, color="darkmagenta",
                     label="confidence worst")
    plt.scatter(x, y, color="black", alpha=.5, label="y")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.savefig(f'Function_{i}_best_worst_models_fit.png')
    plt.show()


def main():
    # ------------------------------------------------------ section 2.1
    # set up the response functions
    f1 = lambda x: x ** 2 - 1
    f2 = lambda x: -x ** 4 + 3 * x ** 2 + 50 * np.sin(x / 6)
    f3 = lambda x: .5 * x ** 6 - .75 * x ** 4 + 2.75 * x ** 2
    f4 = lambda x: 5 / (1 + np.exp(-4 * x)) - (x - 2 > 0) * x
    f5 = lambda x: np.cos(x * 4) + 4 * np.abs(x - 2)
    functions = [f1, f2, f3, f4, f5]
    x = np.linspace(-3, 3, 500)

    # set up model parameters
    degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    noise_var = .25
    alpha = 5

    # go over each response function and polynomial basis function

    for i, f in enumerate(functions):
        ev_d = []
        y = f(x) + np.sqrt(noise_var) * np.random.randn(len(x))

        for j, d in enumerate(degrees):
            # set up model parameters
            pbf = polynomial_basis_functions(d)
            mean, cov = np.zeros(d + 1), np.eye(d + 1) * alpha

            # calculate evidence
            model = BayesianLinearRegression(mean, cov, noise_var, pbf)
            ev = log_evidence(model, x, y)
            ev_d.append(ev)

        # plot evidence versus degree and predicted fit
        best_model_deg, worst_model_deg = plot_evidence_per_d(degrees, ev_d, i + 1)

        best_model, worst_model = get_model_by_degree(best_model_deg, noise_var, alpha, x, y), get_model_by_degree(
            worst_model_deg, noise_var, alpha, x, y)
        plot_models(best_model, worst_model, x, y, i + 1)

        # # ------------------------------------------------------ section 2.2
    # load relevant data
    nov16 = np.load('nov162020.npy')
    hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16) // 2]
    hours_train = hours[:len(nov16) // 2]

    # load prior parameters and set up basis functions
    mu, cov = load_prior()
    pbf = polynomial_basis_functions(7)

    noise_vars = np.linspace(.05, 2, 100)
    evs = np.zeros(noise_vars.shape)
    for i, n in enumerate(noise_vars):
        # calculate the evidence
        mdl = BayesianLinearRegression(mu, cov, n, pbf)
        ev = log_evidence(mdl, hours_train, train)
        evs[i] = ev

    # plot log-evidence versus amount of sample noise
    plot_evidence_per_noise(noise_vars, evs)


if __name__ == '__main__':
    main()
