import numpy as np
import math
import scipy
import scipy.odr
import matplotlib.pyplot as plot
import gc

def form_matrix(terms, dists, brm, rm) :
    mat = np.zeros([terms, terms], dtype=float)

    for i in range(terms) :
        for j in range(i, terms) :
            mat[i, j] = sum(((r - rm) / (r + brm)) ** (i + j) if i + j != 0 else 1 for r in dists) / len(dists)
            mat[j, i] = mat[i, j]
    return mat

def form_results(terms, dists, energies, brm, rm) :
    mat = np.zeros([terms], dtype=float)

    for i in range(terms) :
        mat[i] = sum(e * ((r - rm) / (r + brm)) ** i if i != 0 else e for r, e in zip(dists, energies)) / len(dists)
    return mat

def calc_energy(r, terms, coefs, brm, rm) :
    return sum(coefs[i] * ((r - rm) / (r + brm)) ** i if i != 0 else coefs[i] for i in range(terms))

def calc_coefs(terms, dists, energies, brm, rm) :
    mat = form_matrix(terms, dists, brm, rm)
    results = form_results(terms, dists, energies, brm, rm)

    return np.linalg.solve(mat, results)

def calc_loss(terms, dists, energies, coefs, brm, rm) :
    out = 0.0
    for r, e in zip(dists, energies) :
        curr_term = calc_energy(r, terms, coefs, brm, rm) - e
        out += curr_term * curr_term
    return out / len(dists)

def calc_brm_deriv(terms, dists, energies, coefs, brm, rm, step = 0) :
    if step == 0 :
        out = 0.0
        for r, e in zip(dists, energies) :
            curr_term = calc_energy(r, terms, coefs, brm, rm) - e
            b_term = sum(coefs[i] * i * ((r - rm) / (r + brm)) ** i * (1 / (r + brm)) for i in range(terms))
            out += curr_term * b_term
        return -2 * out / len(dists)

    return (calc_loss(terms, dists, energies, coefs, brm + step, rm) - calc_loss(terms, dists, energies, coefs, brm - step, rm)) / (2 * step)

def calc_rm_deriv(terms, dists, energies, coefs, brm, rm, step = 0) :
    if step == 0 :
        out = 0.0
        for r, e in zip(dists, energies) :
            curr_term = calc_energy(r, terms, coefs, brm, rm) - e
            rm_term = sum(coefs[i] * i * ((r - rm) / (r + brm)) ** (i - 1) * (1 / (r + brm)) for i in range(terms))
            out += curr_term * rm_term
        return -2 * out / len(dists)
    return (calc_loss(terms, dists, energies, coefs, brm, rm + step) - calc_loss(terms, dists, energies, coefs, brm, rm - step)) / (2 * step)

def jacobian(terms, dists, energies, brm, rm, sstot, step = 0) :
    coefs = calc_coefs(terms, dists, energies, brm, rm)

    return [calc_brm_deriv(terms, dists, energies, coefs, brm, rm, step) / sstot, calc_rm_deriv(terms, dists, energies, coefs, brm, rm, step) / sstot]

def find_min_b_rm(terms, dists, energies, guess_b, guess_rm, conv = 1e-6, max_iters = 100) :
    average_e = sum(energies) / len(energies)
    sstot = sum((e - average_e) ** 2 for e in energies) / len(energies)

    result = scipy.optimize.minimize(lambda x: calc_loss(terms, dists, energies, calc_coefs(terms, dists, energies, x[0], x[1]), x[0], x[1]) / sstot,
        [guess_b * guess_rm, guess_rm], tol = conv, options = {'maxiter': max_iters}, jac = lambda x: jacobian(terms, dists, energies, x[0], x[1], sstot, 0.001))

    if not result.success :
        raise RuntimeError("Could not optimize!")
    return result.x[0] / result.x[1], result.x[1], calc_coefs(terms, dists, energies, result.x[0], result.x[1])

def find_min_terms(start, dists, energies, guess_b, guess_rm, cutoff = 1e-8, conv = 1e-10, max_iters = 100) :
    terms = start
    average_e = sum(energies) / len(energies)
    sstot = sum((e - average_e) ** 2 for e in energies) / len(energies)
    b = 0
    rm = 0
    coefs = None
    while terms < len(energies) :
        print(f"Calculating for {terms} terms.")
        try :
            b, rm, coefs = find_min_b_rm(terms, dists, energies, guess_b, guess_rm, conv, max_iters)
            print(f"R squared: {1 - calc_loss(terms, dists, energies, coefs, b * rm, rm) / sstot}")

            if calc_loss(terms, dists, energies, coefs, b * rm, rm) / sstot < cutoff :
                return terms, b, rm, coefs
            terms += 1
        except RuntimeError :
            print("Optimization failed!")
            terms += 1
            continue

def short_loss(dists, energies, a, b, ns, alpha) :
    return sum((a + b / (r ** ns) * math.exp(-alpha * r) - e) ** 2 for r, e in zip(dists, energies)) / len(dists)

def short_energy(r, a, b, ns, alpha) :
    return a + b / (r ** ns) * math.exp(-alpha * r)

def short_all_func(dists, energies, x) :
    num_dists = sum(1 if d <= x[0] else 0 for d in dists)

    return short_loss(dists[:num_dists], energies[:num_dists], x[1], x[2], x[3], x[4])

def short_optimize_all(dists, energies, a_start, b_start, ns_start, alpha_start) :
    # a_start, b_start = short_find_ab(method, molecule, rin_start, e_inf, ns_start, 1e-2)
    # a_start = -0.2600158561e4 / 219474.6
    # b_start = 0.8053173040e9 / 219474.6

    result = scipy.optimize.least_squares(lambda x: [short_energy(r, x[0], x[1], x[2], x[3]) - e for r, e in zip(dists, energies)], [a_start, b_start, ns_start, alpha_start])

    if not result.success :
        raise RuntimeError("Could not optimize parameters!")
    return result.x[0], result.x[1], result.x[2], result.x[3]

def long_loss(dists, singlet, triplet, u_inf, c6, c8, c10, aexc, gamma, beta, sstot = 1) :
    return (sum((u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 + aexc * r ** gamma * math.exp(-beta * r) - e) ** 2 for r, e in zip(dists, singlet)) / len(dists) + \
        sum((u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 - aexc * r ** gamma * math.exp(-beta * r) - e) ** 2 for r, e in zip(dists, triplet)) / len(dists)) / sstot

def long_energy(r, u_inf, c6, c8, c10, aexc, gamma, beta, nex = 0, cex = 0) :
    return u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 + aexc * r ** gamma * math.exp(-beta * r) - cex * r ** -nex

def long_u_inf_deriv(dists, singlet, triplet, u_inf, c6, c8, c10, aexc, gamma, beta, sstot = 1) :
    return (2 * sum((u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 + aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, singlet)) / len(dists) + \
        2 * sum((u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 - aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, triplet)) / len(dists)) / sstot

def long_c6_deriv(dists, singlet, triplet, u_inf, c6, c8, c10, aexc, gamma, beta, sstot = 1) :
    return (-2 * sum(r ** -6 * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 + aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, singlet)) / len(dists) + \
        -2 * sum(r ** -6 * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 - aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, triplet)) / len(dists)) / sstot

def long_c8_deriv(dists, singlet, triplet, u_inf, c6, c8, c10, aexc, gamma, beta, sstot = 1) :
    return (-2 * sum(r ** -8 * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 + aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, singlet)) / len(dists) + \
        -2 * sum(r ** -8 * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 - aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, triplet)) / len(dists)) / sstot

def long_c10_deriv(dists, singlet, triplet, u_inf, c6, c8, c10, aexc, gamma, beta, sstot = 1) :
    return (-2 * sum(r ** -10 * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 + aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, singlet)) / len(dists) + \
        -2 * sum(r ** -10 * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 - aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, triplet)) / len(dists)) / sstot

def long_aexc_deriv(dists, singlet, triplet, u_inf, c6, c8, c10, aexc, gamma, beta, sstot = 1) :
    return (2 * sum(r ** gamma * math.exp(-beta * r) * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 + aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, singlet)) / len(dists) + \
        -2 * sum(r ** gamma * math.exp(-beta * r) * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 - aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, triplet)) / len(dists)) / sstot
    
def long_gamma_deriv(dists, singlet, triplet, u_inf, c6, c8, c10, aexc, gamma, beta, sstot = 1) :
    return (2 * sum(aexc * math.log(r) * r ** gamma * math.exp(-beta * r) * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 + aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, singlet)) / len(dists) + \
        -2 * sum(aexc * math.log(r) * r ** gamma * math.exp(-beta * r) * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 - aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, triplet)) / len(dists)) / sstot

def long_beta_deriv(dists, singlet, triplet, u_inf, c6, c8, c10, aexc, gamma, beta, sstot = 1) :
    return (-2 * sum(aexc * r ** (gamma + 1) * math.exp(-beta * r) * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 + aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, singlet)) / len(dists) + \
        2 * sum(aexc * r ** (gamma + 1) * math.exp(-beta * r) * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 - aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, triplet)) / len(dists)) / sstot
    
def long_deriv(dists, singlet, triplet, x, sstot = 1) :
    return [long_c6_deriv(dists, singlet, triplet, x[6], x[0], x[1], x[2], x[3], x[4], x[5], sstot),
            long_c8_deriv(dists, singlet, triplet, x[6], x[0], x[1], x[2],  x[3], x[4], x[5], sstot),
            long_c10_deriv(dists, singlet, triplet, x[6], x[0], x[1], x[2], x[3], x[4], x[5], sstot),
            long_aexc_deriv(dists, singlet, triplet, x[6], x[0], x[1], x[2], x[3], x[4], x[5], sstot),
            long_gamma_deriv(dists, singlet, triplet, x[6], x[0], x[1], x[2], x[3], x[4], x[5], sstot),
            long_beta_deriv(dists, singlet, triplet, x[6], x[0], x[1], x[2], x[3], x[4], x[5], sstot),
            long_u_inf_deriv(dists, singlet, triplet, x[6], x[0], x[1], x[2], x[3], x[4], x[5], sstot)]

def ssor(coefs, results, omega = 1, conv = 1e-8, maxiter = 1000, max_diis_vectors = 10) :
    diis_vectors = []
    stored_vectors = []



    x0 = np.zeros_like(results)
    z = x0

    L = np.zeros_like(coefs)
    D = np.zeros_like(coefs)

    for i in range(coefs.shape[0]) :
        D[i, i] = coefs[i, i]

    for i in range(coefs.shape[0]) :
        for j in range(i) :
            L[i, j] = coefs[i, j]
    
    precon = omega / (2 - omega) * (D / omega + L) @ np.linalg.inv(D) @ (D / omega + L).T

    print(f"Preconditioner condition number: {np.linalg.cond(precon, 2)}")
    print("Iter    Error   ")
    print("------------------------")

    cycles = 0

    r = results - coefs @ x0

    diis_vectors.append(r)
    stored_vectors.append(x0)

    z = np.linalg.solve(precon, r)

    x0 += z

    while np.linalg.norm(z) / np.linalg.norm(x0) > conv and cycles < maxiter :
        cycles += 1

        print(f"{cycles:<8d}{np.linalg.norm(z) / np.linalg.norm(x0): 16e}")
        print(x0)

        r = results - coefs @ x0

        z = np.linalg.solve(precon, r)

        x0 += z

        if len(diis_vectors) < max_diis_vectors :
            diis_vectors.append(r)
            stored_vectors.append(x0)
        else :
            max_error = 0
            max_index = 0
            for i in range(len(diis_vectors)) :
                error = np.dot(diis_vectors[i], r)
                if error > max_error :
                    max_error = error
                    max_index = i
            diis_vectors[max_index] = r
            stored_vectors[max_index] = r

        if len(diis_vectors) > 1 :
            diis_coefs = np.zeros([len(diis_vectors) + 1, len(diis_vectors) + 1])

            for i in range(len(diis_vectors)) :
                for j in range(i, len(diis_vectors)) :
                    overlap = np.dot(diis_vectors[i], diis_vectors[j])
                    diis_coefs[i, j] = overlap
                    diis_coefs[j, i] = overlap
            for i in range(len(diis_vectors)) :
                diis_coefs[i, -1] = -1
                diis_coefs[-1, i] = -1
            diis_results = np.zeros(len(diis_vectors) + 1)
            diis_results[-1] = 1
            try :
                diis_solution = np.linalg.solve(diis_coefs, diis_results)
                xtemp = sum(coef * vec for coef, vec in zip(diis_solution, stored_vectors))
                x0 = xtemp
            except np.linalg.LinAlgError :
                diis_vectors = []
                stored_vectors = []
    if np.linalg.norm(z) / np.linalg.norm(x0) > conv :
        raise ArithmeticError("Could not converge Symmetric Successive Over-relaxation iterations!")
    return x0

def long_find_disp(dists, triplet, nex) :
    coef_matrix = np.zeros([4, 4])
    res_matrix = np.zeros([4])
    if nex == 0 :
        coef_matrix = np.zeros([3, 3])
        res_matrix = np.zeros([3])

    coef_matrix[0, 0] = sum(-1 / r ** 12 for r in dists)
    coef_matrix[0, 1] = coef_matrix[1, 0] = sum(-1 / r ** 14 for r in dists)
    coef_matrix[0, 2] = coef_matrix[2, 0] = sum(-1 / r ** 16 for r in dists)
    coef_matrix[1, 1] = sum(-1 / r ** 16 for r in dists)
    coef_matrix[1, 2] = coef_matrix[2, 1] = sum(-1 / r ** 18 for r in dists)
    coef_matrix[2, 2] = sum(-1 / r ** 20 for r in dists)

    if nex != 0 :
        coef_matrix[0, 3] = coef_matrix[3, 0] = sum(-1 / r ** (6 + nex) for r in dists)
        coef_matrix[1, 3] = coef_matrix[3, 1] = sum(-1 / r ** (8 + nex) for r in dists)
        coef_matrix[2, 3] = coef_matrix[3, 2] = sum(-1 / r ** (10 + nex) for r in dists)
        coef_matrix[3, 3] = sum(-1 / r ** (2 * nex) for r in dists)

    res_matrix[0] = sum(e / r ** 6 for r, e in zip(dists, triplet))
    res_matrix[1] = sum(e / r ** 8 for r, e in zip(dists, triplet))
    res_matrix[2] = sum(e / r ** 10 for r, e in zip(dists, triplet))

    if nex != 0 :
        res_matrix[3] = sum(e / r ** nex for r, e in zip(dists, triplet))

    precon = np.zeros_like(coef_matrix)

    precon[0, 0] = 1 / coef_matrix[0, 0]
    precon[1, 1] = 1 / coef_matrix[1, 0]
    precon[2, 2] = 1 / coef_matrix[2, 0]

    if nex != 0 :
        precon[3, 3] = 1 / coef_matrix[3, 0]

    coef_matrix = precon @ coef_matrix
    res_matrix = precon @ res_matrix

    condition_number = np.linalg.cond(coef_matrix, 2)

    return np.linalg.lstsq(coef_matrix, res_matrix)[0]

def long_find_exch(dists, data, A, gamma, beta) :
    sign = 1 if sum(data) > 0 else -1

    #return [sign * math.exp(sum(math.log(abs(e)) - gamma * math.log(r) + beta * r for r, e in zip(dists, data)) / len(dists)), gamma, beta]
    coef_matrix = np.zeros([3, 3])
    res_matrix = np.zeros([3])

    coef_matrix[0, 0] = len(dists)
    coef_matrix[0, 1] = sum(math.log(d) for d in dists)
    coef_matrix[0, 2] = -sum(dists)
    coef_matrix[1, 0] = sum(math.log(d) for d in dists)
    coef_matrix[1, 1] = sum(math.log(d) ** 2 for d in dists)
    coef_matrix[1, 2] = -sum(d * math.log(d) for d in dists)
    coef_matrix[2, 0] = sum(dists)
    coef_matrix[2, 1] = sum(d * math.log(d) for d in dists)
    coef_matrix[2, 2] = -sum(d ** 2 for d in dists)

    res_matrix[0] = sum(math.log(abs(e)) for d, e in zip(dists, data))
    res_matrix[1] = sum(math.log(d) * math.log(abs(e)) for d, e in zip(dists, data))
    res_matrix[2] = sum(d * math.log(abs(e)) for d, e in zip(dists, data))

    condition_number = np.linalg.cond(coef_matrix, 2)

    sol = np.linalg.solve(coef_matrix, res_matrix)

    return [sign * math.exp(sol[0]), gamma, beta]

def long_func(dists, singlet, triplet, aexc, gamma, beta) :
    coefs = long_find_disp(dists, singlet, triplet, aexc, gamma, beta)
    return long_loss(dists, singlet, triplet, 0, coefs[0], coefs[1], coefs[2], aexc, gamma, beta)
        
def long_calc_params(dists, singlet, triplet, singlet_binding, ionization, nex_min = 0, nex_max = 0) :
    sum_data = [(s + t) / 2 for s, t in zip(singlet, triplet)]
    diff_data = [(s - t) / 2 for s, t in zip(singlet, triplet)]


    # least-squares fitting to a polynomial.
    
    guess_c10 = long_find_disp(dists, sum_data, 0)
    
    if nex_min != nex_max or nex_min != 0 :
        nex_found = 0
        max_cex = 0
        average = sum(sum_data) / len(sum_data)
        sstot = (sum((e - average) ** 2 for e in sum_data)) / len(sum_data)
        max_rsquare = 1 - sum((long_energy(r, 0, guess_c10[0], guess_c10[1], guess_c10[2],  0, 0, 0) - e) ** 2 for r, e in zip(dists, sum_data)) / sstot

        print(f"Nex: 0, R^2: {max_rsquare}")
        
        for nex in range(nex_min, nex_max + 1) :
            guess_c10 = long_find_disp(dists, sum_data, nex)
            sstot = (sum((e - average) ** 2 for e in sum_data)) / len(sum_data)
            rsquare = 1 - sum((long_energy(r, 0, guess_c10[0], guess_c10[1], guess_c10[2],  0, 0, 0, nex = nex, cex = guess_c10[3]) - e) ** 2 for r, e in zip(dists, sum_data)) / sstot
            print(f"Nex: {nex}, R^2: {rsquare}")
            if rsquare > max_rsquare :
                max_rsquare = rsquare
                nex_found = nex
                max_cex = guess_c10[3]

            


    # res_c10 = scipy.optimize.least_squares(lambda x: [long_energy(r, 0, x[0], x[1], x[2], 0, 0, 0) - e for r, e in zip(dists, sum_data)], guess_c10)

    # if res_c10.success :
    #    guess_c10 = res_c10.x 
    
    # https://doi.org/10.1016/0009-2614(95)01388-1.
    gamma = 7.0 / (2.0 * math.sqrt(ionization * 2))
    beta = 2 * math.sqrt(2 * ionization) * 5.29177210544e-1

    b = math.sqrt(singlet_binding * 2)

    A = ((2 * b) ** (1 / b) * math.sqrt(b) / math.gamma(1 / b + 1)) ** 4 * math.gamma(1 / (2 * b)) * 2 ** (-1 - 1 / b) * b ** (-2 - 1 / (2 * b)) * scipy.integrate.quad(lambda y: math.exp((y - 1) / b) * (1 - y) ** (3 / (2 * b)) * (1 + y) ** (1 / (2 * b)), 0, 1)[0]

    guess_x = long_find_exch(dists, diff_data, A, gamma, beta) # [A, math.sqrt(gamma), math.sqrt(beta)]
    guess_x[1] = math.sqrt(guess_x[1])
    guess_x[2] = math.sqrt(guess_x[2])
    print(f"Guess: {guess_x}")
    # print(f"Calculated: {guess_x}")

    # model = scipy.odr.Model(lambda x, pts: np.array([long_energy(r, 0, 0, 0, 0, x[0], x[1] ** 2, x[2] ** 2) for r in pts]))
    # data = scipy.odr.Data(dists, diff_data)
    # odr_run = scipy.odr.ODR(data, model, beta0 = guess_x)

    # res = odr_run.run()

    res = scipy.optimize.least_squares(lambda x: [long_energy(r, 0, 0, 0, 0, x[0], x[1] ** 2, x[2] ** 2) - e for r, e in zip(dists, diff_data)], guess_x)
    if not res.success :
        raise ArithmeticError("Could not converge third pass!")
    print(f"Optimized: {res.x}")
    res.x[1] = res.x[1] ** 2
    res.x[2] = res.x[2] ** 2
    # res.pprint()
    print(f"Calculated: {res.x}")
    if nex_min == 0 and nex_min == nex_max :
        return [guess_c10[0], guess_c10[1], guess_c10[2], res.x[0], res.x[1], res.x[2]]
    return [guess_c10[0], guess_c10[1], guess_c10[2], res.x[0], res.x[1], res.x[2], nex_found, max_cex]

# First, calculate the energy at infinite separation.
mr_method = "ci6"
sr_method = "ci6"
set basis cc-pVTZ
rout = 7.0

molecule Li {
    0 2
    Li
}

molecule LiP {
    1 1
    Li
}

molecule sLi2 {
    0 1
    Li
    Li 1 R
}

molecule tLi2 {
    0 3
    Li
    Li 1 R
}


set REFERENCE ROHF
set SCF_TYPE PK
set S 0.5
set CALC_S_SQUARED False


# E_inf_t = 2 * energy("cisdt", molecule = Li) # We can do full CI for lithium. Can't do it for bigger alkali metals.
# clean()
# print(f"E_inf_t: {E_inf_t}")
E_inf_t = -14.8921366717017

set REFERENCE RHF
set S 0

set CI_NUM_THREADS 8
set FCI True

# ionization = abs(energy("ccsd", molecule = LiP) - E_inf_t / 2) # CCSD is full ci for Li+.
# print(f"ionization: {ionization}")
# clean()

ionization = 0.19671496091309582

# sLi2.R = 1e10
# E_inf_s = energy(mr_method, molecule = sLi2)
# print(f"E_inf_s: {E_inf_s}")

# clean()
E_inf_s = -14.8921366717017

# E_inf = -14.912503630319984

# Then, calculate the energies along the potential energy surface.

sLi2.R = 2.6660670307193843
# E_bind = abs(optimize(sr_method, molecule = sLi2) - E_inf_s)
# print(f"E_bind: {E_bind}")
# print(f"Optimized R: {sLi2.R}")
E_bind = 0.03917456114807116

guess_rm = sLi2.R

print(f"Finding zero point for singlet:")

def singlet_func(R) :
    sLi2.R = R
    E = (energy(sr_method, molecule=sLi2) - E_inf_s)
    set GUESS_VECTOR DFILE

    # if variable("CC T1 DIAGNOSTIC") > 0.02 :
    #     E = energy(mr_method, molecule = sLi2) - E_inf_s
    return E

# result = scipy.optimize.root_scalar(singlet_func, bracket = (1, guess_rm), x0 = 1.6, xtol = 1e-6)
# print(f"Crossing: {result.root}")
# gc.collect()

s_rin = 1.806914

set GUESS_VECTOR H0_BLOCK

print(f"Calculating singlet short range:")
__short_s_dists = np.linspace(0.2, s_rin, 20)
__short_s_energies = [11.198787883980959, 6.1774479233359365, 3.715391032927881, 2.3214504870950172, 1.4898517809398832, 0.9812639029541472,
                      0.6648676099590318, 0.46491806617074616, 0.33615351568177054, 0.25103855572189815, 0.19269895262264924, 0.1507891143530209,
                      0.11899690541377339, 0.09352408548292956, 0.07215487435395396, 0.0536666842737219, 0.037436340281949754, 0.023169424622810908,
                      0.010722528127015352, -1.814438022051945e-08]

for idx, dist in enumerate(__short_s_dists) :
    if len(__short_s_energies) > idx :
        continue
    sLi2.R = dist
    E = (energy(sr_method, molecule=sLi2) - E_inf_s)
    set GUESS_VECTOR DFILE
    print(E)

    # if variable("CC T1 DIAGNOSTIC") > 0.02 :
    #     E = energy(mr_method, molecule = sLi2) - E_inf_s
    __short_s_energies.append(E)

s_a, short_s_b, s_ns, s_alpha = short_optimize_all(__short_s_dists, __short_s_energies, -167163.9313131849 / 219474.6, 227086.33338132472 / 219474.6, 1.4987380566420632, 0)

gc.collect()

# print(f"short R (å): {__short_s_dists}")
# print(f"singlet short energy (Eh): {__short_s_energies}")

average_e = sum(__short_s_energies) / len(__short_s_energies)
sstot = sum((e - average_e) ** 2 for e in __short_s_energies) / len(__short_s_energies)

print(f"Rin (å): {s_rin}")
print(f"A (cm^-1): {219474.6 * s_a}")
print(f"B (cm^-1 å^Ns): {219474.6 * short_s_b}")
print(f"Ns: {s_ns}")
print(f"alpha (å^-1): {s_alpha}")
print(f"R^2 fit: {1 - short_loss(__short_s_dists, __short_s_energies, s_a, short_s_b, s_ns, s_alpha) / sstot}")

print("Calculating singlet medium range: ")

__med_s_dists = np.linspace(s_rin, rout, 50)
__med_s_energies = [-1.814438022051945e-08, -0.011152598305505634, -0.01998096988105047, -0.026752021321957642, -0.03174830821841823, -0.03524691471285202,
                    -0.03750430180720521, -0.03874747879705964, -0.03917077775702715, -0.03893712567777996, -0.03818214958674382, -0.03701899768461736,
                    -0.03554250334086362, -0.03383232163812622, -0.031955393538643406, -0.02996798792886146, -0.02791747913547482, -0.02584382952482933,
                    -0.023780763826998452, -0.021756662602378185, -0.019795200050747752, -0.01791574222076342, -0.016133585690207397, -0.014460141588111597,
                    -0.012903115621940131, -0.011466771089351013, -0.010152299214119864, -0.008958246277831705, -0.007880977175840798, -0.006915159564227835,
                    -0.006054197733222466, -0.005290659184739255, -0.004616636855491407, -0.004024060429426157, -0.0035049356889160777, -0.003051563009814018,
                    -0.0026566413143456202, -0.002313387754337981, -0.0020155622015085584, -0.0017575281509518703, -0.0015341927805874178, -0.0013410343259376845,
                    -0.001174041283658056, -0.0010296899881598875, -0.0009048893581340423, -0.0007969426638307908, -0.0007035068812744072, -0.0006225581793533053,
                    -0.0005523344781739326, -0.0004913268095076972]

for idx, dist in enumerate(__med_s_dists) :
    if len(__med_s_energies) > idx :
        continue
    sLi2.R = dist

    E = (energy(sr_method, molecule=sLi2) - E_inf_s)
    set GUESS_VECTOR DFILE
    print(E)

    # if variable("CC T1 DIAGNOSTIC") > 0.02 :
    #     E = energy(mr_method, molecule = sLi2) - E_inf_s
    __med_s_energies.append(E)

gc.collect()

# print(f"singlet medium R (å): {__med_s_dists}")
# print(f"singlet medium energy (Eh): {__med_s_energies}")

terms, b, rm, coefs = find_min_terms(10, __med_s_dists, __med_s_energies, -0.13, guess_rm, cutoff = 1e-3, conv = 1e-5, max_iters=1000)

average_e = sum(__med_s_energies) / len(__med_s_energies)
sstot = sum((e - average_e) ** 2 for e in __med_s_energies) / len(__med_s_energies)

print(f"terms: {terms}")
print(f"b (dimensionless): {b}")
print(f"rm (å): {rm}")
print(f"coefs (cm^-1): {list(map(lambda x: 219474.6 * float(x), coefs))}")
print(f"R^2 fit: {1 - calc_loss(terms, __med_s_dists, __med_s_energies, coefs, b * rm, rm) / sstot}")

print(f"Calculating singlet long range: ")

set E_CONVERGENCE 1e-11

__long_dists = list(np.linspace(rout, 30, 50))
__long_s_energies = [-0.0004913284942986706, -0.00030219935885256177, -0.00019547493096183644, -0.0001319312063454703, -9.199797597325698e-05, -6.567521915279428e-05,
                     -4.766307906223233e-05, -3.5005247465136335e-05, -2.5955551249623454e-05, -1.9429251976177397e-05, -1.4691180073356236e-05, -1.123847165374059e-05,
                     -8.710885673224311e-06, -6.842036690102304e-06, -5.443473421351541e-06, -4.3861205067941e-06, -3.5655746817297995e-06, -2.9273931261286634e-06,
                     -2.424415402657587e-06, -2.0188372147345035e-06, -1.69123683235739e-06, -1.424525619242445e-06, -1.2066016648049072e-06, -1.0258128320117521e-06,
                     -8.760890715109326e-07, -7.513539479475639e-07, -6.468173001650257e-07, -5.599304966352747e-07, -4.856431967681374e-07, -4.2263165944689263e-07,
                     -3.6898086364089977e-07, -3.2313359632496486e-07, -2.838183110753789e-07, -2.4999195069597135e-07, -2.2079535000329997e-07, -1.9551803376316457e-07,
                     -1.7356947878965912e-07, -1.544578847045841e-07, -1.3776981511171016e-07, -1.2316450792582145e-07, -1.1034472535698114e-07, -9.906884912425085e-08,
                     -8.912668114646749e-08, -8.034091436570634e-08, -7.256025114088516e-08, -6.565523236190529e-08, -5.951500270384713e-08, -5.404459102464898e-08,
                     -4.916115869946225e-08, -4.479097448495395e-08]

for idx, dist in enumerate(__long_dists) :
    if len(__long_s_energies) > idx :
        continue
    sLi2.R = dist
    E = energy(mr_method, molecule = sLi2)
    # set GUESS_VECTOR DFILE
    print(E - E_inf_s)
    __long_s_energies.append(E - E_inf_s)

gc.collect()
clean()

# print(f"long R (å): {__long_dists}")
# print(f"singlet long energy (Eh): {__long_s_energies}")

print("No parameters until after triplets.")

set REFERENCE ROHF
set SCF_TYPE PK
set S 1
set GUESS_VECTOR H0_BLOCK
set E_CONVERGENCE 1e-6
set FCI True

print(f"Calculating triplet short range:")

tLi2.R = 4.0603916983
# optimize("ccsd(t)/cc-pVTZ", molecule = tLi2)
# print(f"R CCSD(T): {tLi2.R}")
# optimize("cisdtq/cc-pVTZ", molecule = tLi2)
# print(f"R CISDTQ: {tLi2.R}")

# optimize(sr_method, molecule = tLi2)
# print(f"Optimized R: {tLi2.R}")

# gc.collect()

guess_rm = tLi2.R

print(f"Finding zero point for triplet:")
def triplet_func(R) :
    tLi2.R = R
    E = (energy(sr_method, molecule=tLi2) - E_inf_t)
    
    clean()

    # if variable("CC T1 DIAGNOSTIC") > 0.02 :
    #     E = energy(mr_method, molecule = tLi2) - E_inf_t
    return E


# result = scipy.optimize.root_scalar(triplet_func, bracket = (3, guess_rm), x0 = 3.5, xtol = 1e-6)
# gc.collect()

set GUESS_VECTOR H0_BLOCK

# print(f"Crossing: {result.root}")

t_rin = 3.269677113113883
__short_t_dists = np.linspace(0.2, t_rin, 20)
__short_t_energies = [11.533848468199647, 4.070522565363948, 1.7425684687300311, 0.841258765022916, 0.4653817733093586, 0.29577510888929126,
                      0.20907624233061384, 0.15649599537247738, 0.1189158191129529, 0.08940399049632752, 0.06581341873956426, 0.04735403635265456,
                      0.03332277684914686, 0.022911853643224234, 0.015317200716918222, 0.009841503627768589, 0.005935072969181832, 0.0031850086349578532,
                      0.001283423133036976, 4.849454171562684e-13]


for idx, dist in enumerate(__short_t_dists) :
    if len(__short_t_energies) > idx :
        continue
    tLi2.R = dist
    E = (energy(sr_method, molecule=tLi2) - E_inf_t)
    print(E,flush=True)
    clean()

    # set GUESS_VECTOR DFILE

    # if variable("CC T1 DIAGNOSTIC") > 0.02 :
    #     E = energy(mr_method, molecule = tLi2) - E_inf_t
    __short_t_energies.append(E)

t_a, short_t_b, t_ns, t_alpha = short_optimize_all(__short_t_dists, __short_t_energies, -167163.9313131849 / 219474.6, 227086.33338132472 / 219474.6, 1.4987380566420632, 0)

gc.collect()

# print(f"triplet short R (å): {__short_t_dists}")
# print(f"triplet short energy (Eh): {__short_t_energies}")

average_e = sum(__short_t_energies) / len(__short_t_energies)
sstot = sum((e - average_e) ** 2 for e in __short_t_energies) / len(__short_t_energies)

print(f"Rin (å): {t_rin}")
print(f"A (cm^-1): {219474.6 * t_a}")
print(f"B (cm^-1 å^Ns): {219474.6 * short_t_b}")
print(f"Ns: {t_ns}")
print(f"alpha (å^-1): {t_alpha}")
print(f"R^2 fit: {1 - short_loss(__short_t_dists, __short_t_energies, t_a, short_t_b, t_ns, t_alpha) / sstot}")

print("Calculating triplet medium range: ", flush=True)
__med_t_dists = np.linspace(t_rin, rout, 50)
__med_t_energies = [__short_t_energies[-1], -0.0004413079197256309, -0.0007983793994466026, -0.0010835994824756057, -0.0013077145883659824, -0.0014800507186265577,
                    -0.001608701018572134, -0.0017006988167178605, -0.0017621331384773953, -0.0017982855659859354, -0.001813688625048826, -0.0018122244789093855,
                    -0.0017971904231597335, -0.001771348701673503, -0.0017370017545843552, -0.0016960526758307282, -0.0016500669374845955, -0.0016003199293379566,
                    -0.0015478585881059104, -0.0014935383298606553, -0.0014380627026344683, -0.001382008176111782, -0.0013258520499750404, -0.00126998744748974,
                    -0.0012147435218441416, -0.001160383603476589, -0.0011071347353155403, -0.0010551747665967781, -0.0010046457101857698, -0.0009556644973258699,
                    -0.0009083193811427748, -0.0008626690391011493, -0.0008187631360989656, -0.000776625556881072, -0.0007362667373111975, -0.00069768095751499,
                    -0.0006608600026769551, -0.0006257710705277475, -0.0005923828915790352, -0.0005606536607736246, -0.0005305361644385442, -0.0005019743942114019,
                    -0.0004749155836272223, -0.00044930008711574487, -0.00042506416299126215, -0.00040214557738593726, -0.00038048697423143096, -0.00036002210205587915,
                    -0.0003406875114979613, -0.00032243004875809333]

for idx, dist in enumerate(__med_t_dists) :
    if len(__med_t_energies) > idx :
        continue
    tLi2.R = dist
    E = (energy(sr_method, molecule=tLi2) - E_inf_t)
    set GUESS_VECTOR DFILE
    print(E, flush=True)

    # if variable("CC T1 DIAGNOSTIC") > 0.02 :
    #     E = energy(mr_method, molecule = tLi2) - E_inf_t
    __med_t_energies.append(E)

gc.collect()

# print(f"medium R (å): {__med_t_dists}")
# print(f"triplet medium energy (Eh): {__med_t_energies}")

t_terms, t_b, t_rm, t_coefs = find_min_terms(10, __med_t_dists, __med_t_energies, -0.13, guess_rm, cutoff = 1e-3, conv = 1e-5, max_iters=1000)

average_e = sum(__med_t_energies) / len(__med_t_energies)
sstot = sum((e - average_e) ** 2 for e in __med_t_energies) / len(__med_t_energies)

print(f"terms: {t_terms}")
print(f"b (dimensionless): {t_b}")
print(f"rm (å): {t_rm}")
print(f"coefs (cm^-1): {list(map(lambda x: 219474.6 * float(x), t_coefs))}")
print(f"R^2 fit: {1 - calc_loss(t_terms, __med_t_dists, __med_t_energies, t_coefs, t_b * t_rm, t_rm) / sstot}")

print(f"Calculating triplet long range: ", flush=True)

set E_CONVERGENCE 1e-11

__long_t_energies = [__med_t_energies[-1], -0.00023026568558748295, -0.0001653928831757412, -0.00011952723559893741, -8.692982041402786e-05, -6.361710869917658e-05,
                     -4.683157576756969e-05, -3.4673273587415565e-05, -2.582940097362041e-05, -1.9381791204864385e-05, -1.4675933893926185e-05, -1.1235834868728034e-05,
                     -8.711031295405292e-06, -6.843742609063952e-06, -5.446797358032995e-06, -4.386749319351679e-06, -3.5699904739061594e-06, -2.9313962528476623e-06,
                     -2.4255584065713265e-06, -2.0204302622062187e-06, -1.692952160681216e-06, -1.426177423269337e-06, -1.2073964619219169e-06, -1.0269043819732815e-06,
                     -8.771943047491959e-07, -7.523915162011008e-07, -6.47866510661288e-07, -5.599342358664217e-07, -4.856449056234169e-07, -4.2263263466679746e-07,
                     -3.6898176958288786e-07, -3.2313426245877963e-07, -2.838189647746958e-07, -2.499908440256604e-07, -2.2079430372912157e-07, -1.9551701768705243e-07,
                     -1.7356855508410263e-07, -1.544568526412604e-07, -1.3777042795481975e-07, -1.2316392172806445e-07, -1.1034543057064639e-07, -9.906924169911235e-08,
                     -8.912693694185236e-08, -8.034102627618722e-08, -7.256038436764811e-08, -6.565562316040996e-08, -5.9515436134915944e-08, -5.404478997661499e-08,
                     -4.9161288373511525e-08, -4.479361770393098e-08]
for idx, dist in enumerate(__long_dists) :
    if len(__long_t_energies) > idx :
        continue
    tLi2.R = dist
    E = (energy(sr_method, molecule=tLi2) - E_inf_t)
    set GUESS_VECTOR DFILE
    print(E, flush=True)

    # if variable("CC T1 DIAGNOSTIC") > 0.02 :
    #     E = energy(mr_method, molecule = tLi2) - E_inf_t
    __long_t_energies.append(E)

gc.collect()

# print(f"long R (å): {__long_dists}")
# print(f"triplet long energy (Eh): {__long_s_energies}")

average_s = sum(__long_s_energies) / len(__long_s_energies)
average_t = sum(__long_t_energies) / len(__long_t_energies)
sstot = (sum((e - average_t) ** 2 for e in __long_t_energies) + sum((e - average_s) ** 2 for e in __long_s_energies)) / (len(__long_t_energies) + len(__long_s_energies))

u_inf = 0

[c6, c8, c10, aexc, gamma, beta, nex, cex] = long_calc_params(__long_dists, __long_s_energies, __long_t_energies, ionization, ionization, nex_min = 2, nex_max = 30)



print(f"U_inf (Eh): {u_inf}")
print(f"C6 (Eh bohr^6): {c6 / 0.529177249 ** 6}")
print(f"C8 (Eh bohr^8): {c8 / 0.529177249 ** 8}")
print(f"C10 (Eh bohr^10): {c10 / 0.529177249 ** 10}")
if nex != 0 :
    print(f"Cex (Eh bohr^{nex}): {cex / 0.529177249 ** nex}")
print(f"Aexc (Eh): {aexc}")
print(f"U_inf (cm^-1): {219474.6 * u_inf}")
print(f"C6 (cm^-1 å^6): {219474.6 * c6}")
print(f"C8 (cm^-1 å^8): {219474.6 * c8}")
print(f"C10 (cm^-1 å^10): {219474.6 * c10}")
if nex != 0 :
    print(f"Nex: {nex}")
    print(f"Cex (cm^-1 å^{nex}): {219474.6 * cex}")
print(f"Aexc (cm^-1): {219474.6 * aexc}")
print(f"gamma (dimensionless): {gamma}")
print(f"beta (å^-1): {beta}")
print(f"R^2 fit: {1 - long_loss(__long_dists, __long_s_energies, __long_t_energies, u_inf, c6, c8, c10, aexc, gamma, beta) / sstot}", flush=True)



#Plotting
X_singlet = list(__short_s_dists) + list(__med_s_dists) + list(__long_dists)
Y_singlet = list(__short_s_energies) + list(__med_s_energies) + list(__long_s_energies)
F_singlet_short = [s_a + short_s_b * r ** -s_ns * math.exp(-s_alpha * r) for r in __short_s_dists]
F_singlet_med = [calc_energy(r, terms, coefs, b * rm, rm) for r in __med_s_dists]
F_singlet_long = [long_energy(r, u_inf, c6, c8, c10, aexc, gamma, beta) for r in __long_dists]
F_singlet = F_singlet_short + F_singlet_med + F_singlet_long

# print("Singlet surface:")
# for x, y in zip(X_singlet, Y_singlet) :
#     print(f"{x}, {y}")

X_triplet = list(__short_t_dists) + list(__med_t_dists) + list(__long_dists)
Y_triplet = list(__short_t_energies) + list(__med_t_energies) + list(__long_t_energies)
F_triplet_short = [t_a + short_t_b * r ** -t_ns * math.exp(-t_alpha * r) for r in __short_t_dists]
F_triplet_med = [calc_energy(r, t_terms, t_coefs, t_b * t_rm, t_rm) for r in __med_t_dists]
F_triplet_long = [long_energy(r, u_inf, c6, c8, c10, -aexc, gamma, beta) for r in __long_dists]
F_triplet = F_triplet_short + F_triplet_med + F_triplet_long

# print("Triplet surface:")
# for x, y in zip(X_triplet, Y_triplet) :
#     print(f"{x}, {y}")

figure1 = plot.figure()
plot.title("Short range")
plot.plot(__short_s_dists, __short_s_energies, label = "Expected singlet")
plot.plot(__short_s_dists, F_singlet_short, label = "Fit singlet")
plot.plot(__short_t_dists, __short_t_energies, label = "Expected triplet")
plot.plot(__short_t_dists, F_triplet_short, label = "Fit triplet")
plot.xlabel("Li-Li Distance (å)")
plot.ylabel("Relative Energy (Eh)")
plot.legend()

figure2 = plot.figure()
plot.title("Medium range")
plot.plot(__med_s_dists, __med_s_energies, label = "Expected singlet")
plot.plot(__med_s_dists, F_singlet_med, label = "Fit singlet")
plot.plot(__med_t_dists, __med_t_energies, label = "Expected triplet")
plot.plot(__med_t_dists, F_triplet_med, label = "Fit triplet")
plot.xlabel("Li-Li Distance (å)")
plot.ylabel("Relative Energy (Eh)")
plot.legend()

figure3 = plot.figure()
plot.title("Long range")
plot.plot(__long_dists, __long_s_energies, label = "Expected singlet")
plot.plot(__long_dists, F_singlet_long, "yx-", label = "Fit singlet")
plot.plot(__long_dists, __long_t_energies, label = "Expected triplet")
plot.plot(__long_dists, F_triplet_long, label = "Fit triplet")
plot.xlabel("Li-Li Distance (å)")
plot.ylabel("Relative Energy (Eh)")
plot.legend()

figure4 = plot.figure()
plot.plot(X_singlet, Y_singlet, label = "Expected singlet")
plot.plot(X_singlet, F_singlet, label = "Fit singlet")
plot.plot(X_triplet, Y_triplet, label = "Expected triplet")
plot.plot(X_triplet, F_triplet, label = "Fit triplet")
plot.xlabel("Li-Li Distance (å)")
plot.ylabel("Relative Energy (Eh)")
plot.legend()
plot.show()
