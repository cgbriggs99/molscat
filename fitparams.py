import numpy as np
import math
import scipy
import matplotlib.pyplot as plot

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

def short_loss(dists, energies, a, b, ns) :
    return sum((a + b / (r ** ns) - e) ** 2 for r, e in zip(dists, energies)) / len(dists)

def short_find_a(dists, energies, b, ns) :
    return sum(energies) / len(energies) - b * sum(r ** -ns for r in dists) / len(dists)

def short_find_b(dists, energies, ns) :
    average_e = sum(energies) / len(energies)
    average_rns = sum(r ** -ns for r in dists) / len(dists)
    return sum((e - average_e) * (r ** -ns - average_rns) for r, e in zip(dists, energies)) / sum((r ** -ns - average_rns) ** 2 for r in dists)

def short_func(dists, energies, ns) :
    b = short_find_b(dists, energies, ns)
    a = short_find_a(dists, energies, b, ns)

    return short_loss(dists, energies, a, b, ns)

def short_a_deriv(dists, energies, a, b, ns) :
    return 2 * sum(a + b * r ** -ns - e for r, e in zip(dists, energies)) / len(dists)

def short_b_deriv(dists, energies, a, b, ns) :
    return 2 * sum((a + b * r ** -ns - e) * r ** -ns for r, e in zip(dists, energies)) / len(dists)

def short_ns_deriv(dists, energies, a, b, ns) :
    return -2 * b * sum((a + b * r ** -ns - e) * math.log(r) * r ** -ns for r, e in zip(dists, energies)) / len(dists)

def short_hessian(dists, energies, a, b, ns) :
    aa = 2
    bb = 2 * sum(r ** (-2 * ns) for r in dists) / len(dists)
    ab = 2 * sum(r ** -ns for r in dists) / len(dists)
    ans = -2 * b * sum(math.log(r) * r ** -ns for r in dists) / len(dists)
    bns = -2 * sum(b * math.log(r) * r ** (-2 * ns) + (a + b * r ** -ns - e) * (math.log(r) * r ** -ns) for r, e in zip(dists, energies)) / len(dists)
    nsns = 2 * b * sum(math.log(r) ** 2 * (b * r ** (-2 * ns) + (a + b * r ** -ns - e) * (r ** -ns)) for r, e in zip(dists, energies)) / len(dists)
    return np.array([[aa, ab, ans], [ab, bb, bns], [ans, bns, nsns]])

def short_deriv(dists, energies, a, b, ns) :
    return [short_a_deriv(dists, energies, a, b, ns), short_b_deriv(dists, energies, a, b, ns), short_ns_deriv(dists, energies, a, b, ns)]

def short_find_min_ns(dists, energies) :
    result = scipy.optimize.minimize(lambda x: short_loss(dists, energies, x[0], x[1], x[2]), [-0.011847195807624208, 3669.296146342219, 12],
        jac = lambda x: short_deriv(dists, energies, x[0], x[1], x[2]), hess = lambda x: short_hessian(dists, energies, x[0], x[1], x[2]),
        bounds = [(None, None), (None, None), (0, 16)], method = 'BFGS')

    if not result.success :
        print("Could not converge!")

    return result.x[2], result.x[0], result.x[1]

def long_loss(dists, singlet, triplet, u_inf, c6, c8, c10, aexc, gamma, beta) :
    return sum((u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 + aexc * r ** gamma * math.exp(-beta * r) - e) ** 2 for r, e in zip(dists, singlet)) / len(dists) + \
        sum((u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 - aexc * r ** gamma * math.exp(-beta * r) - e) ** 2 for r, e in zip(dists, triplet)) / len(dists)

def long_energy(r, u_inf, c6, c8, c10, aexc, gamma, beta) :
    return u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 + aexc * r ** gamma * math.exp(-beta * r)

def long_u_inf_deriv(dists, singlet, triplet, u_inf, c6, c8, c10, aexc, gamma, beta) :
    return 2 * sum((u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 + aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, singlet)) / len(dists) + \
        2 * sum((u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 - aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, triplet)) / len(dists)

def long_c6_deriv(dists, singlet, triplet, u_inf, c6, c8, c10, aexc, gamma, beta) :
    return -2 * sum(r ** -6 * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 + aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, singlet)) / len(dists) + \
        -2 * sum(r ** -6 * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 - aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, triplet)) / len(dists)

def long_c8_deriv(dists, singlet, triplet, u_inf, c6, c8, c10, aexc, gamma, beta) :
    return -2 * sum(r ** -8 * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 + aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, singlet)) / len(dists) + \
        -2 * sum(r ** -8 * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 - aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, triplet)) / len(dists)

def long_c10_deriv(dists, singlet, triplet, u_inf, c6, c8, c10, aexc, gamma, beta) :
    return -2 * sum(r ** -10 * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 + aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, singlet)) / len(dists) + \
        -2 * sum(r ** -10 * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 - aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, triplet)) / len(dists)

def long_aexc_deriv(dists, singlet, triplet, u_inf, c6, c8, c10, aexc, gamma, beta) :
    return 2 * sum(r ** gamma * math.exp(-beta * r) * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 + aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, singlet)) / len(dists) + \
        -2 * sum(r ** gamma * math.exp(-beta * r) * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 - aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, triplet)) / len(dists)
    
def long_gamma_deriv(dists, singlet, triplet, u_inf, c6, c8, c10, aexc, gamma, beta) :
    return 2 * sum(a * math.log(r) * r ** gamma * math.exp(-beta * r) * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 + aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, singlet)) / len(dists) + \
        -2 * sum(a * math.log(r) * r ** gamma * math.exp(-beta * r) * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 - aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, triplet)) / len(dists)

def long_beta_deriv(dists, singlet, triplet, u_inf, c6, c8, c10, aexc, gamma, beta) :
    return -2 * sum(a * r ** (gamma + 1) * math.exp(-beta * r) * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 + aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, singlet)) / len(dists) + \
        2 * sum(a * r ** (gamma + 1) * math.exp(-beta * r) * (u_inf - c6 * r ** -6 - c8 * r ** -8 - c10 * r ** -10 - aexc * r ** gamma * math.exp(-beta * r) - e) for r, e in zip(dists, triplet)) / len(dists)
    
def long_deriv(dists, singlet, triplet, x) :
    return [long_u_inf_deriv(dists, singlet, triplet, x[0], x[1], x[2], x[3], x[4], x[5], x[6]), 
            long_c6_deriv(dists, singlet, triplet, x[0], x[1], x[2], x[3], x[4], x[5], x[6]),
            long_c8_deriv(dists, singlet, triplet, x[0], x[1], x[2], x[3], x[4], x[5], x[6]),
            long_c10_deriv(dists, singlet, triplet, x[0], x[1], x[2], x[3], x[4], x[5], x[6]),
            long_aexc_deriv(dists, singlet, triplet, x[0], x[1], x[2], x[3], x[4], x[5], x[6]),
            long_gamma_deriv(dists, singlet, triplet, x[0], x[1], x[2], x[3], x[4], x[5], x[6]),
            long_beta_deriv(dists, singlet, triplet, x[0], x[1], x[2], x[3], x[4], x[5], x[6])]
        
def long_calc_params(dists, singlet, triplet, guess_x = [0, 86.22129257781994, 2600.209558645966, 84430.85076815267, 0.04104901387222029, 5.19500, 2.13539], conv = 1e-8) :
    res = scipy.optimize.minimize(lambda x: long_loss(dists, singlet, triplet, x[0], x[1], x[2], x[3], x[4], x[5], x[6]), guess_x, jac = lambda x: long_deriv(dists, singlet, triplet, x), tol = conv)

    if not res.success :
        raise ArithmeticError("Could not converge!")
    return res.x

# First, calculate the energy at infinite separation.
method = "ccsd/cc-pVDZ"

molecule Li {
    0 2
    Li
}
set REFERENCE ROHF
set SCF_TYPE PK
set MAXITER 10000
set RESTART False
E_inf = 2 * energy(method, molecule = Li)

# E_inf = -14.912503630319984

# Then, calculate the energies along the potential energy surface.
molecule sLi2 {
    0 1
    Li
    Li 1 R
}

sLi2.R = 3
optimize(method, molecule = sLi2)

guess_rm = sLi2.R
assert(guess_rm != 3)

set REFERENCE RHF
set SCF_TYPE PK

print("Calculating singlet medium range: ")
__med_dists = np.linspace(1.6, 6.0, 100)
__med_s_energies = []

for dist in __med_dists :
    sLi2.R = dist
    __med_s_energies.append((energy(method, molecule=sLi2) - E_inf))

print(f"medium R (å): {__med_dists}")
print(f"singlet medium energy (Eh): {__med_s_energies}")

terms, b, rm, coefs = find_min_terms(10, __med_dists, __med_s_energies, -0.13, guess_rm, cutoff = 1e-5, conv = 1e-6, max_iters=1000)

average_e = sum(__med_s_energies) / len(__med_s_energies)
sstot = sum((e - average_e) ** 2 for e in __med_s_energies) / len(__med_s_energies)

print(f"terms: {terms}")
print(f"b (dimensionless): {b}")
print(f"rm (å): {rm}")
print(f"coefs (cm^-1): {list(map(lambda x: 219474.6 * float(x), coefs))}")
print(f"R^2 fit: {1 - calc_loss(terms, __med_dists, __med_s_energies, coefs, b * rm, rm) / sstot}")

print(f"Calculating singlet short range:")
__short_dists = np.linspace(0.2, 1.6, 20)
__short_s_energies = []

for dist in __short_dists :
    sLi2.R = dist
    __short_s_energies.append((energy(method, molecule=sLi2) - E_inf))

print(f"short R (å): {__short_dists}")
print(f"singlet short energy (Eh): {__short_s_energies}")

short_b = short_find_b(__short_dists, __short_s_energies, 12)
a = short_find_a(__short_dists, __short_s_energies, short_b, 12)

average_e = sum(__short_s_energies) / len(__short_s_energies)
sstot = sum((e - average_e) ** 2 for e in __short_s_energies) / len(__short_s_energies)

print(f"A (cm^-1): {219474.6 * a}")
print(f"B (cm^-1 å^Ns): {219474.6 * short_b}")
print(f"Ns: {12}")
print(f"R^2 fit: {1 - short_loss(__short_dists, __short_s_energies, a, short_b, 12) / sstot}")

print(f"Calculating singlet long range: ")

__long_dists = np.linspace(8.0, 50, 100)
__long_s_energies = []
for dist in __long_dists :
    sLi2.R = dist
    __long_s_energies.append(energy(method, molecule = sLi2) - E_inf)

print(f"long R (å): {__long_dists}")
print(f"singlet long energy (Eh): {__long_s_energies}")

print("No parameters until after triplets.")

molecule tLi2 {
    0 3
    Li
    Li 1 R
}

set REFERENCE ROHF
set SCF_TYPE PK

print("Calculating triplet medium range: ")
__med_t_energies = []

for dist in __med_dists :
    tLi2.R = dist
    __med_t_energies.append((energy(method, molecule=tLi2) - E_inf))

print(f"medium R (å): {__med_dists}")
print(f"triplet medium energy (Eh): {__med_s_energies}")

t_terms, t_b, t_rm, t_coefs = find_min_terms(10, __med_dists, __med_t_energies, -0.13, 2.673, cutoff = 1e-5, conv = 1e-6, max_iters=1000)

average_e = sum(__med_t_energies) / len(__med_t_energies)
sstot = sum((e - average_e) ** 2 for e in __med_t_energies) / len(__med_t_energies)

print(f"terms: {t_terms}")
print(f"b (dimensionless): {t_b}")
print(f"rm (å): {t_rm}")
print(f"coefs (cm^-1): {list(map(lambda x: 219474.6 * float(x), t_coefs))}")
print(f"R^2 fit: {1 - calc_loss(terms, __med_dists, __med_t_energies, coefs, t_b * t_rm, t_rm) / sstot}")

print(f"Calculating triplet short range:")
__short_t_energies = []

for dist in __short_dists :
    tLi2.R = dist
    __short_t_energies.append((energy(method, molecule=tLi2) - E_inf))

print(__short_dists)
print(__short_t_energies)

short_t_b = short_find_b(__short_dists, __short_t_energies, 6)
t_a = short_find_a(__short_dists, __short_t_energies, short_t_b, 6)

average_e = sum(__short_t_energies) / len(__short_t_energies)
sstot = sum((e - average_e) ** 2 for e in __short_t_energies) / len(__short_t_energies)

print(f"A (cm^-1): {219474.6 * t_a}")
print(f"B (cm^-1 å^Ns): {219474.6 * short_t_b}")
print(f"Ns: {6}")
print(f"R^2 fit: {1 - short_loss(__short_dists, __short_t_energies, t_a, short_t_b, 6) / sstot}")

print(f"Calculating triplet long range: ")

__long_t_energies = []
for dist in __long_dists :
    tLi2.R = dist
    __long_t_energies.append(energy(method, molecule = tLi2) - E_inf)

print(f"long R (å): {__long_dists}")
print(f"triplet long energy (Eh): {__long_s_energies}")

average_e = (sum(__long_t_energies) + sum(__long_s_energies)) / (len(__long_t_energies) + len(__long_s_energies))
sstot = (sum((e - average_e) ** 2 for e in __long_t_energies) + sum((e - average_e) ** 2 for e in __long_s_energies)) / (len(__long_t_energies) + len(__long_s_energies))

[u_inf, c6, c8, c10, aexc, gamma, beta] = long_calc_params(__long_dists, __long_s_energies, __long_t_energies)

print(f"U_inf (cm^-1): {219474.6 * u_inf}")
print(f"C6 (cm^-1 å^6): {219474.6 * c6}")
print(f"C8 (cm^-1 å^8): {219474.6 * c8}")
print(f"C10 (cm^-1 å^10): {219474.6 * c10}")
print(f"Aexc (cm^-1): {219474.6 * aexc}")
print(f"gamma (dimensionless): {gamma}")
print(f"beta (å^-1): {beta}")
print(f"R^2 fit: {1 - long_loss(__long_dists, __long_s_energies, __long_t_energies, u_inf, c6, c8, c10, aexc, gamma, beta) / sstot}")

#Plotting
X = list(__short_dists) + list(__med_dists) + list(__long_dists)
Y_singlet = list(__short_s_energies) + list(__med_s_energies) + list(__long_s_energies)
F_singlet_short = [a + short_b * r ** -12 for r in __short_dists]
F_singlet_med = [calc_energy(r, terms, coefs, b * rm, rm) for r in __med_dists]
F_singlet_long = [long_energy(r, u_inf, c6, c8, c10, aexc, gamma, beta) for r in __long_dists]
F_singlet = F_singlet_short + F_singlet_med + F_singlet_long

Y_triplet = list(__short_t_energies) + list(__med_t_energies) + list(__long_t_energies)
F_triplet_short = [t_a + short_t_b * r ** -6 for r in __short_dists]
F_triplet_med = [calc_energy(r, t_terms, t_coefs, t_b * t_rm, t_rm) for r in __med_dists]
F_triplet_long = [long_energy(r, u_inf, c6, c8, c10, -aexc, gamma, beta) for r in __long_dists]
F_triplet = F_triplet_short + F_triplet_med + F_triplet_long

figure1 = plot.figure()
plot.title("Short range")
plot.plot(__short_dists, __short_s_energies, label = "Expected singlet")
plot.plot(__short_dists, F_singlet_short, label = "Fit singlet")
plot.plot(__short_dists, __short_t_energies, label = "Expected triplet")
plot.plot(__short_dists, F_triplet_short, label = "Fit triplet")
plot.xlabel("Li-Li Distance (å)")
plot.ylabel("Relative Energy (Eh)")
plot.legend()
plot.show()

figure2 = plot.figure()
plot.title("Medium range")
plot.plot(__med_dists, __med_s_energies, label = "Expected singlet")
plot.plot(__med_dists, F_singlet_med, label = "Fit singlet")
plot.plot(__med_dists, __med_t_energies, label = "Expected triplet")
plot.plot(__med_dists, F_triplet_med, label = "Fit triplet")
plot.xlabel("Li-Li Distance (å)")
plot.ylabel("Relative Energy (Eh)")
plot.legend()
plot.show()

figure3 = plot.figure()
plot.title("Long range")
plot.plot(__long_dists, __long_s_energies, label = "Expected singlet")
plot.plot(__long_dists, F_singlet_long, label = "Fit singlet")
plot.plot(__long_dists, __long_t_energies, label = "Expected triplet")
plot.plot(__long_dists, F_triplet_long, label = "Fit triplet")
plot.xlabel("Li-Li Distance (å)")
plot.ylabel("Relative Energy (Eh)")
plot.legend()
plot.show()

figure4 = plot.figure()
plot.plot(X, Y_singlet, label = "Expected singlet")
plot.plot(X, F_singlet, label = "Fit singlet")
plot.plot(X, Y_triplet, label = "Expected triplet")
plot.plot(X, F_triplet, label = "Fit triplet")
plot.xlabel("Li-Li Distance (å)")
plot.ylabel("Relative Energy (Eh)")
plot.legend()
plot.show()