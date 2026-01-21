import numpy as np
import random
import math

goodsEach = 2
typeSpace = 15

n_goods = goodsEach * 2

len_O = 2 ** n_goods
o0 = np.array([1]*goodsEach + [0]*goodsEach)
o0pl = o0 ^ 1


def typeGenerator():
    return np.array(random.choices(range(1, 6), k=n_goods))

type1 = typeGenerator()

def sampleKTypes(K):
    types_set = set()
    while len(types_set) < K:
        t = tuple(typeGenerator())
        types_set.add(t)
    return np.array([np.array(t) for t in types_set])

types = sampleKTypes(typeSpace)

# Precompute all integer-to-binary outcomes
int_to_bin = np.array([[(i >> j) & 1 for j in range(n_goods-1, -1, -1)] for i in range(len_O)])

utilityCache = {}

def utility(type_array, outcome, player=True):
    key = (type_array.tobytes(), outcome.tobytes(), int(player))
    if key not in utilityCache:
        o_now = outcome ^ 1 if player else outcome
        utilityCache[key] = np.dot(type_array, o_now) - np.dot(type_array, o0pl if player else o0)
    return utilityCache[key]

def v(X, Y):
    value = 0.0
    # Precompute feasible outcomes as a mask
    mask = np.ones(len_O, dtype=bool)
    mask[list(Y)] = False
    feasible_outcomes = np.where(mask)[0]

    for type_array in types:
        if X:
            X_bin = int_to_bin[list(X)]
            max_util_X = max([utility(type_array, x) for x in X_bin])

            admissible_mask = []
            for o in feasible_outcomes:
                uo = utility(type_array, int_to_bin[o])
                if uo >= max_util_X:
                    admissible_mask.append(o)
        else:
            admissible_mask = feasible_outcomes.copy()

        if len(admissible_mask)>0:
            best_g = max(utility(type1, int_to_bin[o], player=False) for o in admissible_mask)
            value += best_g/typeSpace

    return value

# Branch-and-bound DFS
def bnb_dfs():
    CB = None
    L = -math.inf

    def SEARCH1(X, Y, w, d):
        nonlocal CB, L
        if d >= len_O:
            CB = X.copy()
            L = w
            return

        # Include d
        X_new = X | {d}
        val_X = v(X_new, Y)
        if val_X > L:
            SEARCH1(X_new, Y, val_X, d + 1)

        # Exclude d
        Y_new = Y | {d}
        val_Y = v(X, Y_new)
        if val_Y > L:
            remaining_mask = np.ones(len_O, dtype=bool)
            remaining_mask[list(Y_new)] = False
            feasible = True
            for type_array in types:
                if not any(utility(type_array, int_to_bin[o]) >= 0 for o in np.where(remaining_mask)[0]):
                    feasible = False
                    break
            if feasible:
                SEARCH1(X, Y_new, val_Y, d + 1)

    SEARCH1(set(), set(), 0, 0)
    return sorted(CB), L

def M(type2, cb):
    utilities = np.array([utility(type2, int_to_bin[o]) for o in cb])
    
    # Identify best responses: o such that u(theta, o) is maximal
    max_utility = np.max(utilities)
    best_responses = [o for o, util in zip(cb, utilities) if util >= max_utility]

    # Among best responses, select o that maximizes g(o)
    g_values = [utility(type1, int_to_bin[o], player=False) for o in best_responses]
    return best_responses[np.argmax(g_values)]

print("--- Results")
cb, l = bnb_dfs()
print(f"Type 1 : {type1}")
for type2 in types:
    res = M(type2, cb)
    out = int_to_bin[res]
    print(f"Type 2 : {type2}, outcome: {out}, agent1_util: {utility(type1, out, player=False)}, agent2_util: {utility(type2, out)}")

print("--- Analisys on first type for agent 2")
type2 = types[0]

real_out = int_to_bin[M(type2, cb)]
real_util_1 = utility(type1, real_out, player=False)
real_util_2 = utility(type2, real_out)

print(f"Type 1 : {type1}, Type 2 : {types[0]}, mechanism outcome : {real_out}, agent1 util : {real_util_1}, agent2 util : {real_util_2}")

def truthfulness_check(outcome):
    outcome_int = int("".join(str(b) for b in outcome), 2)
    if outcome_int not in cb:
        cb_plus = cb.copy()
        cb_plus.append(outcome_int)

        for true_type in types:
            truthful_out = int_to_bin[M(true_type, cb)]
            truthful_util = utility(true_type, truthful_out)

            for lie_type in types:
                if np.array_equal(true_type,lie_type):
                    continue
                lie_out = int_to_bin[M(lie_type, cb_plus)]
                lie_util = utility(true_type, lie_out)

                if lie_util > truthful_util:
                    return f"X -> truthfulness constraint ({true_type} could lie to be {lie_type} to reach outcome {lie_out} with util {lie_util} instead of reaching {truthful_out} with util {truthful_util})"
        return "X -> excluded because it lowers agent1 expected objective across the type distribution"
    return "Error"

for outcome in int_to_bin:
    reason = ""

    util2 = utility(type2, outcome)
    if util2<0:
        reason = f"X -> individual rationality constraint (agent2_util={util2})"
    
    else:
        if util2>real_util_2:
            reason = truthfulness_check(outcome)
        elif util2<real_util_2:
            reason = f"X -> not the maximum valuation for agent2 ({util2}<{real_util_2})"
        else:
            util1=utility(type1, outcome, player=False)
            if util1>real_util_1:
                reason = truthfulness_check(outcome)
            elif util1<real_util_1:
                reason = f"X -> same util for agent2 ({util2}), not the maximum valuation for agent1 ({util1}<{real_util_1})"
            else:
                if all(outcome == real_out):
                    reason = f"* -> best outcome chosen from the mechanism"
                else:
                    reason = f"~ -> valid alternative: same util for agent1 ({util1}) and agent2 ({util2})"
    
    o_int = int("".join(str(b) for b in outcome), 2)
    print(f"{outcome}:{'O' if o_int in cb else ' '},{reason}")
