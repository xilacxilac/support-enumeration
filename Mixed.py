import numpy as np
import pulp
from itertools import chain, combinations

# Calculates all possible combination of supports of size 1 to n
def powerset(n):
    return chain.from_iterable(combinations(range(n), r) for r in range(1, n + 1))

# Validates if a given mixed strategy has valid probability
def validate_probs(prob, support, n, tol=10**-6):
    if not (1 - tol <= np.sum(prob) <= 1 + tol):
        return False

    for i in range(n):
        if prob[i] < 0:
            return False
        if i in support and prob[i] == 0:
            return False
        if i not in support and prob[i] != 0:
            return False

    return True

# Determines if there is a better strategy to play over the strategy found
def best_response(payoff1, payoff2, prob1, prob2, p1_support, p2_support, n, tol=10**-6):
    utility1 = np.dot(payoff1, prob2)
    utility2 = np.dot(prob1, payoff2)
    max1 = np.max(utility1)
    max2 = np.max(utility2)

    for i in p1_support:
        if not(max1 - tol <= utility1[i] <= max1 + tol):
            return False
    for i in p2_support:
        if not(max2 - tol <= utility2[i] <= max2 + tol):
            return False    

    return True

# Support enumeration algorithm to find all possible mixed strategies
def support_enumeration(payoff1, payoff2, n):
    supports = list(powerset(n))
    probability = np.array()

    for support1 in supports:
        for support2 in supports: 
            p1_support = list(support1)
            p2_support = list(support2)
            p1_sl = len(p1_support)
            p2_sl = len(p2_support)

            model_1 = pulp.LpProblem("Mixed_Nash_Equilibrium_1")
            model_2 = pulp.LpProblem("Mixed_Nash_Equilibrium_2")

            p1 = [pulp.LpVariable(f"x{i}", lowBound = 0, upBound = 1) for i in range(n)]
            p2 = [pulp.LpVariable(f"y{i}", lowBound = 0, upBound = 1) for i in range(n)]

            for i in range(n):
                if i not in p1_support:
                    model_1 += p1[i] == 0
                if i not in p2_support:
                    model_2 += p2[i] == 0

            arr1 = []
            arr2 = []
            for j in p2_support:
                arr1.append(pulp.lpSum(p1[i] * payoff2[i][j] for i in p1_support))
            for i in p1_support:
                arr2.append(pulp.lpSum(p2[j] * payoff1[i][j] for j in p2_support))
            
            for i in range(p2_sl):
                for j in range(i + 1, p2_sl):
                    model_1 += (arr1[i] == arr1[j])
            for i in range(p1_sl):
                for j in range(i + 1, p1_sl):
                    model_2 += (arr2[i] == arr2[j])

            model_1 += pulp.lpSum(p1) == 1
            model_2 += pulp.lpSum(p2) == 1

            model_1.solve(pulp.PULP_CBC_CMD(msg=0))
            model_2.solve(pulp.PULP_CBC_CMD(msg=0))

            prob1 = np.array([pulp.value(p) for p in p1])
            prob2 = np.array([pulp.value(p) for p in p2])

            if validate_probs(prob1, p1_support, n) and validate_probs(prob2, p2_support, n):
                if best_response(payoff1, payoff2, prob1, prob2, p1_support, p2_support, n):
                    probability.append(prob1, prob2)

    return probability
    
if __name__ == "__main__":
    # Takes a 2-player pay-off matrix
    n = int(input())

    payoff1 = []
    for _ in range(n):
        row = list(map(float, input().split()))
        payoff1.append(row)

    payoff2 = []
    for _ in range(n):
        row = list(map(float, input().split()))
        payoff2.append(row)

    # Support Enumeration
    probs = support_enumeration(np.array(payoff1), np.array(payoff2), n)
    
    # Print the Mixed Strategy Nash Equilibriums 
    for prob1, prob2 in probs:
        prob1 = np.around(prob1, 4)
        prob2 = np.around(prob2, 4)
        for prob in prob1:
            print(prob, end=" ")
        print()

        for prob in prob2:
            print(prob, end=" ")
        print()