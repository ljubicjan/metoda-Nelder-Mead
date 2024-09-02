import numpy as np

def createInitialSimplex(x0, stepSize):
    n = len(x0)
    simplex = np.zeros((n+1, n))
    simplex[0] = x0
    for i in range(1, n+1):
        simplex[i] = np.array(x0 + stepSize * np.eye(n)[i-1])
    return simplex

def calculateCentroid(simplex):
    n = len(simplex[0])
    centroid = np.zeros(n)
    for i in range(n):
        for j in range(len(simplex)-1):
            centroid[i] += simplex[j][i]
        centroid[i] /= n
    return centroid

def sortSimplex(simplex, simplexFunValues):
    sortedIndexes = np.argsort(simplexFunValues)
    sortedFunctionValues = simplexFunValues[sortedIndexes]
    sortedSimplex = simplex[sortedIndexes]
    return sortedSimplex, sortedFunctionValues

def rosenbrock(x):
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1))

def rastrigin(x):
    A = 10
    return A * len(x) + sum([xi**2 - A * np.cos(2 * np.pi * xi) for xi in x])

def easom(x):
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi)**2 - (x[1] - np.pi)**2)

def neldermead(f, x_initial, isGreedy, midPrint, tol=1e-6, max_iter=1000, reflectIndex=1.0, expandIndex=2.0, contractIndex=0.5, shrinkIndex=0.5):
    x0 = x_initial
    iterations = 0
    simplex = createInitialSimplex(x0, 1.0)
    simplex_values = np.zeros(len(simplex))
    for i in range(len(simplex)):
        simplex_values[i] = f(simplex[i])
    simplex, func_values = sortSimplex(simplex, simplex_values)
    centroid = calculateCentroid(simplex)

    numOfReflections = 0
    numOfExpansions = 0
    numOfOuterContractions = 0
    numOfInnerContractions = 0
    numOfShrinks = 0

    while np.linalg.norm(simplex[0] - simplex[-1]) > tol and iterations < max_iter:
        iterations += 1
        action = ""
        centroid = calculateCentroid(simplex)
        x_r = centroid + reflectIndex * (centroid - simplex[-1])
        f_xr = f(x_r)

        if f_xr < func_values[-2]:
            if f_xr < func_values[0]:
                x_e = centroid + expandIndex * (x_r - centroid)
                f_xe = f(x_e)
                if (not isGreedy and f_xr > f_xe):
                    simplex[-1] = x_e
                    func_values[-1] = f_xe
                    numOfExpansions += 1
                    action = "expansion"
                elif (isGreedy and f_xe < func_values[0]):
                    simplex[-1] = x_e
                    func_values[-1] = f_xe
                    numOfExpansions += 1
                    action = "expansion"
                else:
                    simplex[-1] = x_r
                    func_values[-1] = f_xr
                    numOfReflections += 1
                    action = "reflection"
            else:
                simplex[-1] = x_r
                func_values[-1] = f_xr
                numOfReflections += 1
                action = "reflection"
        else:
            f_c = 3
            isInnerContraction = False
            if f_xr < func_values[-1]:
                x_oc = centroid + contractIndex * (x_r - centroid)
                f_c = f(x_oc)
                action = "outer contraction"
            else:
                isInnerContraction = True
                x_ic = centroid + contractIndex * (simplex[-1] - centroid)
                f_c = f(x_ic)
                action = "inner contraction"
            if f_c < min(f_xr, func_values[-1]):
                if isInnerContraction:
                    simplex[-1] = x_ic
                    func_values[-1] = f_c
                    numOfInnerContractions += 1
                else:
                    simplex[-1] = x_oc
                    func_values[-1] = f_c
                    numOfOuterContractions += 1  
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + shrinkIndex * (simplex[i] - simplex[0])
                    func_values[i] = f(simplex[i])
                numOfShrinks += 1
                action = "shrink"
        
        simplex, func_values = sortSimplex(simplex, func_values)
        formatted_str = ', '.join([f'{x:.4f}' for i, x in enumerate(simplex[0])])
        if midPrint:
            print(f'Iteration {iterations}: Current approximation: {formatted_str} with value {func_values[0]}: {action}')
        
    print("Done in ", iterations, "iterations")
    print("Simplex at the end:\n", simplex)
    print("Function values at the end:\n", func_values)
    formatted_str = ', '.join([f'{x:.4f}' for i, x in enumerate(simplex[0])])
    print("Minimum is at:", {formatted_str})
    print("Function value at minimum is:", func_values[0])
    print("Number of reflections:", numOfReflections)
    print("Number of expansions:", numOfExpansions)
    print("Number of contractions:", numOfOuterContractions+numOfInnerContractions)
    print("Number of shrinks:", numOfShrinks)

x0 = np.array([0, 0]) # set up initial guess
function = easom # chooose function to optimize (easom, rastrigin, rosenbrock)
isGreedy = False # use greedy expansion or not
midPrint = True # print intermediate results or not
neldermead(function, x0, isGreedy, midPrint)