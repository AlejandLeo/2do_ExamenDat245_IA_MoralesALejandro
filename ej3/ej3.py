import numpy as np
import random
import math

# Definir el problema: coordenadas de las ciudades
cities = np.array([
    [0, 0], [1, 5], [2, 3], [5, 2], [6, 6], [8, 3], [9, 9], [10, 1]
])

# Función de costo: distancia total de la ruta
def total_distance(route, cities):
    distance = 0
    for i in range(len(route)):
        city1 = cities[route[i]]
        city2 = cities[route[(i + 1) % len(route)]]
        distance += np.linalg.norm(city1 - city2)
    return distance

# Generar una solución vecina
def neighbor_solution(route):
    new_route = route.copy()
    i, j = random.sample(range(len(route)), 2)
    new_route[i], new_route[j] = new_route[j], new_route[i]  # Swap two cities
    return new_route

# Simulated Annealing
def simulated_annealing(cities, initial_temperature, cooling_rate, max_iterations):
    num_cities = len(cities)
    current_route = list(range(num_cities))
    random.shuffle(current_route)
    current_cost = total_distance(current_route, cities)
    temperature = initial_temperature

    best_route = current_route
    best_cost = current_cost

    for iteration in range(max_iterations):
        new_route = neighbor_solution(current_route)
        new_cost = total_distance(new_route, cities)
        
        # Calcular el cambio en la energía (costo)
        delta_cost = new_cost - current_cost
        
        # Decidir si aceptar la nueva solución
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / temperature):
            current_route = new_route
            current_cost = new_cost
            
            # Actualizar la mejor solución encontrada
            if current_cost < best_cost:
                best_route = current_route
                best_cost = current_cost
        
        # Enfriar la temperatura
        temperature *= cooling_rate
        
        # (Opcional) Imprimir el progreso
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Best Cost: {best_cost}")
    
    return best_route, best_cost

# Parámetros del algoritmo
initial_temperature = 1000
cooling_rate = 0.99
max_iterations = 1000

# Ejecutar Simulated Annealing
best_route, best_cost = simulated_annealing(cities, initial_temperature, cooling_rate, max_iterations)
print(f"Best Route: {best_route}, Best Cost: {best_cost}")


