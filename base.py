from collections import defaultdict
import copy
from queue import PriorityQueue

import numpy as np

def serialize (state):
    result = []
    for row in state:
        for col in row:
            result.append(str(col))
    return ':'.join(result)

def deserialize (serialized):
    splitted = serialized.split(':')
    splitted = [int(x) for x in splitted]
    order_ = int(len(splitted)**0.5)
    return np.array(splitted).reshape(order_, order_).tolist()

def get_neighbours(state, order_):
    deserialized = deserialize(state)
    neighbours = []
    blank_i = -1
    blank_j = -1

    for i in range(0, order_):
        for j in range(0, order_):
            if deserialized[i][j] == 0:
                blank_i, blank_j = i, j
                break

    i = blank_i
    j = blank_j

    if i > 0:
        new_matrix = copy.deepcopy(deserialized)
        new_matrix[i][j] = new_matrix[i - 1][j]
        new_matrix[i - 1][j] = 0
        neighbours.append(serialize(new_matrix))
    if i < order_-1:
        new_matrix = copy.deepcopy(deserialized)
        new_matrix[i][j] = new_matrix[i + 1][j]
        new_matrix[i + 1][j] = 0
        neighbours.append(serialize(new_matrix))
    if j > 0:
        new_matrix = copy.deepcopy(deserialized)
        new_matrix[i][j] = new_matrix[i][j - 1]
        new_matrix[i][j - 1] = 0
        neighbours.append(serialize(new_matrix))
    if j < order_-1:
        new_matrix = copy.deepcopy(deserialized)
        new_matrix[i][j] = new_matrix[i][j + 1]
        new_matrix[i][j + 1] = 0
        neighbours.append(serialize(new_matrix))

    return zip(neighbours, [1 for x in neighbours])

def h(state):
    deserialized = deserialize(state)
    order_ = len(deserialized)
    H = 0
    for i in range(0, order_):
        for j in range(0, order_):
            H += abs(deserialized[i][j] % order_ - j) + abs(deserialized[i][j] / order_ - i)
    return H

def in_open_set_with_lowest_heuristic_guess(open_set, heuristic_guess):
    result, min_guess = None, float('inf')
    for v in open_set:
        if v in heuristic_guess:
            guess = heuristic_guess[v]
            if guess < min_guess:
                result = v
                min_guess = guess
    return result

def dijkstra(start_node):
    order_ = len(start_node)

    target_node = np.array(range(order_**2)).reshape(order_, order_).tolist()
    start_node = serialize(start_node)
    target_node = serialize(target_node)

    visited = set()
    D = defaultdict(lambda: float('inf'))
    D[start_node] = 0

    pq = PriorityQueue()
    pq.put((0, start_node))

    parent = dict()
    parent[start_node] = None
    path_found = False
    iteratrion = 0
    while not pq.empty():
        (dist, current_node) = pq.get()
        if current_node == target_node:
            path_found = True
            break
        visited.add(current_node)

        for (neighbour, distance_from_current_node) in get_neighbours(current_node, order_):
            if neighbour not in visited:
                old_cost = D[neighbour]
                new_cost = D[current_node] + distance_from_current_node
                if new_cost < old_cost:
                    pq.put((new_cost, neighbour))
                    D[neighbour] = new_cost
                    parent[neighbour] = current_node
        iteratrion += 1

    path = []
    if path_found:
        path.append(target_node)
        while True:
            parent_node = parent[target_node]
            if parent_node is None:
                break
            path.append(parent_node)
            target_node = parent_node
        path.reverse()
    return (path, iteratrion)

def astar_lloyd(start_node, h):
    order_ = len(start_node)

    target_node = np.array(range(order_**2)).reshape(order_, order_)
    start_node = serialize(start_node)
    target_node = serialize(target_node)

    open_set = set([start_node])

    parents = {}
    parents[start_node] = None

    cheapest_paths = defaultdict(lambda: float('inf'))
    cheapest_paths[start_node] = 0

    heuristic_guess = defaultdict(lambda: float('inf'))
    heuristic_guess[start_node] = h(start_node)

    path_found = False
    iteration = 0
    while len(open_set) > 0:
        # O(1)
        current_node = in_open_set_with_lowest_heuristic_guess(open_set, heuristic_guess)
        if current_node == target_node:
            path_found = True
            break

        open_set.remove(current_node)
        for (neighbour_node, weight) in get_neighbours(current_node, order_):
            new_cheapest_path = cheapest_paths[current_node] + weight
            if new_cheapest_path < cheapest_paths[neighbour_node]:
                parents[neighbour_node] = current_node
                cheapest_paths[neighbour_node] = new_cheapest_path
                heuristic_guess[neighbour_node] = new_cheapest_path + h(neighbour_node)
                if neighbour_node not in open_set:
                    open_set.add(neighbour_node)
        iteration += 1

    path = []
    if path_found:
        while target_node is not None:
            path.append(target_node)
            target_node = parents[target_node]
        path.reverse()
    return (path, iteration)

if __name__ == '__main__':
    start_state = [[2,3,5], [1,4,0], [7,8,6]]

    print('Dijkstra benchmark')
    (path, iteration) = dijkstra(start_state)
    print(path, iteration)
    print('Astar benchmark')
    (path, iteration) = astar_lloyd(start_state, h)
    print(path, iteration)