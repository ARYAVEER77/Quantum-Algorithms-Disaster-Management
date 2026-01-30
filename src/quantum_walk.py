import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def simulate_quantum_walk():

    G = nx.Graph()

    G.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F'])

    edges = [
        ('A', 'B', 1),
        ('A', 'C', 2),
        ('B', 'D', 1),
        ('C', 'D', 3),
        ('C', 'E', 2),
        ('D', 'F', 1),
        ('E', 'F', 1)
    ]

    for u, v, w in edges:
        G.add_edge(u, v, weight=w)


    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=800, font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Evacuation Route Graph")
    plt.show()

    A = nx.to_numpy_array(G)
    print("\nAdjacency Matrix:")
    print(A)

    from scipy.linalg import expm

    H = A
    t = 1  
    U = expm(-1j * H * t)  

   
    initial_state = np.zeros(len(G.nodes))
    initial_state[0] = 1

    final_state = np.dot(U, initial_state)
    probabilities = np.abs(final_state) ** 2

    print("\nQuantum Walk Probabilities:")
    for node, prob in zip(G.nodes, probabilities):
        print(f"{node}: {prob:.4f}")

    return dict(zip(G.nodes, probabilities))


if __name__ == "__main__":
    simulate_quantum_walk()