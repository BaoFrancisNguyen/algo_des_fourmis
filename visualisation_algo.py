import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from geopy.distance import geodesic


class AntColonyOptimization:
    def __init__(self, cities, n_ants=10, alpha=1, beta=2, evaporation=0.5, iterations=90, capacity=5):
        self.cities = cities  # Dictionnaire des villes avec leurs coordonnées
        self.n_cities = len(cities)
        self.distances = self._calculate_distances(
            cities)  # Matrice des distances
        self.pheromones = np.ones((self.n_cities, self.n_cities))
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.iterations = iterations
        self.capacity = capacity  # Capacité maximale d'un livreur
        self.best_distance = float('inf')
        self.best_path = None

    def _calculate_distances(self, cities):
        """Calcule la matrice des distances entre les villes."""
        n = len(cities)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    coord1 = (cities[i]['lat'], cities[i]['lon'])
                    coord2 = (cities[j]['lat'], cities[j]['lon'])
                    distances[i, j] = geodesic(coord1, coord2).km
        return distances

    def run(self):
        """Exécute l'algorithme de colonie de fourmis."""
        plt.figure(figsize=(10, 8))
        for iteration in range(self.iterations):
            all_ant_paths = self._construct_solutions()
            self._update_pheromones(all_ant_paths)

            for path, distance in all_ant_paths:
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_path = path

            self._print_iteration_details(iteration, all_ant_paths)
            self._plot_paths(all_ant_paths, iteration)
            plt.pause(0.6)  # Pause pour permettre la visualisation dynamique

        plt.show()
        return self.best_path, self.best_distance

    def _construct_solutions(self):
        """Construit les solutions pour toutes les fourmis."""
        return [self._construct_solution() for _ in range(self.n_ants)]

    def _construct_solution(self):
        """Construit une solution pour une fourmi."""
        path = [0]  # Commencer au dépôt (Paris)
        remaining_capacity = self.capacity
        unvisited = set(range(1, self.n_cities))  # Villes non visitées

        while unvisited:
            next_city = self._select_next_city(
                path[-1], unvisited, remaining_capacity)
            if next_city is None:  # Retour au dépôt si la capacité est épuisée
                path.append(0)
                remaining_capacity = self.capacity
            else:
                path.append(next_city)
                unvisited.remove(next_city)
                remaining_capacity -= 1  # Un colis est livré

        path.append(0)  # Retour au dépôt à la fin
        distance = self._calculate_path_distance(path)
        return path, distance

    def _select_next_city(self, current, unvisited, remaining_capacity):
        """Sélectionne la prochaine ville à visiter."""
        if not unvisited or remaining_capacity == 0:
            return None

        probabilities = []
        total = 0
        for j in unvisited:
            p = (self.pheromones[current, j] ** self.alpha) * \
                ((1 / self.distances[current, j]) ** self.beta)
            probabilities.append((j, p))
            total += p

        if total == 0:  # Éviter une division par zéro
            return random.choice(list(unvisited))

        probabilities = [(j, p / total) for j, p in probabilities]
        return random.choices([j for j, _ in probabilities], [p for _, p in probabilities])[0]

    def _update_pheromones(self, all_ant_paths):
        """Met à jour les phéromones sur les chemins."""
        self.pheromones *= (1 - self.evaporation)
        for path, distance in all_ant_paths:
            for i in range(len(path) - 1):
                self.pheromones[path[i], path[i + 1]] += 1.0 / distance

    def _calculate_path_distance(self, path):
        """Calcule la distance totale d'un chemin."""
        return sum(self.distances[path[i], path[i + 1]] for i in range(len(path) - 1))

    def _print_iteration_details(self, iteration, all_ant_paths):
        """Affiche les détails des trajets dans le terminal."""
        print(f"\n--- Itération {iteration + 1} ---")
        for idx, (path, distance) in enumerate(all_ant_paths):
            path_names = [self.cities[i]['name'] for i in path]
            print(
                f"Fourmi {idx + 1}: Trajet = {path_names}, Distance = {distance:.2f} km")

    def _plot_paths(self, all_ant_paths, iteration):
        """Affiche les chemins des fourmis sur une carte."""
        plt.clf()
        plt.title(
            f"Iteration {iteration + 1} - Best Distance: {self.best_distance:.2f} km")

        # Afficher les villes
        lats = [self.cities[i]['lat'] for i in range(self.n_cities)]
        lons = [self.cities[i]['lon'] for i in range(self.n_cities)]
        plt.scatter(lons, lats, c='red', s=100, label="Villes")
        for i, city in enumerate(self.cities):
            plt.text(city['lon'], city['lat'], f"{
                     city['name']}", fontsize=9, ha='right')

        # Afficher les chemins des livreurs
        for path, distance in all_ant_paths:
            x = [self.cities[i]['lon'] for i in path]
            y = [self.cities[i]['lat'] for i in path]
            plt.plot(x, y, marker='o', linestyle='-', alpha=0.5,
                     label=f"Distance: {distance:.2f} km")
            for i in range(len(path) - 1):
                mid_x = (self.cities[path[i]]['lon'] +
                         self.cities[path[i + 1]]['lon']) / 2
                mid_y = (self.cities[path[i]]['lat'] +
                         self.cities[path[i + 1]]['lat']) / 2
                plt.text(mid_x, mid_y, f"{
                         self.distances[path[i], path[i + 1]]:.1f} km", fontsize=8, color='blue')

        # Afficher le meilleur chemin
        if self.best_path:
            x = [self.cities[i]['lon'] for i in self.best_path]
            y = [self.cities[i]['lat'] for i in self.best_path]
            plt.plot(x, y, marker='o', linestyle='-', color='green',
                     linewidth=2, label=f"Best Path: {self.best_distance:.2f} km")

        plt.legend(loc="upper right")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        # plt.xlim(1.5, 3.5)  # Centrer sur l'Île-de-France (longitude)
        # plt.ylim(48.4, 49.2)  # Centrer sur l'Île-de-France (latitude)
        plt.draw()


# Charger le dataset avec pandas
def load_dataset(file_path, n_rows=10):
    """Charge le dataset avec pandas et extrait les informations nécessaires."""
    df = pd.read_csv(file_path, delimiter=';').head(n_rows)
    cities = []
    for _, row in df.iterrows():
        lat, lon = map(float, row['geo_point_2d'].split(','))
        cities.append({
            'name': f"{row['ADRESSE']}, {row['COMMUNE']}",
            'lat': lat,
            'lon': lon,
            'h_debut': row['H_DEBUT'],
            'h_fin': row['H_FIN'],
            'tonnage': float(row['TONNAGE'])
        })
    return cities


if __name__ == '__main__':
    # Charger le dataset
    file_path = 'aires-de-livraison.csv'
    cities = load_dataset(file_path)

    # Exemple d'utilisation
    aco = AntColonyOptimization(cities, n_ants=10, capacity=3, iterations=50)
    best_path, best_distance = aco.run()
    print("\n--- Meilleur trajet trouvé ---")
    print("Trajet:", [cities[i]['name'] for i in best_path])
    print("Distance totale:", best_distance, "km")
