import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import folium

# ---- PARAMÈTRES ---- #
NUM_VILLES = 20  # Nombre de villes
NUM_FOURMIS = 20  # Nombre de fourmis simulées
ITERATIONS = 100  # Nombre d'itérations
EVAPORATION = 0.5  # Taux d'évaporation des phéromones
ALPHA = 1  # Importance des phéromones
BETA = 2  # Importance de la distance
Q = 100  # Quantité de phéromone déposée

# ---- GÉNÉRATION DES DONNÉES ---- #
np.random.seed(42)

# Génération de villes avec des coordonnées aléatoires
villes = np.random.rand(NUM_VILLES, 2) * 100
noms_villes = [f'Ville_{i+1}' for i in range(NUM_VILLES)]

# Calcul des distances entre villes
def calculer_distances(villes):
    num_villes = len(villes)
    distances = np.zeros((num_villes, num_villes))
    for i in range(num_villes):
        for j in range(num_villes):
            if i != j:
                distances[i, j] = np.linalg.norm(villes[i] - villes[j])
    return distances

distances = calculer_distances(villes)

# Sélection d'un entrepôt aléatoire
entrepot_index = np.random.randint(NUM_VILLES)
entrepot_coords = villes[entrepot_index]

# ---- INITIALISATION DES PHÉROMONES ---- #
pheromones = np.ones((NUM_VILLES, NUM_VILLES))

# ---- FONCTIONS DE L'ALGORITHME ---- #
def choisir_prochaine_ville(ville_actuelle, villes_non_visitées, pheromones, distances):
    probabilites = []
    for ville in villes_non_visitées:
        attrait = (pheromones[ville_actuelle, ville] ** ALPHA) * ((1 / distances[ville_actuelle, ville]) ** BETA)
        probabilites.append(attrait)
    probabilites = np.array(probabilites) / np.sum(probabilites)
    return np.random.choice(villes_non_visitées, p=probabilites)

def simulation_fourmis():
    global pheromones
    meilleur_chemin = None
    meilleure_distance = float('inf')

    for iteration in range(ITERATIONS):
        chemins = []
        longueurs_chemins = []
        
        for _ in range(NUM_FOURMIS):
            ville_depart = entrepot_index
            chemin = [ville_depart]
            villes_restantes = list(range(NUM_VILLES))
            villes_restantes.remove(ville_depart)
            
            while villes_restantes:
                prochaine_ville = choisir_prochaine_ville(chemin[-1], villes_restantes, pheromones, distances)
                chemin.append(prochaine_ville)
                villes_restantes.remove(prochaine_ville)
            
            chemin.append(ville_depart)
            longueur = sum(distances[chemin[i], chemin[i+1]] for i in range(len(chemin)-1))
            chemins.append(chemin)
            longueurs_chemins.append(longueur)
            
            if longueur < meilleure_distance:
                meilleure_distance = longueur
                meilleur_chemin = chemin

        pheromones *= (1 - EVAPORATION)
        for chemin, longueur in zip(chemins, longueurs_chemins):
            contribution = Q / longueur
            for i in range(len(chemin) - 1):
                pheromones[chemin[i], chemin[i+1]] += contribution
                pheromones[chemin[i+1], chemin[i]] += contribution

    return meilleur_chemin, meilleure_distance

# ---- EXÉCUTION ---- #
chemin_optimal, distance_minimale = simulation_fourmis()

# ---- AFFICHAGE SUR FOLIUM ---- #
latitude_centre = np.mean(villes[:, 0])
longitude_centre = np.mean(villes[:, 1])
map_livraison = folium.Map(location=[latitude_centre, longitude_centre], zoom_start=10)

# Ajouter l'entrepôt
folium.Marker(
    location=[entrepot_coords[0], entrepot_coords[1]],
    popup="Entrepôt",
    icon=folium.Icon(color="red", icon="home"),
).add_to(map_livraison)

# Ajouter les villes
for i, ville in enumerate(villes):
    folium.Marker(
        location=[ville[0], ville[1]],
        popup=noms_villes[i],
        icon=folium.Icon(color="blue"),
    ).add_to(map_livraison)

# Ajouter le trajet optimal
chemin_coords = [[villes[i][0], villes[i][1]] for i in chemin_optimal]
folium.PolyLine(chemin_coords, color="black", weight=2.5, opacity=1).add_to(map_livraison)

# Sauvegarde de la carte
map_livraison.save("carte_livraison.html")
print("Carte sauvegardée sous 'carte_livraison.html'")
