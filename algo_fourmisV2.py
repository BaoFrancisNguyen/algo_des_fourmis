import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import webbrowser
from math import radians, sin, cos, sqrt, atan2

# ---- CHARGEMENT DES DONNÉES ---- #
print("Chargement des données...")
file_path = "aires-de-livraison.csv"
df_villes = pd.read_csv(file_path, delimiter=";")
print(f"Nombre de points de livraison : {len(df_villes)}")

# Extraction des coordonnées GPS
df_villes[['Latitude', 'Longitude']] = df_villes['geo_point_2d'].str.split(',', expand=True)
df_villes['Latitude'] = df_villes['Latitude'].astype(float)
df_villes['Longitude'] = df_villes['Longitude'].astype(float)

# Liste des villes et coordonnées
noms_villes = df_villes['ADRESSE'].sample(n=6, random_state=42).tolist()
villes = df_villes[['Latitude', 'Longitude']].sample(n=6, random_state=42).to_numpy()
NUM_VILLES = 6
print(f"Nombre total de villes chargées : {NUM_VILLES}")

# ---- CALCUL DES DISTANCES (Haversine) ---- #
R = 6371  # Rayon de la Terre en kilomètres
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def calculer_distances(villes):
    num_villes = len(villes)
    distances = np.zeros((num_villes, num_villes))
    for i in range(num_villes):
        for j in range(num_villes):
            if i != j:
                distances[i, j] = haversine(villes[i][0], villes[i][1], villes[j][0], villes[j][1])
    return distances

distances = calculer_distances(villes)
print("Distances calculées avec succès (en km).")

# Sélection d'un entrepôt aléatoire
entrepot_index = np.random.randint(NUM_VILLES)
entrepot_coords = villes[entrepot_index]
print(f"Entrepôt sélectionné : {noms_villes[entrepot_index]} ({entrepot_coords})")

# ---- PARAMÈTRES ACO ---- #
NUM_FOURMIS = 20
ITERATIONS = 100
EVAPORATION = 0.5
ALPHA = 1
BETA = 2
Q = 100
pheromones = np.ones((NUM_VILLES, NUM_VILLES))
print("Paramètres de l'algorithme initialisés.")

# ---- FONCTIONS DE L'ALGORITHME ---- #
def choisir_prochaine_ville(ville_actuelle, villes_non_visitées, pheromones, distances):
    probabilites = []
    for ville in villes_non_visitées:
        attrait = (pheromones[ville_actuelle, ville] ** ALPHA) * ((1 / distances[ville_actuelle, ville]) ** BETA)
        probabilites.append(attrait)
    
    probabilites = np.array(probabilites)
    somme_probabilites = np.sum(probabilites)
    
    if somme_probabilites == 0:
        return np.random.choice(villes_non_visitées)  # Sélection aléatoire si aucune probabilité valide
    
    probabilites /= somme_probabilites  # Normalisation
    return np.random.choice(villes_non_visitées, p=probabilites)

def simulation_fourmis():
    global pheromones
    meilleur_chemin = None
    meilleure_distance = float('inf')

    print("Début de la simulation des fourmis...")
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
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration} : meilleure distance = {meilleure_distance:.2f} km")
        
        pheromones *= (1 - EVAPORATION)
        for chemin, longueur in zip(chemins, longueurs_chemins):
            contribution = Q / longueur
            for i in range(len(chemin) - 1):
                pheromones[chemin[i], chemin[i+1]] += contribution
                pheromones[chemin[i+1], chemin[i]] += contribution
    
    print("Simulation terminée.")
    return meilleur_chemin, meilleure_distance

# ---- EXÉCUTION ---- #
chemin_optimal, distance_minimale = simulation_fourmis()
print("Chemin optimal trouvé :", chemin_optimal)
print(f"Distance minimale parcourue : {distance_minimale:.2f} km")

# ---- AFFICHAGE SUR FOLIUM ---- #
latitude_centre = np.mean(villes[:, 0])
longitude_centre = np.mean(villes[:, 1])
map_livraison = folium.Map(location=[latitude_centre, longitude_centre], zoom_start=12)

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

# Ouvrir automatiquement la carte
webbrowser.open("carte_livraison.html")
