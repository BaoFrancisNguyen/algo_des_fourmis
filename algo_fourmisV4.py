import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import webbrowser
import osmnx as ox
import networkx as nx

# ---- CHARGEMENT DES DONNÉES ---- #
print("Chargement des données...")
file_path = "aires-de-livraison.csv"
df_villes = pd.read_csv(file_path, delimiter=";")
print(f"Nombre de points de livraison : {len(df_villes)}")

# Demander les paramètres à l'utilisateur
NUM_VILLES = int(input('Nombre de villes à inclure dans la simulation : '))
NUM_FOURMIS = int(input('Nombre de fourmis dans la simulation : '))
ITERATIONS = int(input('Nombre d\'itérations de l\'algorithme : '))

# Extraction des coordonnées GPS
df_villes[['Latitude', 'Longitude']] = df_villes['geo_point_2d'].str.split(',', expand=True)
df_villes['Latitude'] = df_villes['Latitude'].astype(float)
df_villes['Longitude'] = df_villes['Longitude'].astype(float)

# Sélection aléatoire des villes et coordonnées
noms_villes = df_villes['ADRESSE'].sample(n=NUM_VILLES, random_state=42).tolist()
villes = df_villes[['Latitude', 'Longitude']].sample(n=NUM_VILLES, random_state=42).to_numpy()
print(f"Nombre total de villes sélectionnées : {NUM_VILLES}")

# ---- CALCUL DES DISTANCES AVEC OSM ---- #
def calculer_distances_osm(villes):
    print("Chargement du réseau routier OSM...")
    G = ox.graph_from_point((np.mean(villes[:, 0]), np.mean(villes[:, 1])), dist=10000, network_type='drive')
    print("Réseau routier chargé. Calcul des distances en cours...")
    
    nodes = [ox.distance.nearest_nodes(G, ville[1], ville[0]) for ville in villes]
    num_villes = len(villes)
    distances = np.zeros((num_villes, num_villes))
    
    for i in range(num_villes):
        for j in range(num_villes):
            if i != j:
                try:
                    distances[i, j] = nx.shortest_path_length(G, nodes[i], nodes[j], weight='length') / 1000  # Convertir en km
                except nx.NetworkXNoPath:
                    distances[i, j] = float('inf')  # Pas de route trouvée entre ces points
    
    print("Distances routières calculées avec succès !")
    return distances

distances = calculer_distances_osm(villes)

# Sélection d'un entrepôt aléatoire
entrepot_index = np.random.randint(NUM_VILLES)
entrepot_coords = villes[entrepot_index]
print(f"Entrepôt sélectionné : {noms_villes[entrepot_index]} ({entrepot_coords})")

# ---- PARAMÈTRES ACO ---- #
EVAPORATION = 0.5
ALPHA = 1
BETA = 2
Q = 100
pheromones = np.ones((NUM_VILLES, NUM_VILLES))
print("Paramètres de l'algorithme initialisés.")

# ---- FONCTION POUR AFFICHER LES PHÉROMONES SUR LA CARTE ---- #
#Trajets affichés avec une intensité proportionnelle au niveau de phéromones

# ---- FONCTION POUR AFFICHER LES PHÉROMONES SUR LA CARTE ---- #
def afficher_pheromones_sur_carte():
    print("Génération de la carte avec trajets routiers...")
    G = ox.graph_from_point((np.mean(villes[:, 0]), np.mean(villes[:, 1])), dist=10000, network_type='drive')
    map_livraison = folium.Map(location=[np.mean(villes[:, 0]), np.mean(villes[:, 1])], zoom_start=12)
    
    # Ajouter les villes et entrepôt
    for i, ville in enumerate(villes):
        folium.CircleMarker(
            location=[ville[0], ville[1]],
            radius=5,
            color='blue' if i != entrepot_index else 'red',
            fill=True,
            fill_color='blue' if i != entrepot_index else 'red',
            popup=noms_villes[i]
        ).add_to(map_livraison)
    
    # Ajouter les chemins en suivant le réseau routier (phéromones en vert, chemin optimal en rouge)
    for i in range(NUM_VILLES):
        for j in range(NUM_VILLES):
            if i != j:
                try:
                    route = nx.shortest_path(G, ox.distance.nearest_nodes(G, villes[i][1], villes[i][0]), ox.distance.nearest_nodes(G, villes[j][1], villes[j][0]), weight='length')
                    route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
                    
                    if (i, j) in zip(chemin_optimal, chemin_optimal[1:]):
                        color = 'red'  # Chemin optimal
                        weight = 4
                    else:
                        color = 'green'  # Phéromones
                        weight = 2
                    
                    folium.PolyLine(route_coords, color=color, weight=weight, opacity=0.7).add_to(map_livraison)
                except nx.NetworkXNoPath:
                    continue
    
    # Sauvegarde et ouverture
    map_livraison.save("carte_pheromones.html")
    webbrowser.open("carte_pheromones.html")



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
# ---- FONCTIONS DE L'ALGORITHME ---- #

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

# Afficher les phéromones sur la carte
afficher_pheromones_sur_carte()