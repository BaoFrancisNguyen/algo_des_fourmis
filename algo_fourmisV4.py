import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import webbrowser
import osmnx as ox
import networkx as nx
from tqdm import tqdm

# ---- CHARGEMENT DES DONNÉES ---- #
print("Chargement des données...")
file_path = "aires-de-livraison.csv"
df_villes = pd.read_csv(file_path, delimiter=";")
print(f"Nombre de points de livraison : {len(df_villes)}")

# Demander les paramètres à l'utilisateur pour la simulation / nombre de villes, fourmis et itérations
NUM_VILLES = int(input('Nombre de villes à inclure dans la simulation : '))
NUM_FOURMIS = int(input('Nombre de fourmis dans la simulation : '))
ITERATIONS = int(input('Nombre d\'itérations de l\'algorithme : '))

# Extraction des coordonnées GPS de la colonne 'geo_point_2d' / à couper en deux
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
    
    for i in tqdm(range(num_villes), desc="Calcul des distances"):
        for j in range(num_villes):
            if i != j:
                try:
                    distances[i, j] = nx.shortest_path_length(G, nodes[i], nodes[j], weight='length') / 1000  # Convertir en km
                except nx.NetworkXNoPath:
                    distances[i, j] = float('inf')  # Pas de route trouvée entre ces points
    
    print("Distances routières calculées avec succès !")
    return distances

distances = calculer_distances_osm(villes)

# Sélection d'un entrepôt aléatoire / adresse de l'entrepôt et coordonnées
entrepot_index = np.random.randint(NUM_VILLES)
entrepot_coords = villes[entrepot_index]
print(f"Entrepôt sélectionné : {noms_villes[entrepot_index]} ({entrepot_coords})")

# ---- PARAMÈTRES ACO ---- #

EVAPORATION = 0.7 # Taux d'évaporation des phéromones entre chaque itération /exemple : 0.5 → 50% des phéromones s'évaporent à chaque itération
ALPHA = 1 # Poids des phéromones  /Si ALPHA est élevé → Les fourmis suivent fortement les chemins déjà empruntés, 
#si ALPHA est faible → Les fourmis ont tendance à explorer de nouveaux chemins
BETA = 2 # (Attractivité heuristique - inverse de la distance) / Poids des distances /Si BETA est élevé → 
#Les fourmis privilégient les chemins les plus courts, si BETA est faible → Les fourmis privilégient les chemins avec plus de phéromones
Q = 500 # Quantité de phéromones déposée

# ayant testé plusieurs valeurs, les paramètres suivants ont été retenus :*
# avec Q = 100, ALPHA = 1, BETA = 2 et EVAPORATION = 0.5 >>> constatation : allers-retours inutiles
# ALPHA = 1, BETA = 2, Q = 500, EVAPORATION = 0.7


#Pour favoriser l'exploration → Diminue ALPHA, augmente BETA.
#Pour renforcer les chemins déjà découverts → Augmente ALPHA, diminue BETA.
#Pour accélérer la convergence vers une solution → Augmente Q, mais cela peut réduire la diversité des solutions.

pheromones = np.ones((NUM_VILLES, NUM_VILLES))
print("Paramètres de l'algorithme initialisés.")

# ---- FONCTION POUR AFFICHER LES PHÉROMONES SUR LA CARTE ---- #
#Trajets affichés avec une intensité proportionnelle au niveau de phéromones

def afficher_pheromones_sur_carte():
    print("Génération de la carte avec trajets routiers...")
    G = ox.graph_from_point((np.mean(villes[:, 0]), np.mean(villes[:, 1])), dist=10000, network_type='drive')
    map_livraison = folium.Map(location=[np.mean(villes[:, 0]), np.mean(villes[:, 1])], zoom_start=12)
    
    # Afficher les villes dans l'ordre optimal
    for index, ville_index in enumerate(chemin_optimal):
        ville = villes[ville_index]
        folium.Marker(
            location=[ville[0], ville[1]],
            icon=folium.DivIcon(html=f'<div style="font-size: 12pt; font-weight: bold; color: black;">{index+1}</div>'),
            popup=f"{noms_villes[ville_index]} (Stop {index+1})"
        ).add_to(map_livraison)
    
    # Ajouter les phéromones en vert
    for i in range(NUM_VILLES):
        for j in range(NUM_VILLES):
            if i != j and pheromones[i, j] > 0:
                try:
                    route = nx.shortest_path(G, ox.distance.nearest_nodes(G, villes[i][1], villes[i][0]), ox.distance.nearest_nodes(G, villes[j][1], villes[j][0]), weight='length')
                    route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
                    folium.PolyLine(route_coords, color='blue', weight=2, opacity=0.5).add_to(map_livraison)
                except nx.NetworkXNoPath:
                    continue
    
    # Ajouter le chemin optimal en rouge avec flèches
    for i in tqdm(range(len(chemin_optimal) - 1), desc="Affichage des trajets"):
        try:
            route = nx.shortest_path(G, ox.distance.nearest_nodes(G, villes[chemin_optimal[i]][1], villes[chemin_optimal[i]][0]), ox.distance.nearest_nodes(G, villes[chemin_optimal[i+1]][1], villes[chemin_optimal[i+1]][0]), weight='length')
            route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
            for k in range(len(route_coords) - 1):
                folium.PolyLine([route_coords[k], route_coords[k+1]], color='red', weight=6, opacity=0.7, dash_array='5,5').add_to(map_livraison)
                
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

def initialiser_solution_gloutonne():
    """Crée une première solution basée sur la ville la plus proche"""
    ville_depart = entrepot_index
    chemin = [ville_depart]
    villes_restantes = list(range(NUM_VILLES))
    villes_restantes.remove(ville_depart)

    while villes_restantes:
        derniere_ville = chemin[-1]
        prochaine_ville = min(villes_restantes, key=lambda x: distances[derniere_ville, x])
        chemin.append(prochaine_ville)
        villes_restantes.remove(prochaine_ville)

    chemin.append(ville_depart)  # Retour à l'entrepôt
    return chemin

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

def amelioration_2_opt(chemin):
    """Applique l'algorithme 2-opt pour supprimer les croisements inutiles"""
    amelioration = True
    while amelioration:
        amelioration = False
        for i in range(1, len(chemin) - 2):
            for j in range(i + 1, len(chemin) - 1):
                if distances[chemin[i - 1], chemin[j]] + distances[chemin[i], chemin[j + 1]] < \
                   distances[chemin[i - 1], chemin[i]] + distances[chemin[j], chemin[j + 1]]:
                    chemin[i:j + 1] = reversed(chemin[i:j + 1])
                    amelioration = True
    return chemin

chemin_optimal = amelioration_2_opt(chemin_optimal)

print("Chemin optimal trouvé :", chemin_optimal)
print(f"Distance minimale parcourue : {distance_minimale:.2f} km")

# Afficher les phéromones sur la carte
afficher_pheromones_sur_carte()