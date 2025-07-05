# 🧠 GenIA & NLP – Bases et Projets avec LangChain

Bienvenue dans ce dépôt consacré à l'**Intelligence Artificielle Générative (GenIA)** et au **Traitement Automatique du Langage Naturel (NLP)**. Il est structuré en deux grandes parties :  
- **Les fondamentaux du NLP en machine learning and Deep learning**
- **L'architecture des Transformers**
- **Des projets basics et avancés basés sur LangChain et les LLMs + deployment**


# Concepts de Traitement du Langage Naturel (NLP)

Ce document couvre les techniques fondamentales de prétraitement et de représentation de texte en NLP. Voici une explication détaillée :

## 📝 Prétraitement de Texte
- **Objectif** : Préparer le texte brut pour les algorithmes de machine learning
- **Étapes courantes** :
  - `Tokenisation` (découpage du texte en mots/jetons)
  - `Suppression des stopwords` (mots courants comme "le", "est")
  - `Lemmatisation` (réduction des mots à leur forme de base)
  - `Nettoyage` (gestion des caractères spéciaux et ponctuation)

## 🔢 Techniques de Représentation de Texte

### 1️⃣ Sac de Mots (Bag of Words - BoW)
- **Concept** : Représente le texte par des fréquences de mots
- **Exemple** :
  ```python
  Document 1 : "La nourriture est bonne" → {"la":1, "nourriture":1, "est":1, "bonne":1}
  Document 2 : "La nourriture est mauvaise" → {"la":1, "nourriture":1, "est":1, "mauvaise":1}
  ```
- **Limitations** :
  - ❌ Matrices creuses
  - ❌ Pas de sens sémantique
  - ❌ Problème de mots hors vocabulaire (OOV)

### 2️⃣ TF-IDF (Term Frequency-Inverse Document Frequency)
- **Concept** : Pondere les mots par leur importance relative dans un document
- **Calcul** :
  ```
  TF = (nb d'occurrences du mot dans le doc) / (nb total de mots du doc)
  IDF = log(nb total de docs / nb de docs contenant le mot)
  TF-IDF = TF * IDF
  ```
- **Avantages** :
  - ✅ Capture l'importance des mots
  - ✅ Taille fixe (taille du vocabulaire)

### 3️⃣ Word Embeddings (Word2Vec)
- **Concept** : Représente les mots comme des vecteurs denses
- **Propriétés** :
  - Similarité sémantique préservée
  - Relations du type : roi - homme + femme ≈ reine
- **Types** :
  - `CBOW` : Prédit le mot cible à partir du contexte
  - `Skip-gram` : Prédit le contexte à partir du mot cible
- **Avantages** :
  - ✅ Vecteurs denses
  - ✅ Capture le sens sémantique
  - ✅ Gère les mots OOV

### 4️⃣ Moyenne de Word2Vec
- **Concept** : Représente des documents par la moyenne des vecteurs de leurs mots
- **Exemple** :
  ```python
  Document : "La nourriture est bonne"
  Vecteur = moyenne(Word2Vec("la"), Word2Vec("nourriture"), ...)
  ```

## 📐 Similarité Cosinus
- Mesure la similarité entre vecteurs
- `Distance = 1 - Similarité Cosinus`
- Plage : 0 (différent) à 1 (identique)

## 🛠 Outils d'Implémentation
- Bibliothèques Python :
  - `scikit-learn` (pour BoW, TF-IDF)
  - `Gensim` (pour Word2Vec)
  - `NLTK/spaCy` (pour le prétraitement)

## 📊 Comparatif des Méthodes
| Méthode       | Sémantique | Densité | Taille | OOV  |
|---------------|------------|---------|--------|------|
| BoW           | ❌         | Creuse  | Variable | ❌   |
| TF-IDF        | ⚠️         | Creuse  | Fixe   | ❌   |
| Word2Vec      | ✅         | Dense   | Fixe   | ✅   |

Ce résumé présente les techniques essentielles pour convertir du texte en représentations numériques exploitables par le machine learning.

# *TRANSFORMERS ---EXPLICATION SIMPLE*
L'Architecture des Transformers Expliquée Simplement
Vue d'ensemble : Qu'est-ce qu'un Transformer ?
Un Transformer est comme un traducteur super intelligent qui peut comprendre et générer du texte. Il est composé de deux parties principales :

L'Encodeur (à gauche) : lit et comprend le texte d'entrée
Le Décodeur (à droite) : génère le texte de sortie

Imaginez que vous donnez une phrase en français à un ami polyglotte, il la comprend (encodeur) puis la traduit en anglais (décodeur).
Fonctionnement étape par étape
1. Input/Output Embeddings (Plongements d'entrée/sortie)
Rôle : Convertir les mots en nombres que l'ordinateur peut comprendre
Analogie : C'est comme donner un code numérique à chaque mot :

"chat" → [0.2, 0.8, 0.1, ...]
"chien" → [0.3, 0.7, 0.2, ...]
"amour" → [0.9, 0.1, 0.8, ...]

Les mots similaires ont des codes similaires.
2. Positional Encoding (Encodage positionnel)
Rôle : Indiquer la position de chaque mot dans la phrase
Pourquoi c'est important : L'ordre des mots change le sens !

"Le chat mange le poisson" ≠ "Le poisson mange le chat"

Analogie : C'est comme numéroter les mots dans une phrase :

"Le¹ chat² mange³ le⁴ poisson⁵"

3. Multi-Head Attention (Attention multi-têtes)
Rôle : Permettre à chaque mot de "regarder" et comprendre sa relation avec tous les autres mots
Analogie : Imaginez que vous lisez une phrase et que votre cerveau fait automatiquement des liens :

Dans "Marie donne sa pomme à Jean", le mot "sa" se réfère à "Marie"
L'attention permet au modèle de faire ces connexions

Pourquoi "Multi-Head" (plusieurs têtes) ? : C'est comme avoir plusieurs types d'attention :

Une tête se concentre sur la grammaire
Une autre sur le sens
Une troisième sur les relations entre personnes

4. Masked Multi-Head Attention (Attention masquée)
Rôle : Empêcher le décodeur de "tricher" en regardant les mots futurs
Analogie : C'est comme jouer au jeu du "cadavre exquis" où vous ne pouvez voir que les mots déjà écrits, pas ceux qui viennent après.
Quand le modèle génère "Le chat mange...", il ne doit pas voir que le mot suivant est "poisson" pour que l'apprentissage soit honnête.
5. Add & Norm (Addition et Normalisation)
Rôle : Stabiliser l'apprentissage et éviter que les valeurs deviennent trop grandes ou trop petites
Analogie : C'est comme un régulateur dans une voiture qui maintient la vitesse stable :

Add : Combine l'ancienne information avec la nouvelle
Norm : Remet tout à une échelle "normale"

6. Feed Forward (Réseau de neurones)
Rôle : Traiter l'information de chaque position indépendamment
Analogie : C'est comme un filtre qui améliore chaque mot individuellement :

Prend un mot avec son contexte
Le "polit" et l'améliore
Le renvoie plus "intelligent"

7. Linear + Softmax (Couche finale)
Rôle : Convertir les nombres finaux en probabilités pour chaque mot possible
Analogie : C'est comme un vote pour décider quel mot vient ensuite :

"chat" : 60% de chance
"chien" : 30% de chance
"oiseau" : 10% de chance

Flux complet d'information
Dans l'Encodeur :

Mots → Embeddings (conversion en nombres)
+ Positional Encoding (ajout de la position)
Multi-Head Attention (les mots se "parlent" entre eux)
Add & Norm (stabilisation)
Feed Forward (traitement individuel)
Add & Norm (stabilisation finale)

Dans le Décodeur :

Mots de sortie → Embeddings
+ Positional Encoding
Masked Multi-Head Attention (attention sur les mots déjà générés)
Add & Norm
Multi-Head Attention (attention sur l'encodeur)
Add & Norm
Feed Forward
Add & Norm
Linear + Softmax (choix du prochain mot)

Pourquoi cette architecture est révolutionnaire ?
1. Parallélisation

Contrairement aux anciens modèles qui lisaient mot par mot
Les Transformers peuvent traiter tous les mots en même temps
Analogie : C'est comme lire une page entière d'un coup au lieu de lettre par lettre

2. Attention globale

Chaque mot peut "voir" tous les autres mots
Permet de comprendre des relations complexes
Analogie : C'est comme avoir une vue d'ensemble d'un puzzle au lieu de voir pièce par pièce

3. Flexibilité

Peut traiter des phrases de longueur variable
S'adapte automatiquement au contexte
Analogie : C'est comme un élastique qui s'adapte à ce qu'on y met

Exemples concrets d'utilisation
Traduction

Entrée : "Hello, how are you?"
Sortie : "Bonjour, comment allez-vous ?"

Génération de texte

Entrée : "Il était une fois"
Sortie : "Il était une fois un royaume lointain où vivait une princesse..."

Résumé de texte

Entrée : Long article de presse
Sortie : Résumé en 3 phrases

Métaphore globale
Imaginez un orchestre symphonique :

Les musiciens = les mots
Le chef d'orchestre = l'attention (coordonne tout le monde)
Les partitions = les embeddings (donnent les instructions)
La disposition sur scène = l'encodage positionnel
Le concert final = la sortie du modèle

Chaque musicien (mot) écoute les autres (attention), suit sa partition (embedding), connaît sa place (position), et ensemble ils créent une symphonie (texte cohérent).
Analogie avec la lecture humaine
Quand vous lisez cette phrase : "Marie a donné sa pomme rouge à Jean car il avait faim"
Votre cerveau fait automatiquement :

Embedding : Comprend chaque mot
Position : Sait l'ordre des mots
Attention : Comprend que "sa" = Marie, "il" = Jean, "rouge" = pomme
Contexte : Comprend la relation cause-effet entre la faim et le don

Les Transformers imitent ce processus naturel !