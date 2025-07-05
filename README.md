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

# 🌐 TRANSFORMERS — EXPLICATION SIMPLE

## 🧠 Vue d'ensemble : Qu'est-ce qu'un Transformer ?

Un **Transformer** est comme un traducteur super intelligent qui peut comprendre et générer du texte. Il est composé de deux parties principales :

- **L'Encodeur** (à gauche) : lit et comprend le texte d'entrée  
- **Le Décodeur** (à droite) : génère le texte de sortie  

> 🎯 Imaginez que vous donnez une phrase en français à un ami polyglotte : il la comprend (encodeur), puis la traduit en anglais (décodeur).

---

## ⚙️ Fonctionnement étape par étape

### 1️⃣ Input/Output Embeddings (Plongements d'entrée/sortie)

**Rôle** : Convertir les mots en vecteurs numériques que l'ordinateur comprend.  
**Analogie** : Chaque mot reçoit un "code" unique :

- "chat" → [0.2, 0.8, 0.1, ...]  
- "chien" → [0.3, 0.7, 0.2, ...]  
- "amour" → [0.9, 0.1, 0.8, ...]

Les mots proches en sens ont des vecteurs similaires.

---

### 2️⃣ Positional Encoding (Encodage positionnel)

**Rôle** : Indiquer la position de chaque mot dans la phrase.  
**Pourquoi ?** : L’ordre change le sens !

- "Le chat mange le poisson" ≠ "Le poisson mange le chat"  
**Analogie** : On numérote chaque mot :

- "Le¹ chat² mange³ le⁴ poisson⁵"

---

### 3️⃣ Multi-Head Attention (Attention multi-têtes)

**Rôle** : Chaque mot "regarde" les autres pour comprendre le contexte.  
**Analogie** : Votre cerveau établit des liens :

- Dans "Marie donne sa pomme à Jean", le mot **"sa"** fait référence à **"Marie"**.

**Pourquoi plusieurs têtes ?**

- Une tête analyse la grammaire  
- Une autre le sens  
- Une autre les relations entre entités

---

### 4️⃣ Masked Multi-Head Attention (Décodeur uniquement)

**Rôle** : Empêcher le modèle de "voir le futur" lors de la génération.  
**Analogie** : Comme le jeu du "cadavre exquis" — on ne voit que les mots déjà générés.

---

### 5️⃣ Add & Norm (Addition et Normalisation)

**Rôle** : Stabiliser les valeurs pendant l’apprentissage.  
**Analogie** : Un régulateur dans une voiture :

- `Add` : Ajoute les nouvelles infos  
- `Norm` : Normalise l’échelle

---

### 6️⃣ Feed Forward (Réseau de neurones)

**Rôle** : Améliorer individuellement chaque mot avec son contexte.  
**Analogie** : Comme un filtre qui polit chaque mot.

---

### 7️⃣ Linear + Softmax (Couche finale)

**Rôle** : Prédire le mot suivant en générant des probabilités.  
**Exemple** :

- "chat" : 60%  
- "chien" : 30%  
- "oiseau" : 10%

---

## 🔄 Flux d’information complet

### 📘 Dans l'Encodeur :

