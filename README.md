# 🧠 GenIA & NLP – Bases et Projets avec LangChain

Bienvenue dans ce dépôt consacré à l'**Intelligence Artificielle Générative (GenIA)** et au **Traitement Automatique du Langage Naturel (NLP)**. Il est structuré en deux grandes parties :  
- **Les fondamentaux du NLP en machine learning and Deep learning**
- **L'architecture des Transformers**
- **Des projets basics et avancés basés sur LangChain et les LLMs + deployment**


```markdown
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
