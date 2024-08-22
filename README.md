# Benchmark et Développement d'une Interface pour l'Analyse des Données scRNA-seq

## Introduction

Ce projet vise à benchmarker et développer une interface pour l'analyse des données de séquençage de l'ARN à cellule unique (scRNA-seq). Nous avons exploité la puissance des GPU en utilisant RAPIDS pour optimiser le traitement des données tout en comparant les performances avec des exécutions sur CPU.

## Configuration de l'Environnement

### Python et RAPIDS

- **Python Version :** 3.11.9
- **RAPIDS Version :** 24.06
- **rapids-singlecell (GPU) :** 0.10.6
- **Scanpy (CPU) :** 1.10.2

Nous avons utilisé Python version 3.11.9 et installé RAPIDS version 24.06 pour optimiser le traitement des données sur GPU.

## Description du Dataset

Le jeu de données utilisé pour cette étude provient de cellules cérébrales de souris, séquencées par la méthode 10X Genomics. Les données sont disponibles publiquement à l'adresse suivante :

- **[Dataset 10X Genomics - 1M cellules cérébrales de souris](https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/1M_brain_cells_10X.sparse.h5ad)**
Ce dataset contient un million de cellules et est stocké au format H5AD. Il comprend des profils d'expression génique et des métadonnées associées. Pour évaluer les performances du traitement sur CPU et GPU, des sous-ensembles de tailles variées, allant de 10 000 à 500 000 cellules, ont été extraits.


## Développement de l'Interface

L'interface de l'application a été développée pour faciliter l'analyse des données scRNA-seq en utilisant les outils suivants :

- **Scanpy (CPU) :** 1.10.2
- **rapids-singlecell (GPU) :** 0.10.6
- **cuML (GPU) :** 24.06.01
- **Dash (Interface) :** 2.17.1
- **Plotly (Visualisation) :** 5.22.0
- **Scikit-learn (Analyse) :** 1.5.1
- **Pandas (Analyse) :** 2.2.2
- **CuPy (GPU) :** 13.2.0
- **Anndata (Manipulation de Données) :** 0.10.6
- **scGPT (Annotation Automatique) :** 0.2.1

- ## Formats de Fichiers Supportés

L'application prend en charge les formats de fichiers suivants :

- **Fichiers MTX (10x Genomics)** 
- **Fichiers H5AD** 
- **Fichiers H5**
- 
### Lancer l'Application

Pour lancer l'application avec un fichier de données spécifique et une taille de sous-ensemble, utilisez la commande suivante :

```bash
python App.py --file_path /chemin/vers/votre_fichier/ --subset_size 200000
```

## Annotation Automatique

Pour l'annotation automatique des données de cellule unique, nous avons intégré **scGPT** (version 0.2.1), un modèle d'apprentissage profond adapté pour automatiser l'annotation des cellules.




