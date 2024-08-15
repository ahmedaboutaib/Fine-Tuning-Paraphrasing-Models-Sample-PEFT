# Fine-Tuning-Paraphrasing-Models-Sample-PEFT


# Projet de Paraphrasage avec T5 et PEFT

Ce projet utilise des modèles T5 pour la génération de paraphrases en utilisant les techniques de fine-tuning standard et l'optimisation des hyperparamètres avec PEFT (Parameter Efficient Fine-Tuning). Le projet est divisé en deux parties principales :

1. **Entraînement d'un modèle T5 de paraphrasage avec des données d'exemple.**
2. **Optimisation du modèle avec PEFT pour une meilleure performance.**

## Table des Matières

- [Introduction](#introduction)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Structure du Projet](#structure-du-projet)
- [Utilisation](#utilisation)
- [Exemples](#exemples)
- [Contribution](#contribution)
- [Licence](#licence)

## Introduction

Ce projet consiste à fine-tuner un modèle T5 pour la génération de paraphrases en utilisant deux approches :

1. **Code 1 :** Entraînement de base du modèle T5 sur un ensemble de données de paraphrases.
2. **Code 2 :** Optimisation du modèle avec PEFT pour améliorer l'efficacité de l'entraînement et la performance du modèle.

## Prérequis

Avant de commencer, assurez-vous d'avoir les éléments suivants installés sur votre machine :

- Python 3.7 ou supérieur
- [Google Colab](https://colab.research.google.com/) (pour exécuter les notebooks)
- Bibliothèques Python :
  - `torch`
  - `transformers`
  - `datasets`
  - `pandas`
  - `sklearn`
  - `peft`

Vous pouvez installer les bibliothèques nécessaires avec :

```bash
pip install torch transformers datasets pandas scikit-learn peft
```

## Installation

1. Clonez ce dépôt :

   ```bash
   git clone https://github.com/votre-utilisateur/votre-repository.git
   ```

2. Accédez au répertoire du projet :

   ```bash
   cd votre-repository
   ```

3. Assurez-vous d'avoir tous les fichiers nécessaires (notebooks, scripts, etc.) et de les organiser comme suit :

   - `code1.ipynb` : Notebook pour l'entraînement de base du modèle T5.
   - `code2.ipynb` : Notebook pour l'optimisation du modèle avec PEFT.
   - `install.ipynb` : Notebook contenant toutes les exigences d'installation.

## Structure du Projet

### Code 1 : Entraînement de Base

**Fichier : `code1.ipynb`**

1. **Importation des Bibliothèques et Configuration Initiale** : Chargement des bibliothèques nécessaires et configuration des paramètres.
2. **Définition des Constantes et Chargement des Données** : Définition des hyperparamètres et chargement des données depuis Google Drive.
3. **Chargement des Datasets et Prétraitement** : Chargement et prétraitement des données pour l'entraînement.
4. **Initialisation du Modèle et Entraînement** : Initialisation du modèle T5 et configuration des arguments d'entraînement.
5. **Sauvegarde du Modèle et du Tokenizer** : Sauvegarde des résultats d'entraînement sur Google Drive.
6. **Test du Modèle** : Chargement du modèle sauvegardé et génération de paraphrases.

### Code 2 : Optimisation avec PEFT

**Fichier : `code2.ipynb`**

1. **Importation des Bibliothèques et Configuration Initiale** : Chargement des bibliothèques et configuration initiale.
2. **Définition des Constantes et Chargement des Données** : Définition des hyperparamètres et chargement des données.
3. **Chargement des Datasets et Prétraitement** : Prétraitement des données pour l'entraînement avec PEFT.
4. **Configuration de PEFT et Entraînement** : Application de PEFT au modèle et configuration des arguments d'entraînement.
5. **Sauvegarde du Modèle et du Tokenizer** : Sauvegarde des résultats sur Google Drive.
6. **Test du Modèle** : Chargement du modèle optimisé et génération de paraphrases.

## Utilisation

1. **Exécutez `code1.ipynb` pour entraîner le modèle T5 de base :**
   - Ouvrez le notebook dans Google Colab ou Jupyter Notebook.
   - Suivez les instructions pour exécuter chaque cellule et entraîner le modèle.

2. **Exécutez `code2.ipynb` pour optimiser le modèle avec PEFT :**
   - Ouvrez le notebook dans Google Colab ou Jupyter Notebook.
   - Suivez les instructions pour exécuter chaque cellule et fine-tuner le modèle avec PEFT.

## Exemples

Voici un exemple de génération de paraphrases avec le modèle optimisé :

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Charger le modèle et le tokenizer sauvegardés
model_path = '/content/drive/My Drive/PEFT_1/PEFT_v1/checkpoint-10000'
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained('/content/drive/My Drive/PEFT_1/PEFT_v1/')

def do_correction(text, model, tokenizer):
    input_text = f" paraphrase : {text}"
    inputs = tokenizer.encode(
        input_text,
        return_tensors='pt',
        max_length=256,
        padding='max_length',
        truncation=True
    )
    corrected_ids = model.generate(
        inputs,
        max_length=256,
        num_beams=5,
        early_stopping=True
    )
    corrected_sentence = tokenizer.decode(
        corrected_ids[0],
        skip_special_tokens=True
    )
    return corrected_sentence

text = "Thunderstorms and strong gusts of wind with dust blowers on Sunday"
print(do_correction(text, model, tokenizer))
```

## Contribution

Les contributions sont les bienvenues ! Veuillez ouvrir une issue ou soumettre une pull request si vous avez des améliorations ou des corrections à proposer.

## Licence

Ce projet est sous la [Licence MIT](LICENSE).
