# Intel Image Classification — Julianna

Projet de classification d'images CNN (PyTorch + TensorFlow) avec interface web Flask.

## Classes
buildings · forest · glacier · mountain · sea · street

## Structure
```
projet_intel/
├── app.py                    # Serveur Flask
├── requirements.txt
├── README.md
├── models/
│   ├── cnn1.py               # Architecture CNN PyTorch
│   ├── Julianna_model.pth    # Modèle PyTorch entraîné
│   └── Julianna_model.keras  # Modèle TensorFlow entraîné
├── ml/
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── main.py
├── static/css/style.css
└── templates/index.html
```

## Installation

```bash
pip install -r requirements.txt
```

## Entraînement des modèles

```bash
# PyTorch
python ml/main.py --model pytorch --firstname Julianna --epochs 20

# TensorFlow
python ml/main.py --model tensorflow --firstname Julianna --epochs 20
```

## Lancer l'application web

```bash
python app.py
```

Ouvrir : http://localhost:5000

## Déploiement (PythonAnywhere)

1. Uploader le dossier `projet_intel/` sur PythonAnywhere
2. Créer un virtualenv et installer `requirements.txt`
3. Configurer le fichier WSGI pour pointer vers `app.py`
4. Placer les modèles `.pth` et `.keras` dans `models/`

## Dépendances principales

| Package | Version |
|---------|---------|
| Flask | >= 2.3 |
| PyTorch | >= 2.0 |
| TensorFlow | >= 2.13 |
| Pillow | >= 9.5 |
| NumPy | >= 1.24 |
