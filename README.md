# HandCountingPY

HandCountingPY est une application Python qui utilise la webcam pour detecter les mains visibles a l'ecran.

Elle affiche :

- les points des doigts et articulations en rouge
- les connexions de la main en vert
- le nombre de mains detectees avec `nb_de_main`
- le nombre de FPS

Pour dessiner, leve seulement l'index et garde les autres doigts fermes.
La couleur du trait se regle avec les curseurs `R`, `G` et `B` dans la fenetre `Couleur`.
Le curseur `Clear` efface le dessin.

Pour afficher le cube 3D, ferme une main en poing.
Tant que le poing reste ferme, utilise l'autre main pour regler la taille du cube :

- pouce et index ecartes : agrandir
- pouce et index rapproches : retrecir

## Installation

```powershell
python -m pip install -r requirements.txt
```

## Lancement

```powershell
python hand_tracking.py
```

Changer de camera :

```powershell
python hand_tracking.py --camera 1
```

Changer la limite de detection :

```powershell
python hand_tracking.py --max-hands 50
```

## Commandes

- `q` : fermer la fenetre
- `Echap` : fermer la fenetre
- `c` : effacer le dessin
