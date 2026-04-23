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
Le curseur `Style` change le design de l'effet de charge.
Le curseur `Mode` change le mode de jeu.

Modes disponibles :

- `1` : Dessin, avec couleur et suppression du canvas
- `2` : Cube, avec poing ferme pour afficher le cube et autre main pour regler sa taille
- `3` : Pouvoir, avec charge du poing, explosion au relachement et lune quand l'index seul est leve
- `4` : Nerd, avec lunettes, dents et ampoule Eureka quand l'index seul est leve

Pour lancer l'animation de charge, ferme une main en poing.
Plus le poing reste ferme, plus l'aura, les particules, les anneaux et les eclairs grandissent.
Quand tu relaches le poing, l'energie explose avec une onde de choc et des lignes d'impact.
En mode pouvoir, leve seulement l'index pour afficher une lune de la couleur du style choisi.
En mode nerd, leve seulement l'index pour afficher les lunettes, les dents et l'ampoule au-dessus du doigt.

Styles disponibles :

- Inferno : flammes et explosion chaude
- Plasma : arcs electriques et noyau losange
- Frost : pics et cristaux de glace
- Toxic : nuage de poison et bulles

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
- `1` : mode dessin
- `2` : mode cube
- `3` : mode pouvoir
- `4` : mode nerd
