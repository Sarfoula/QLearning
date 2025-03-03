The deep Q networks:

- Bellman equation:
  - Q(s,a) = r + **γ**max**Q**(**s**′**,**a**′**)

Sert a calculer la valeur d'une action **a** pour un etat **s**

- r : reward obtenu
- **γ** : facteur de discount()
- max**Q**(**s**′**,**a**′**) : meilleur valeur pour la prochaine action au prochaine etat

Le reseau de neurones que l'on entraine vas essayer de predire les Q(s, a) pour chaque action

Donc en entree le reseaux prend un etat et retourn 3 valeurs.


Mais d'abord il dois s'entrainer pour coller au mieux a la fonction Q.

Pour ce faire on calcule la perte grace a l'ecart entre la Q_value obtenu et la "cible"

- Q_value est le resultat donner par notre reseaux de neurones
- "cible" etant l'equation de Bellman **y**=**r**+**γ**max**Q**′**(**s**′**, **a**′) (Attention: pas obligatoire mais on utilise un autre reseaux pour predire la "cible" P.S: c'est pour ca le signe prime pour la fonction Q)


Fonction de perte:

On utilise cette fonction avec la Q_value et la "cible", il y a plusieurs choix possible et disponible dans le module pytorch.


Exploration vs exploitation:

Pour ajuster le reseaux il faut faire de l'exploration et ainsi decouvrir l'environnement donc on peut utiliser une politique **ϵ**-greedy (Rapidement: on a **0 ≤ ϵ ≤ 1** et ϵ decroit ainsi si un nombre random est plus grand que ϵ alors on fais une action aleatoire sinon on suit la Qvalue)





AMELIORATION:


Utilisation d'un reseau cible:

- On utilise un autre reseaux pour calculer la prochaine max**Q′(s′, a′)**

Experience Replay

- On stocke les transitions (state, action, reward, next_state, done)
- Puis on creer un echantillon de plusieurs transition selectionne au hasard
- On fais notre equation sur cette echantillon pour le reseaux

Prioritized Experience Replay

- On fais tout pareil sauf que on donne une priorite au transition (comme si on les notait/20)
- Erreur TD: **δi = ∣yi − Q(si, ai)∣** (Calcul de la perte) ---> erreurTD = |Q_target - Q_value|
- Priorites: **pi = ∣δi∣^α + ϵ**
  - α : amplification des priorite (en general 0.4 ≤ α ≤ 1)
  - ϵ : constante positive pour eviter une priorite nul
- Echantillonage: **P(i) = pi / ( ∑j * pj )**
  - Cette equation definit les proba d'etre choisis
- Correction du biais (On ne choisit plus aleatoirement donc il y a un biais alors il faut le corriger)
  - Poids d'importance: **wi = (1/N . 1/P(i))^β**
  - N : buffer size
  - β : correction du biais (crois vers 1 pendant l'entrainement)
- Ainsi on P(i) qui permet de choisir les transitions et Wi qui est sont poids associer pour palier au probleme de biais
-
