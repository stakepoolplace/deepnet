# Prédiction du mot suivant — rapport (non commité)

## Résultat
Gagnant **D2** (corrections numériques + données dynamiques), sélectionné sur la VALIDATION seule
(`target/nextword-experiment2/selection.txt`, écrit avant toute lecture du TEST-FINAL).
Validité grammaticale : **150/150 validation**, **200/200 TEST-FINAL** (IC95 Wilson 98,1–100 %).
Écart à la perte optimale (oracle exact de la grammaire) : +0,0095 nat/token sur TEST-FINAL.

## Comparaison (TEST-FINAL, 200 amorces inédites, jamais lu avant le verdict)
| Modèle | Validité (IC95) | Perte | Excès vs oracle | vs A (signes appariés) |
|---|---|---|---|---|
| A  best.ser historique (313 ep., gradients faux) | 68,0 % (61,2–74,1) | 2,874 | +0,906 | – |
| B  reprise de A, LR cosinus | 62,5 % (55,6–68,9) | 2,546 | +0,578 | p=0,18 |
| D0-best  gradients corrigés seuls, ep. 5 (min. perte val.) | 69,0 % (62,3–75,0) | 2,493 | +0,525 | p=0,91 |
| D0-last  idem, ep. 40 | 80,0 % (73,9–85,0) | 3,582 | +1,614 | p=0,009 |
| D1-best  + numérique corrigé, ep. 5 | 76,5 % (70,2–81,8) | 2,411 | +0,443 | p=0,06 |
| D1-last  idem, ep. 40 | 89,0 % (83,9–92,6) | 3,862 | +1,894 | p<0,0001 |
| **D2  + données dynamiques, ep. 40** | **100 % (98,1–100)** | **1,978** | **+0,010** | p<0,0001 (64 meilleurs / 0 pire) |

Budget identique pour D0/D1/D2 : 40 epochs x 600 paires = 1520 pas, tokens supervisés 104 120 (fixe) vs 101 600 (dynamique).

## Causes identifiées (Phase 1, différences finies)
1. `MultiHeadAttention.backward` : gradients de Wq/Wk/Wv aplatis SANS permuter têtes/séquence
   (erreur ≈ taille du gradient ; Wo exact). Test : `MultiHeadAttentionWeightGradientCheckTest`.
2. `LinearProjection.backward` : gradient des poids calculé sur l'entrée brute au lieu de `scaled`,
   gradient d'entrée avec somme au lieu de moyenne et terme -mean(g) omis.
   Test : `TransformerEndToEndGradientCheckTest` (117 éléments, à l'init ET après perturbation de tous les poids).
3. Échelle : gradient de la SOMME alors que la perte rapportée est une MOYENNE (facteur N = tokens du batch ;
   vérifié exact). Option `MEAN_PER_TOKEN` ajoutée (défaut inchangé).
4. Pics de perte : 0 hausse >10 % d'une epoch à l'autre sur 5 exécutions corrigées (132 epochs) ; pics de +35 à +180 %
   avec les gradients faux (B, C, epoch 100 historique). Association forte, mais PAS d'ablation contrôlée
   (bugs réintroduits) : cause « probable », non prouvée causalement.

## Limites (à lire avant de conclure)
- Corpus synthétique à 46 mots, grammaire à 8 structures : le test mesure la maîtrise de cette grammaire, pas du langage.
- « Validité sémantique » non mesurable (aucune contrainte de sens, accords inclus). Seule la validité grammaticale est évaluée.
- La correspondance exacte (0,5–2 %) est au niveau de son plafond théorique (1,2 %) : la suite est aléatoire.
- Les gains D1 vs D0 (89 % vs 80 %) ne sont pas testés par paire ; les IC se chevauchent peu.
- Le clipping global (1,0) est actif presque à chaque pas (norme ~1,5–3,5) : son utilité n'a pas été ablatée.
- Run D2 non répété (graines fixées ; non-déterminisme GPU possible mais non mesuré).
- L'ancien jeu TEST a été lu une fois (comparaison A/B/C) : intact mais plus vierge ; d'où TEST-FINAL.
- Le LR de pic (1e-3) a été choisi sur la validation (pilote 12 epochs), jamais sur un TEST.

## Reproduire
```
mvn test -Dtest=NextWordExperiment2Test -Dnw2.mode=train -Dnw2.tag=D2 -Dnw2.epochs=40 \
  -Dnw2.data=dynamic -Dnw2.norm=MEAN -Dnw2.clip=GLOBAL -Dnw2.clipNorm=1.0 -Dnw2.sched=COSINE -Dnw2.lr=1e-3 -Dnw2.warmup=100
mvn test -Dtest=NextWordExperiment2Test -Dnw2.mode=final -Dnw2.arms="A=...,D2-best=target/nextword-experiment2/D2/best.ser"
```
