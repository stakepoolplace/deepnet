# Corpus v2 — accords et contraintes sémantiques (non commité)

Code : `src/test/java/RN/transformer/CorpusV2.java` (le corpus v1, `NextWordProtocol`, est conservé intact).
Tests : `CorpusV2Test` (14), `CorpusV2PipelineSmokeTest` (1).

## Structure
`SUJET VERBE [COMPLÉMENT]` avec `SUJET = DET NOM [ADJ]` (adjectif postposé, p = 0,5) et un complément selon le cadre du verbe :
- **T** (transitif) : objet `DET NOM [ADJ]` obligatoire — mange, lit, conduit, voit, ouvre, traverse
- **I** (intransitif) : rien (p = 0,5) ou groupe prépositionnel `PREP DET NOM [ADJ]` — dort, court, roule
- **L** (locatif) : groupe prépositionnel obligatoire — habite, vit

Exemples tirés : `des souris voient un enfant petit` · `le garcon gentil court sous la table vieille` ·
`la voiture petite traverse des villes` · `des femmes dorment`.

## Lexique
30 noms (genre, nombre irrégulier : oiseau/oiseaux, cheval/chevaux, souris/souris…), 7 classes sémantiques
(HUMAN, ANIMAL, FOOD, READABLE, VEHICLE, PLACE, OBJECT), 10 adjectifs à 4 formes (petit/petite/petits/petites…),
11 verbes (formes singulier/pluriel), 5 prépositions, 6 déterminants. **Vocabulaire : 127 jetons** (dont 4 spéciaux).

## Trois niveaux de validité distincts (cumulatifs)
| Niveau | Vérifie | Exemple rejeté |
|---|---|---|
| N1 structure | patron `D N A? V (D N A? \| P D N A?)?` | `le mange chat` |
| N2 accord | déterminant–nom, adjectif–nom (genre + nombre), verbe–sujet (nombre), sous-catégorisation | `les chats mange…`, `la chat…`, `le chat mange` (objet manquant) |
| N3 sémantique | classe du sujet / de l'objet / du lieu, adjectif compatible, préposition compatible | `la pomme mange le chat`, `le chat mange la table`, `le chat frais…`, `dort dans la pomme` |

Violations typées (`CorpusV2.Violation`) : structure, incomplète, jeton inconnu, accord (déterminant, adjectif, sujet-verbe),
sous-catégorisation, sémantique (sujet, objet, adjectif, lieu).
**Les phrases générées sont exactement les phrases de niveau N3** (support de la loi) : `pGen(s) > 0 ⇔ N3`, vérifié par
40 000 mutations aléatoires, avec recalcul indépendant de la probabilité.

## Oracle exact (comme pour v1)
Probabilité conditionnelle exacte de la suite sachant l'amorce (postérieur sur la longueur ∝ 1/(n−1) car k ~ U{1..n−1}),
calculée par factorisation (sujet et complément indépendants sachant le verbe). Vérifié :
énumération brute (40 préfixes, égalité à 1e-12) et identité de normalisation Σ_préfixes Z = E[1{n>k}/(n−1)] (k = 1, 2).
- **Plancher de perte : 1,50 ± 0,004 nat/token** (v1 : 1,96) ; ln(127) = 4,84. Les contraintes rendent la tâche plus prévisible.
- Un décodage glouton optimal donne 100 % à N1, N2 et N3 par construction ; l'exactitude stricte reste plafonnée (< 10 %).

## Jeux (mêmes tailles que v1, tous neufs)
entraînement fixe 200 phrases (600 paires), validation 150, test 200, TEST-FINAL 200, disjoints ; échantillonnage dynamique
(200 phrases nouvelles/epoch, graine reproductible, hors jeux, sans doublon dans l'epoch). Taille effective ≈ 1,9·10⁵ phrases
(v1 : ≈ 7·10⁶) : après 100 000 tirages, 45 % de la masse est couverte — un entraînement de 40–100 epochs ne mémorise pas l'espace.

## Limites
- Contraintes **codées à la main** et volontairement simples (pas de gender inanimé « naturel » vérifié par un linguiste,
  adjectifs tous postposés, pas de négation ni de subordonnées).
- Le test mesure la maîtrise de CETTE grammaire, pas du français.
- Le runner d'expériences (`NextWordExperiment2Test`) n'est pas encore branché sur v2 (l'évaluation à 3 niveaux existe :
  `CorpusV2.evaluate`).

## Suite proposée
Brancher v2 sur le runner et rejouer D1/D2 (même recette, même budget, 40 epochs) avec N1/N2/N3 mesurés séparément,
sélection sur validation seule, TEST-FINAL lu une fois.
