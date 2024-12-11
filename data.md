# Fill-Mask

## MODELLE
https://huggingface.co/google-bert/bert-base-cased # USE
https://huggingface.co/google-bert/bert-base-uncased


## DATASETS
https://huggingface.co/datasets/mbien/recipe_nlg # USE


## INFO
50_000 Rezepte von 2_231_142 (2,2% des Datensatzes) dauert knapp 35 Minuten
(211_135 useable sentences)

### useable_sentences
1) FINISHED EXTRACTING SENTENCES... (sentences of all lengths)
>>> Size of all_sentences: '18880765' Lines.

1) FINISHED EXTRACTING SENTENCES... (sentences of length >= 5)
>>> Size of all_sentences: '15678968' Lines.

### unuseable_sentences
>>> unuseable_sentences: '71'
>>> unuseable_sentences: '42'


## WAHRSCHEINLICHKEITEN
TOP 5 lösungen => schauen ob Mask-token unter TOP 5
    => Ja: Wahscheinlichkeit notieren
    => Nein: Wahrscheinlichkeit = 0

Wahrscheinlichkeiten aufsummieren und mittel bilden

Variante 2:
oberste Lösung des Modells getroffen?

### zero-shot
#### Versuch 1 (10 rows)
score_of_model:  0.06253818273544312
runtime: 2.09 sec

#### Versuch 2 (224 rows)
score_of_model: 0.11983776877501182
runtime: 4.17 sec

#### Versuch 3 (1222 rows)
score_of_model:  0.1322530933675614
runtime: 13.45 sec


### top-k with 5
#### Versuch 1 (10 rows)
score_of_model:  0.1037619911134243
runtime: 1.83 sec

#### Versuch 2 (224 rows)
score_of_model:  0.13975950878479385
runtime: 4.17 sec

#### Versuch 3 (1222 rows)
score_of_model:  0.15039981787745366
runtime: 14.15 sec
