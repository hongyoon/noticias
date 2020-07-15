import genanki
from stop_words import get_stop_words

### crawler
BASE_URL = "https://www.bbc.com/mundo"
LOCALE_ES = "es_ES"
PATH_DB_MASTER =  "data\\db_master.tsv"
PATH_DB_URL = "data\\db_url.tsv"
PATH_DB_WORD = "data\\db_word.tsv"
PATH_DB_INTERESTING_WORD = "data\\db_interesting_word.tsv"
PATH_DECKS_FOLDER = "decks\\"

### preprocessor
STOP_SENTENCES = set([
    "Este artículo fue publicado The Conversation y reproducido aquí bajo la licencia Creative Commons.",
    "Ahora puedes recibir notificaciones de BBC Mundo. Descarga la nueva versión de nuestra app y actívalas para no perderte nuestro mejor contenido.",
    "Visita nuestra cobertura especial Ahora puedes recibir notificaciones de BBC Mundo.",
    "Haz clic aquí para leer la historia original."
])

STOP_WORDS = set(get_stop_words('es'))
STOP_POS = set(['PROPN','SYM','SPACE','PUNCT','NUM'])

COGNATE_PERCENTILE = 95

### preference
NEW_WORDS_ONLY = True

### Number of interesting words per article
NUM_INTERESTING_WORDS = 10

MODEL = genanki.Model(
  1000000000,
  'First Model',
  fields=[
    {'name': 'Word'},
    {'name': 'Meaning'},
    {'name': 'Sentence'}
  ],
  templates=[
    {
      'name': 'Standard',
      'qfmt': '{{Word}}',
      'afmt': '{{FrontSide}}<hr id="answer">{{Meaning}}<br>{{Sentence}}',
    },
  ])