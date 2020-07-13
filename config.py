from stop_words import get_stop_words

### crawler
BASE_URL = "https://www.bbc.com/mundo"
LOCALE_ES = "es_ES"
PATH_DB_MASTER =  "data\\db_master.tsv"
PATH_DB_URL = "data\\db_url.tsv"
PATH_DB_WORD = "data\\db_word.tsv"

### preprocessor
STOP_SENTENCES = set([
    "Este artículo fue publicado The Conversation y reproducido aquí bajo la licencia Creative Commons.",
    "Ahora puedes recibir notificaciones de BBC Mundo. Descarga la nueva versión de nuestra app y actívalas para no perderte nuestro mejor contenido.",
    "Visita nuestra cobertura especial Ahora puedes recibir notificaciones de BBC Mundo.",
    "Haz clic aquí para leer la historia original."
])

STOP_WORDS = set(get_stop_words('es'))
STOP_POS = set(['PROPN','SYM','SPACE','PUNCT','NUM'])

COGNATE_PERCENTILE = 0.95

### preference
NEW_WORD_ONLY = True