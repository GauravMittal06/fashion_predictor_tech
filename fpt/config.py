"""Shared settings for all four layers of the prototype. Edit lists here, not inside the scripts.

Adapted from the team's code (reference/fashion_prediction_technology/config.py) for a local,
one-country prototype: Spain, using the public GeoStyle data for Barcelona.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'outputs'
IMAGE_DIR = ROOT / 'dataset_images_1'   # batch 1: the team's labelled screenshots (Southern Europe)
LABEL_FILE = ROOT / 'labels.csv'
if (ROOT / 'labels_reviewed.csv').exists():          # the team's checked labels win (review/label_review.html)
    LABEL_FILE = ROOT / 'labels_reviewed.csv'
TEAM_DIR = IMAGE_DIR
BATCH2_DIR = ROOT / 'dataset_images_2' / 'photos'   # batch 2: 100 influencer photos (10 Instagram pages)
BATCH2_LABELS = ROOT / 'labels_batch2.csv'           # made by scripts/build_labels_batch2.py
SEED = 42

# ---------- Layer 1: tagger ----------
STYLES = ['Evening/Party', 'Casual', 'Street', 'Resort/Swim',
          'Tailored/Suiting', 'Winter/Outerwear']
FABRICS = ['Denim', 'Leather', 'Knit', 'Satin/Silk', 'Cotton/Linen',
           'Wool/Tweed', 'Sequin/Metallic', 'Lace/Mesh/Sheer',
           'Fur/Shearling', 'Jersey/Stretch']
COLOURS = ['black', 'white/cream', 'beige', 'red', 'blue', 'pink',
           'green', 'brown', 'grey', 'metallic']
CUES = ['button', 'zipper', 'motif']

TAGGER_MODEL = 'hf-hub:Marqo/marqo-fashionSigLIP'      # Plan A, frozen
TAGGER_MODEL_B = 'patrickjohncyh/fashion-clip'         # Plan B, frozen (FashionCLIP 2.0)
STYLE_TEXT = {'Evening/Party': 'an evening party outfit', 'Casual': 'a casual everyday outfit',
              'Street': 'a streetwear outfit', 'Resort/Swim': 'a resort beachwear outfit',
              'Tailored/Suiting': 'a tailored suit outfit', 'Winter/Outerwear': 'a winter coat outfit'}
FABRIC_TEXT = {'Denim': 'denim', 'Leather': 'leather', 'Knit': 'knitwear', 'Satin/Silk': 'satin or silk',
               'Cotton/Linen': 'cotton or linen', 'Wool/Tweed': 'wool or tweed',
               'Sequin/Metallic': 'sequins', 'Lace/Mesh/Sheer': 'lace or sheer fabric',
               'Fur/Shearling': 'fur or shearling', 'Jersey/Stretch': 'stretch jersey'}
COLOUR_TEXT = {c: c.replace('/', ' or ') for c in COLOURS}
CUE_TEXT = {'button': ('visible buttons', 'no buttons'),
            'zipper': ('a visible zipper', 'no zipper'),
            'motif': ('a print, pattern or slogan', 'plain fabric with no print')}

# ---------- Layer 2: trends and forecasting ----------
COUNTRY, COUNTRY_NAME, CITY = 'ES', 'Spain', 'Barcelona'
MIN_PHOTOS_PER_MONTH = 30     # months with fewer photos are treated as missing
MIN_MEAN_SHARE = 0.02         # attributes worn by fewer than 2% of people on average are skipped
SEASON = 12                   # months per year
HORIZON = 6                   # forecast 6 months ahead
MIN_HISTORY = 24              # a forecast needs two full years behind it
BACKTEST_CUTOFFS = 12         # the last 12 forecast start points with a full 6-month truth
CALL_THRESHOLD = 0.10         # 10% above/below last year = rising/fading
QUANTILES = [0.1, 0.5, 0.9]   # low, middle, high forecast
CHRONOS2_MODEL = 'amazon/chronos-2'          # Plan A (120M parameters)
LSTM_IN, LSTM_OUT, LSTM_UNITS = 12, 6, 64    # Plan C: 12 months in, 6 months out
LSTM_EPOCHS, LSTM_PATIENCE = 80, 5

# Barcelona attributes (GeoStyle's 12 CNN attributes) -> readable trend names
GEO_ATTR_NAMES = {
    'clothing_pattern': 'Pattern', 'major_color': 'Colour', 'wearing_necktie': 'Necktie',
    'collar_presence': 'Collar', 'wearing_scarf': 'Scarf', 'sleeve_length': 'Sleeves',
    'neckline_shape': 'Neckline', 'clothing_category': 'Garment', 'wearing_jacket': 'Jacket',
    'wearing_hat': 'Hat', 'wearing_glasses': 'Glasses', 'multiple_layers': 'Layers',
}
# Text the tagger uses to check a generated image for each trend attribute
GEO_TEXT = {
    'Pattern': lambda v: {'Solid': 'plain solid-colour clothes', 'Striped': 'striped clothes',
                          'Graphics': 'a graphic print t-shirt', 'Floral': 'floral print clothes',
                          'Plaid': 'plaid tartan clothes', 'Spotted': 'polka dot clothes'}[v],
    'Colour': lambda v: 'multicoloured clothes' if v == 'More than 1 color' else f'{v.lower()} clothes',
    'Garment': lambda v: {'Dress': 'a dress', 'Outerwear': 'a coat or jacket', 'T-shirt': 'a t-shirt',
                          'Suit': 'a suit', 'Shirt': 'a button-up shirt', 'Sweater': 'a sweater',
                          'Tank top': 'a tank top'}[v],
    'Sleeves': lambda v: {'Short sleeve': 'short sleeves', 'Long sleeve': 'long sleeves',
                          'No sleeve': 'sleeveless clothes'}[v],
    'Neckline': lambda v: {'Round': 'a round neckline', 'Folded': 'a folded collar neckline',
                           'V-shape': 'a v-neck'}[v],
    'Layers': lambda v: 'several layered garments' if v == 'Multiple layers' else 'a single layer of clothing',
    'Jacket': lambda v: 'a jacket', 'Hat': lambda v: 'a hat', 'Glasses': lambda v: 'glasses',
    'Scarf': lambda v: 'a scarf', 'Necktie': lambda v: 'a necktie', 'Collar': lambda v: 'a collar',
}

# Barcelona photos blocked by hand after review (anything the automatic screen misses)
EXCLUDE_PHOTOS = {'2018-03_b97c4130.jpg'}   # painted poster of a saint

# ---------- Layer 3: generation ----------
GEN_MODEL = 'stabilityai/sdxl-turbo'          # fits a 6 GB laptop GPU; non-commercial licence
GEN_STEPS, GEN_SIZE = 4, 512
CANDIDATES_PER_BRIEF = 8
MATCH_THRESHOLD = 0.5         # tagger probability needed for each requested attribute
NEAR_COPY_SIMILARITY = 0.92   # starting value; calibrate on known near-duplicate photo pairs
