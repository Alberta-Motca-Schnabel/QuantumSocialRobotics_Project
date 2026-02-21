import re
from nltk.corpus import stopwords
import pandas as pd
import seaborn as sns
import plotly.express as px

def clean_text(text):
    # Converte tutto in minuscolo
    text = text.lower()

    # Rimuove le espressioni tipo [NAME] o [qualcosa dentro]
    text = re.sub(r'\[.*?\]', '', text)

    # Rimuove le emoticon testuali come :) :( :D ecc.
    text = re.sub(r'[:;=][\-\^]?[)\(\/\\\|O0pP]', '', text)

    # Rimuove la punteggiatura
    text = re.sub(r'[^\w\s]', '', text)

    # Rimuove gli spazi bianchi in eccesso
    text = re.sub(r'\s+', ' ', text).strip()

    # Rimuove le stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text

def clean_dataset(dataset):
    texts = []
    for text in dataset['text']:
        texts.append(clean_text(text))
    return texts

def encoder_labels(labels):
    ekman_emotion = { "angry": 0,
                      "disgust": 1,
                      "fear": 2,
                      "happy": 3,
                      "sad": 4,
                      "surprise": 5,
                      "neutral": 6
                    }

    labels_encoder = []
    for l in labels:
        l = ekman_emotion[l]
        labels_encoder.append(l)
    return labels_encoder

def decoder_labels(label):
    emotion = { 0: "angry",
                1: "disgust",
                2: "fear",
                3: "happy",
                4: "sad",
                5: "surprise",
                6: "neutral"
                }
    return emotion[label]

def plot_3d_interactive(embeddings_2d, labels, title="3D Embeddings Visualization"):
    # Crea un DataFrame con le prime 3 componenti
    df = pd.DataFrame({
        'PC1': embeddings_2d[:, 0],
        'PC2': embeddings_2d[:, 1],
        'PC3': embeddings_2d[:, 2],
        'Class': labels
    })

    # Crea il plot 3D interattivo
    fig = px.scatter_3d(df,
                        x='PC1',
                        y='PC2',
                        z='PC3',
                        color='Class',
                        title=title,
                        labels={'PC1': 'Principal Component 1',
                                'PC2': 'Principal Component 2',
                                'PC3': 'Principal Component 3'},
                        hover_name='Class',
                        width=1000,
                        height=800)

    # Migliora la leggibilità
    fig.update_layout(
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show()