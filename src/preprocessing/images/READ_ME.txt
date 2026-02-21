- File EmbeddingsExtraction: file python in cui puoi trovare l'estrazione degli embeddings da modelli pre-trainati. Output: embeddings troppo grandi da 768 --> si passa l'output di questa funzione all'autoencoder per ridurne la dimensione a 8.

- img_autoencoder.py: autoencoder per le immagini che fa passare l'embeddings da una dimensione di 768 a 8.

- AE_function.py: file python con tutte le funzioni che servono all'autoencoder. Qui trovi due tipi di train: usa quello con l'early stopping. C'è anche la funzione di normalizzazione e di estrazione dell'embedding usando il modello (extract_embeddings).
