from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import sys

torch.manual_seed(1234)

class EmbeddingsExtraction():
    def __init__(self):
        self.tokenizerText = AutoTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
        self.tokenizerText.add_special_tokens(False)
        self.modelText = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion",output_hidden_states=True)

        self.processorImg = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
        self.modelImg = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection",output_hidden_states = True)

    def extract_Text_Emb(self, text, device='cpu'):
        inputs = self.tokenizerText(text, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = self.modelText(**inputs)

        hidden_states = outputs.hidden_states
        last_layer_embeddings = hidden_states[-1]

        # Estrae il token [CLS] (che sta in posizione 0 lungo seq_len)
        cls_embedding = last_layer_embeddings[:, 0, :]  # shape: (batch_size, hidden_dim)

        return cls_embedding.squeeze(0)

    def extract_Img_Emb(self, img_path, device='cpu'):
        image = Image.open(sys.path[1] + "\\Frontend\\" + img_path)
        inputs = self.processorImg(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = self.modelImg(**inputs)

        hidden_states = outputs.hidden_states
        last_layer_embeddings = hidden_states[-1]  # shape: (1, num_patches, hidden_dim)

        #Prende il primo token (CLS)
        cls_embedding = last_layer_embeddings[:, 0, :]  # shape: (1, hidden_dim)

        return cls_embedding.squeeze(0)

#if __name__ == '__main__':
#    dc = EmbeddingsExtraction()
#    a = dc.extract_Img_Emb("temp_uploads\\foto.jpg")
#    print(a.shape)

#    b = dc.extract_Text_Emb("AOOO come stai?")
#    print(b.shape)










