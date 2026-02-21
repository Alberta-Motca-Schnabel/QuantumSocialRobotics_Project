import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import sys
from General.Utility import decoder_labels

# === Dataset Triplet per fusione ===
class TripletFusionDataset(Dataset):
    def __init__(self, text_emb, image_emb, labels, to_be_saved = True):
        self.text_emb = torch.tensor(text_emb, dtype=torch.float32)
        self.image_emb = torch.tensor(image_emb, dtype=torch.float32)
        self.labels_ds = [decoder_labels(l) for l in labels]
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.combined = torch.cat((self.text_emb, self.image_emb), dim=1)  # [n, 32]

        if to_be_saved:
            dataset_df = pd.DataFrame(self.combined.numpy())
            dataset_df['labels'] = self.labels_ds
            dataset_df.to_csv(sys.path[1] + "\\IncrementalModels\\Dataset_incr\\fusion_incr_16_100.csv", index=False)

        self.label_to_indices = {int(label): (self.labels == label).nonzero(as_tuple=True)[0]
                                 for label in torch.unique(self.labels)}

    def __len__(self):
        return len(self.combined)

    def __getitem__(self, index):
        anchor = self.combined[index]
        label = self.labels[index].item()

        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[label].numpy())
        positive = self.combined[positive_index]

        negative_label = label
        while negative_label == label:
            negative_label = np.random.choice(list(self.label_to_indices.keys()))
        negative_index = np.random.choice(self.label_to_indices[negative_label].numpy())
        negative = self.combined[negative_index]

        return anchor, positive, negative


# === Rete neurale: da 32D -> 8D ===
class FusionEmbedding(nn.Module):
    def __init__(self):
        super(FusionEmbedding, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(16, 8)
        )

    def forward(self, x):
        return self.model(x)
