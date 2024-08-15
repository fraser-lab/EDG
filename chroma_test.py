import torch
from chroma import Chroma, Protein, api, conditioners
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('/home/kchrispens/.secrets/chroma_api_key', 'r') as f:
    api_key = f.read().strip()

api.register_key(api_key)

chroma = Chroma()
protein = chroma.sample()

print(protein)
protein.to('test.cif')