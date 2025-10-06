from ray import serve
from sentence_transformers import SentenceTransformer
import torch

@serve.deployment(ray_actor_options={"num_gpus": 1})
class EmbeddingModel:
    def __init__(self):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    async def __call__(self, request):
        data = await request.json()
        text = data.get("text", "")
        embeddings = self.model.encode([text], convert_to_tensor=True)
        return {"embeddings": embeddings[0].tolist()}

app = EmbeddingModel.bind()
