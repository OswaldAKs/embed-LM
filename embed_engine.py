# embed_engine.py
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from ray import serve

app = FastAPI()

class EmbedRequest(BaseModel):
    texts: List[str]
    instruction: Optional[str] = ""     # NV-Embed expects an instruction prefix for queries
    batch_size: int = 8
    normalize: bool = True
    max_length: int = 32768

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

@serve.deployment(
    name="NVEmbedService",
    num_replicas=1,
    max_concurrent_queries=64,
    ray_actor_options={"num_gpus": 1.0}
)
@serve.ingress(app)
class EmbeddingService:
    def __init__(self, model: str = "nvidia/NV-Embed-v2", trust_remote_code: bool = True, device: Optional[str] = None):
        self.model_name = model
        self.trust_remote_code = trust_remote_code
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[NVEmbed] loading {self.model_name} on {self.device}")
        # This loads the model code provided by the repository (so trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
        try:
            self.model.to(self.device)
        except Exception:
            # some HF wrappers manage device internally
            pass
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)

    @app.post("/embed", response_model=EmbedResponse)
    async def embed(self, req: EmbedRequest, raw_request: Request) -> EmbedResponse:
        texts = req.texts or []
        instruction = req.instruction or ""
        batch_size = max(1, int(req.batch_size))
        max_length = int(req.max_length)
        try:
            if len(texts) == 0:
                return EmbedResponse(embeddings=[])
            # small requests: direct encode
            if len(texts) <= batch_size:
                with torch.no_grad():
                    emb = self.model.encode(texts, instruction=instruction, max_length=max_length)
                    if isinstance(emb, torch.Tensor):
                        emb_np = emb.cpu().numpy()
                    else:
                        emb_np = np.asarray(emb)
            else:
                # use the model's batching helper as recommended in README
                emb_np = self.model._do_encode(
                    texts,
                    batch_size=batch_size,
                    instruction=instruction,
                    max_length=max_length,
                    num_workers=4,
                    return_numpy=True
                )

            if req.normalize:
                norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                emb_np = emb_np / norms

            return EmbedResponse(embeddings=emb_np.tolist())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"embedding error: {str(e)}")

def deployment_embed(args: Dict[str, str]):
    # This factory is the import target for serveConfigV2
    return EmbeddingService.bind(**args)

