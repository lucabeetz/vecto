import numpy as np
import uuid
from enum import Enum
from pydantic import BaseModel
from provider import run_openai_embedding


class EmbeddingType(Enum):
    TEXT = "TEXT"
    IMAGE = "IMAGE"


class TextEmbeddingProvider(Enum):
    OPEN_AI = "OPEN_AI"


class ImageEmbeddingProvider(Enum):
    CLIP = "CLIP"


embedding_provider_to_function = {
    TextEmbeddingProvider.OPEN_AI: run_openai_embedding,
}


class Vecto:
    text_embedding_provider: TextEmbeddingProvider = TextEmbeddingProvider.OPEN_AI
    image_embedding_provider: ImageEmbeddingProvider = ImageEmbeddingProvider.CLIP

    text_embeddings: np.ndarray = None
    image_embeddings: np.ndarray = np.array([])

    metadata: dict = {}
    index_to_uuid: dict = {}

    def add(self, data: str, metadata: dict, type: EmbeddingType = EmbeddingType.TEXT):
        """Store embedding with metadata"""
        embedding_uuid = uuid.uuid4()
        self.metadata[embedding_uuid] = metadata

        if type == EmbeddingType.TEXT:
            self._add_text_embedding(data, embedding_uuid)
        else:
            raise NotImplementedError

        return embedding_uuid

    def query(self, data: str, type: EmbeddingType = EmbeddingType.TEXT) -> dict:
        if type == EmbeddingType.TEXT:
            return self._query_text_embedding(data)
        else:
            raise NotImplementedError

    def _add_text_embedding(self, text: str, uuid: uuid):
        embedding_fn = embedding_provider_to_function[self.text_embedding_provider]
        embedding = embedding_fn(text).reshape(1, -1)

        # Concatenate new embedding
        self.text_embeddings = (
            embedding
            if self.text_embeddings is None
            else np.concatenate((self.text_embeddings, embedding), axis=0)
        )

        self.index_to_uuid[len(self.text_embeddings) - 1] = uuid
        print(f"Updated embeddings: {self.text_embeddings.shape}")
        print(f"Index to UUID: {self.index_to_uuid}")

    def _add_image_embedding(self, image: str, uuid: uuid):
        raise NotImplementedError

    def _query_text_embedding(self, text: str) -> dict:
        embedding_fn = embedding_provider_to_function[self.text_embedding_provider]
        embedding = embedding_fn(text)

        # calculate cos sim
        cos_sim = np.dot(self.text_embeddings, embedding) / (
            np.linalg.norm(self.text_embeddings) * np.linalg.norm(embedding)
        )

        # find closest embedding
        closest_embedding = np.argmax(cos_sim)
        embedding_uuid = self.index_to_uuid[closest_embedding]
        return self.metadata[embedding_uuid]

    def _query_image_embedding(self, image: str) -> dict:
        raise NotImplementedError
