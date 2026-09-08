import os
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel



MODEL_NAME = "distilbert-base-uncased"

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


class DistilBERTEmbedder:

    def __init__(
        self,
        model_name=MODEL_NAME
    ):

        print(
            f"Loading model: {model_name}"
        )

        print(
            f"Using device: {DEVICE}"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
        if tokenizer is None:
            raise ValueError(f"Failed to load tokenizer for model: {model_name}")
        self.tokenizer = tokenizer

        self.model = (
            AutoModel.from_pretrained(
                model_name
            )
        )

        self.model.to(DEVICE)

        self.model.eval()


    def encode(
        self,
        texts,
        batch_size=32,
        max_length=128
    ):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized.")

        embeddings = []

        for start in range(
            0,
            len(texts),
            batch_size
        ):

            batch = texts[
                start:start + batch_size
            ]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            encoded = {
                key: value.to(DEVICE)
                for key, value in encoded.items()
            }

            with torch.no_grad():

                outputs = self.model(
                    **encoded
                )

            # DistilBERT does not have a
            # pooler output like BERT.
            #
            # We therefore perform mean pooling
            # over the token embeddings.

            token_embeddings = (
                outputs.last_hidden_state
            )

            attention_mask = (
                encoded["attention_mask"]
            )

            mask = (
                attention_mask
                .unsqueeze(-1)
                .expand(
                    token_embeddings.size()
                )
                .float()
            )

            summed = torch.sum(
                token_embeddings * mask,
                dim=1
            )

            counts = torch.clamp(
                mask.sum(dim=1),
                min=1e-9
            )

            batch_embeddings = (
                summed / counts
            )

            embeddings.append(
                batch_embeddings.cpu().numpy()
            )

            print(
                f"Processed "
                f"{min(start + batch_size, len(texts))}"
                f"/{len(texts)}"
            )

        return np.vstack(
            embeddings
        )


def generate_embeddings(
    texts,
    output_path="results/distilbert_embeddings.npy"
):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    embedder = (
        DistilBERTEmbedder()
    )

    embeddings = embedder.encode(
        list(texts)
    )

    np.save(
        output_path,
        embeddings
    )

    print(
        f"\nSaved embeddings to: "
        f"{output_path}"
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    return embeddings