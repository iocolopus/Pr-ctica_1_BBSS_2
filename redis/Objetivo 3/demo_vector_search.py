import argparse
import os

import numpy as np
import pandas as pd
import redis
from redis.commands.search.field import NumericField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

INDEX_NAME = "idx:cards_semantic"
KEY_PREFIX = "cards:"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def connect_redis() -> redis.Redis:
    """Uses a binary-safe client for vector fields (no automatic UTF-8 decoding)."""
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    db = int(os.getenv("REDIS_DB", "0"))
    password = os.getenv("REDIS_PASSWORD")

    return redis.Redis(
        host=host,
        port=port,
        db=db,
        password=password,
        decode_responses=False,
    )


def create_index(r: redis.Redis, dim: int) -> None:
    schema = (
        TextField("name"),
        TextField("text"),
        TextField("faction_code"),
        NumericField("xp"),
        VectorField(
            "embedding",
            "HNSW",
            {
                "TYPE": "FLOAT32",
                "DIM": dim,
                "DISTANCE_METRIC": "COSINE",
                "M": 16,
                "EF_CONSTRUCTION": 200,
            },
        ),
    )

    definition = IndexDefinition(prefix=[KEY_PREFIX], index_type=IndexType.HASH)

    try:
        r.ft(INDEX_NAME).info()
        print(f"Indice '{INDEX_NAME}' ya existe. Se reutiliza.")
    except Exception:
        r.ft(INDEX_NAME).create_index(fields=schema, definition=definition)
        print(f"Indice '{INDEX_NAME}' creado correctamente.")


def ingest_cards(
    r: redis.Redis,
    model: SentenceTransformer,
    csv_path: str,
    limit: int | None = None,
) -> int:
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)

    texts = (
        df["name"].fillna("")
        + "\n"
        + df["text"].fillna("")
        + "\n"
        + df["traits"].fillna("")
    ).tolist()

    vectors = model.encode(texts, normalize_embeddings=True).astype(np.float32)

    pipe = r.pipeline(transaction=False)

    for i, row in df.iterrows():
        key = f"{KEY_PREFIX}{row['code']}"
        mapping = {
            "code": str(row.get("code", "")),
            "name": str(row.get("name", "")),
            "text": str(row.get("text", "")),
            "type_code": str(row.get("type_code", "")),
            "traits": str(row.get("traits", "")),
            "xp": float(row.get("xp", 0) or 0),
            "faction_code": str(row.get("faction_code", "")),
            "embedding": vectors[i].tobytes(),
        }
        pipe.hset(key, mapping=mapping)

    pipe.execute()
    print(f"Insertadas/actualizadas {len(df)} cartas con embeddings en Redis.")
    return len(df)


def search_top_k(
    r: redis.Redis,
    model: SentenceTransformer,
    query_text: str,
    k: int = 5,
) -> list:
    query_vector = model.encode([query_text], normalize_embeddings=True).astype(np.float32)[0]

    q = (
        Query(f"*=>[KNN {k} @embedding $query_vector AS score]")
        .return_fields("code", "name", "text", "xp", "faction_code", "score")
        .sort_by("score", asc=True)
        .dialect(2)
    )

    res = r.ft(INDEX_NAME).search(q, query_params={"query_vector": query_vector.tobytes()})
    return res.docs


def b2s(value: bytes | str) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo: indice vectorial en Redis + busqueda semantica Top-5 de cartas"
    )
    parser.add_argument(
        "--csv",
        default="data/cards_final_with_xp.csv",
        help="Ruta CSV con cartas",
    )
    parser.add_argument(
        "--query",
        default="busco cartas de apoyo para investigar y manipular fichas del caos",
        help="Texto de consulta semantica",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Numero de resultados",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Cuantas cartas indexar (para demo rapida)",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Si ya tienes embeddings cargados, no reindexa datos",
    )

    args = parser.parse_args()

    r = connect_redis()
    r.ping()

    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()

    create_index(r, dim)

    if not args.skip_ingest:
        ingest_cards(r, model, args.csv, limit=args.limit)

    docs = search_top_k(r, model, args.query, k=args.k)

    print("\nTop resultados semanticos:\n")
    for i, d in enumerate(docs, start=1):
        score = float(b2s(d.score))
        print(f"{i}. [{b2s(d.code)}] {b2s(d.name)} | faction={b2s(d.faction_code)} | xp={b2s(d.xp)} | distance={score:.5f}")
        print(f"   {b2s(d.text)[:180]}...")


if __name__ == "__main__":
    main()
