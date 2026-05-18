#!/usr/bin/env python3
import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import requests
from tqdm import tqdm


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    path: str
    text: str
    start_char: int


def iter_text_files(docs_dir: Path) -> Iterable[Path]:
    for p in docs_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            yield p


def read_text(path: Path) -> str:
    # Lectura simple; si hay PDFs/HTML, conviene convertir antes a texto.
    return path.read_text(encoding="utf-8", errors="replace")


def normalize_whitespace(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(text: str, chunk_chars: int = 1500, overlap_chars: int = 250) -> List[Tuple[str, int]]:
    """
    Chunking simple por caracteres (baseline).
    Devuelve lista de (chunk_text, start_char).
    """
    text = normalize_whitespace(text)
    if not text:
        return []
    out: List[Tuple[str, int]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            out.append((chunk, start))
        if end >= n:
            break
        start = max(0, end - overlap_chars)
    return out


def ollama_embed(texts: List[str], model: str, base_url: str) -> np.ndarray:
    """
    API local de Ollama: POST /api/embeddings
    Respuesta: { "embedding": [...] }
    """
    vecs = []
    for t in tqdm(texts, desc="Embeddings", unit="chunk"):
        r = requests.post(
            f"{base_url}/api/embeddings",
            json={"model": model, "prompt": t},
            timeout=120,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Embeddings failed ({r.status_code}): {r.text}")
        data = r.json()
        vecs.append(np.array(data["embedding"], dtype=np.float32))
    return np.vstack(vecs)


def ollama_generate(prompt: str, model: str, base_url: str, temperature: float = 0.2) -> str:
    """
    API local de Ollama: POST /api/generate
    """
    r = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=300,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Generate failed ({r.status_code}): {r.text}")
    return r.json().get("response", "").strip()


def build_prompt(question: str, retrieved: List[Chunk]) -> str:
    sources = []
    for i, ch in enumerate(retrieved, start=1):
        sources.append(f"[Fuente {i}] archivo={ch.path}\n{ch.text}\n")
    sources_block = "\n".join(sources)

    return f"""Eres un asistente que responde de forma precisa y fundamentada.

Instrucciones:
- Responde usando ÚNICAMENTE la información contenida en las FUENTES.
- Si las FUENTES no contienen suficiente información para responder, indica explícitamente qué falta.
- Cuando afirmes algo, apóyalo citando la fuente correspondiente (por ejemplo: [Fuente 2]).

FUENTES:
{sources_block}

PREGUNTA:
{question}

RESPUESTA:
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="RAG local con Ollama + FAISS (docs .txt/.md).")
    ap.add_argument("--docs", required=True, help="Directorio con documentos .txt/.md")
    ap.add_argument("--question", required=True, help="Pregunta del usuario")
    ap.add_argument("--topk", type=int, default=4, help="Número de chunks a recuperar")
    ap.add_argument("--chunk-chars", type=int, default=1500, help="Tamaño de chunk (caracteres)")
    ap.add_argument("--overlap-chars", type=int, default=250, help="Solapamiento (caracteres)")
    ap.add_argument("--embed-model", default="nomic-embed-text", help="Modelo de embeddings en Ollama")
    ap.add_argument("--llm-model", default="llama3.2:3b", help="Modelo generativo en Ollama")
    ap.add_argument("--ollama-url", default="http://localhost:11434", help="URL base de Ollama")
    args = ap.parse_args()

    docs_dir = Path(args.docs).expanduser().resolve()
    if not docs_dir.exists():
        print(f"ERROR: no existe el directorio: {docs_dir}", file=sys.stderr)
        return 2

    files = sorted(iter_text_files(docs_dir))
    if not files:
        print("ERROR: no se encontraron archivos .txt/.md en el directorio de documentos.", file=sys.stderr)
        return 2

    chunks: List[Chunk] = []
    for p in files:
        text = read_text(p)
        for chunk, start_char in chunk_text(text, chunk_chars=args.chunk_chars, overlap_chars=args.overlap_chars):
            chunks.append(
                Chunk(
                    doc_id=p.name,
                    path=str(p.relative_to(docs_dir)),
                    text=chunk,
                    start_char=start_char,
                )
            )

    if not chunks:
        print("ERROR: no se generaron chunks (documentos vacíos).", file=sys.stderr)
        return 2

    # Embeddings de chunks
    chunk_texts = [c.text for c in chunks]
    X = ollama_embed(chunk_texts, model=args.embed_model, base_url=args.ollama_url)

    # Construir índice FAISS. Normalizamos para aproximar coseno y usamos producto interno.
    import faiss

    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])  # inner product ~= coseno si está normalizado
    index.add(X)

    # Embedding de la pregunta
    q_vec = ollama_embed([args.question], model=args.embed_model, base_url=args.ollama_url)[0:1]
    faiss.normalize_L2(q_vec)

    scores, idxs = index.search(q_vec, k=min(args.topk, len(chunks)))
    retrieved = [chunks[i] for i in idxs[0].tolist()]

    print("\n=== Chunks recuperados (evidencia) ===")
    for rank, (ch, score) in enumerate(zip(retrieved, scores[0].tolist()), start=1):
        snippet = ch.text if len(ch.text) <= 400 else (ch.text[:400] + " ...")
        print(f"\n[{rank}] score={score:.4f} archivo={ch.path} start_char={ch.start_char}")
        print(snippet)

    prompt = build_prompt(args.question, retrieved)
    answer = ollama_generate(prompt, model=args.llm_model, base_url=args.ollama_url, temperature=0.2)

    print("\n=== Respuesta ===")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())