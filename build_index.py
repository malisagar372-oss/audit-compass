"""
Builds a local embedding index from the corpus/ text files.

app.py imports parse_chunks/REGULATOR_FILES/CORPUS_DIR from this file and
builds index.json itself automatically on first startup — you do NOT need
to run this script manually before deploying.

Run it by hand only if you want to rebuild index.json after editing the
corpus/ files without restarting the whole app (e.g. during local
development): `python build_index.py`.
"""
import json
import os
import re

def _resolve_corpus_dir():
    """Find corpus/ whether it sits next to this file (the flattened layout
    used when all files are uploaded directly to a Hugging Face Space's
    root) or one level above it (the original audit-rag/app/ + audit-rag/
    corpus/ zip layout). Checking both makes this work correctly in either
    arrangement without the person deploying it needing to get the folder
    structure exactly right."""
    here = os.path.dirname(os.path.abspath(__file__))
    same_level = os.path.join(here, "corpus")
    one_up = os.path.join(here, "..", "corpus")
    if os.path.isdir(same_level):
        return same_level
    if os.path.isdir(one_up):
        return one_up
    # Neither found yet (e.g. first-ever run before corpus/ is uploaded) —
    # default to same-level, since that's the layout DEPLOY.md instructs.
    return same_level


CORPUS_DIR = _resolve_corpus_dir()
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "index.json")

REGULATOR_FILES = {
    "usfda": "usfda.txt",
    "tga": "tga.txt",
    "ema": "ema.txt",
    "mhra": "mhra.txt",
    "sahpra": "sahpra.txt",
    "anvisa": "anvisa.txt",
    "who": "who.txt",
    "picks": "picks.txt",
    "hc": "hc.txt",
}

REGULATOR_LABELS = {
    "usfda": "US FDA",
    "tga": "Australia TGA",
    "ema": "EU GMP / EMA",
    "mhra": "UK MHRA",
    "sahpra": "South Africa SAHPRA",
    "anvisa": "Brazil ANVISA",
    "who": "WHO Prequalification",
    "picks": "PIC/S (general)",
    "hc": "Health Canada",
}


def parse_chunks(text, regulator_id):
    """Split a corpus file into chunks by TOPIC: blocks, keeping the
    REGULATOR/SOURCE/VERIFY header attached to every chunk so retrieval
    always carries its citation."""
    header_match = re.search(
        r"\[REGULATOR:.*?\]\n\[SOURCE:(.*?)\]\n\[VERIFY AT:(.*?)\]", text, re.S
    )
    source = header_match.group(1).strip() if header_match else ""
    verify_url = header_match.group(2).strip() if header_match else ""

    topic_blocks = re.split(r"\nTOPIC:\s*", text)
    chunks = []
    for block in topic_blocks[1:]:
        lines = block.strip().split("\n", 1)
        topic = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        chunks.append(
            {
                "regulator": regulator_id,
                "regulator_label": REGULATOR_LABELS[regulator_id],
                "topic": topic,
                "text": body,
                "source": source,
                "verify_url": verify_url,
            }
        )
    return chunks


def main():
    from sentence_transformers import SentenceTransformer

    print("Loading local embedding model (first run downloads ~90MB, then cached)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    all_chunks = []
    for reg_id, fname in REGULATOR_FILES.items():
        path = os.path.join(CORPUS_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        all_chunks.extend(parse_chunks(text, reg_id))

    print(f"Parsed {len(all_chunks)} chunks across {len(REGULATOR_FILES)} regulators.")

    texts = [f"{c['topic']}. {c['text']}" for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    for chunk, emb in zip(all_chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f)

    print(f"Wrote {OUTPUT_PATH} ({len(all_chunks)} chunks).")


if __name__ == "__main__":
    main()
