import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
from app.config import invoke_gemini

# ---------------------------------------------------------
# Document Path
# ---------------------------------------------------------

DOCUMENTS_DIR = Path(__file__).resolve().parents[2] / "documents"
DOCUMENT_PATH = DOCUMENTS_DIR / "company_policy.pdf"


# ---------------------------------------------------------
# Structured RAG Output Class
# ---------------------------------------------------------

class RAGOutput(str):
    """
    Subclass of str that preserves backward compatibility with
    existing code expecting a string, while attaching structured
    metadata: contexts (list[str]) and sources (list[str]).
    """
    def __new__(cls, text: str, contexts: Optional[List[str]] = None, sources: Optional[List[str]] = None):
        obj = super().__new__(cls, text)
        obj.contexts = contexts or []
        obj.context = obj.contexts  # alias for RAGAS compatibility
        obj.sources = sources or []
        obj.text = text
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": str(self),
            "contexts": self.contexts,
            "sources": self.sources
        }


# ---------------------------------------------------------
# Load Documents & Chunks with Metadata
# ---------------------------------------------------------

def load_document_chunks() -> List[Dict[str, Any]]:
    """
    Read all PDF files and return structured chunks with page numbers and sources.
    """
    if not DOCUMENTS_DIR.exists():
        return []

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        return []

    chunks = []
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))
            for page_idx, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue
                # Split page into meaningful paragraphs or line groups
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                if not paragraphs:
                    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]

                for para in paragraphs:
                    if len(para) > 15:  # filter noise
                        chunks.append({
                            "source": pdf_path.name,
                            "page": page_idx,
                            "content": para
                        })
        except Exception:
            continue

    return chunks


def load_documents_text() -> str:
    """
    Read text from all PDF files in the documents directory.
    (Preserved for backward compatibility)
    """
    if not DOCUMENTS_DIR.exists():
        return "Company policy document not found."

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        return "Company policy document not found."

    all_texts = []
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            if pages:
                all_texts.append(f"[{pdf_path.name}]\n" + "\n".join(pages))
        except Exception:
            continue

    return "\n\n".join(all_texts) if all_texts else "Company policy document not found."


def load_document():
    """
    Backward-compatible document loader.
    """
    return load_documents_text()


# ---------------------------------------------------------
# RAG Agent
# ---------------------------------------------------------

def rag_agent_structured(query: str) -> Dict[str, Any]:
    """
    Structured RAG execution returning:
    - text: str (the answer/snippets)
    - contexts: list[str] (retrieved document chunks for RAGAS evaluation)
    - sources: list[str] (source documents and pages)
    """
    chunks = load_document_chunks()
    if not chunks:
        doc_text = load_documents_text()
        if doc_text == "Company policy document not found." or not doc_text:
            return {
                "text": "RAG Agent: Company policy document not found.",
                "contexts": [],
                "sources": []
            }

    # Extract query terms
    words = [w for w in re.findall(r"\b\w+\b", query.lower()) if len(w) >= 3]
    variations = set(words)
    for w in words:
        if w.endswith("ies") and len(w) > 4:
            variations.add(w[:-3] + "y")
        if w.endswith("es") and len(w) > 4:
            variations.add(w[:-2])
        if w.endswith("s") and len(w) > 3:
            variations.add(w[:-1])
        if w.endswith("ing") and len(w) > 5:
            variations.add(w[:-3])

    if any(k in words for k in ["wfh", "remote", "remotely"]):
        variations.update(["home", "work", "approval"])

    matched_chunks = []
    for ch in chunks:
        content_lower = ch["content"].lower()
        if any(term in content_lower for term in variations):
            matched_chunks.append(ch)

    if matched_chunks:
        # Deduplicate and sort by relevance
        contexts = [c["content"] for c in matched_chunks[:5]]
        sources = list(dict.fromkeys([f"{c['source']} (p. {c['page']})" for c in matched_chunks[:5]]))
        text = "RAG Agent Result:\n" + "\n".join(contexts)
        return {
            "text": text,
            "contexts": contexts,
            "sources": sources
        }

    # Fallback to LLM semantic extraction if keyword matching yielded no chunks
    document_text = load_documents_text()
    try:
        prompt = (
            f"You are a RAG retrieval agent for company policy documents.\n"
            f"Document Content:\n{document_text}\n\n"
            f"Query: {query}\n"
            f"Extract relevant facts or rules from the documents that answer the query. "
            f"If nothing is relevant, say 'No relevant information found in the company policy document.'"
        )
        ans = invoke_gemini(prompt).strip()
        contexts = [ans] if "No relevant information found" not in ans else []
        sources = ["company_policy.pdf"] if contexts else []
        return {
            "text": f"RAG Agent Result:\n{ans}",
            "contexts": contexts,
            "sources": sources
        }
    except Exception:
        return {
            "text": "RAG Agent: No relevant information found in the company policy document.",
            "contexts": [],
            "sources": []
        }


def rag_agent(query: str) -> RAGOutput:
    """
    RAG agent retrieves relevant information from company documents.
    Returns a RAGOutput instance (subclass of str) that seamlessly behaves
    as a string for existing code, but carries `.contexts` and `.sources`
    for RAGAS evaluation and UI displays.
    """
    res = rag_agent_structured(query)
    return RAGOutput(
        text=res["text"],
        contexts=res["contexts"],
        sources=res["sources"]
    )