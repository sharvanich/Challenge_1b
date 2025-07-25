import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pdfplumber
from dataclasses import dataclass, asdict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of document with metadata"""
    text: str
    doc_id: str
    chunk_id: str
    page_num: int
    font_info: Dict[str, Any]
    position: Dict[str, float]
    chunk_type: str  # 'heading', 'paragraph', 'table', etc.

@dataclass
class RetrievalResult:
    """Result from FAISS retrieval"""
    chunk: DocumentChunk
    score: float

class PDFProcessor:
    """Extract layout-aware text from PDFs"""
    
    def extract_text_with_layout(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract structured text with font and layout information"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                doc_structure = {
                    'doc_id': pdf_path.stem,
                    'total_pages': len(pdf.pages),
                    'pages': [],
                    'extracted_at': datetime.now().isoformat()
                }
                
                for page_num, page in enumerate(pdf.pages, 1):
                    chars = page.chars
                    if chars:
                        elements = self._group_text_elements(chars)
                        doc_structure['pages'].append({
                            'page_num': page_num,
                            'elements': elements
                        })
                
                logger.info(f"Extracted {len(doc_structure['pages'])} pages from {pdf_path.name}")
                return doc_structure
                
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path.name}: {e}")
            return {}
    
    def _group_text_elements(self, chars: List[Dict]) -> List[Dict]:
        """Group characters into text elements based on font and position"""
        if not chars:
            return []
        
        elements = []
        current_element = self._init_element()
        avg_size = sum(c.get('size', 12) for c in chars) / len(chars)
        
        for char in chars:
            font_key = f"{char.get('fontname', 'unknown')}_{char.get('size', 0)}"
            
            # Check if we should start a new element
            if (current_element['text'] and 
                (font_key != current_element['font'] or 
                 abs(char.get('y0', 0) - current_element['y0']) > char.get('size', 12) * 1.5)):
                
                if current_element['text'].strip():
                    # Determine element type based on font size
                    if current_element['size'] > avg_size * 1.2:
                        current_element['element_type'] = 'heading'
                    elements.append(current_element)
                
                current_element = self._init_element(char, font_key)
            else:
                # Continue current element
                self._update_element(current_element, char)
        
        # Add final element
        if current_element['text'].strip():
            elements.append(current_element)
        
        return elements
    
    def _init_element(self, char: Dict = None, font_key: str = '') -> Dict:
        """Initialize a new text element"""
        if char is None:
            return {
                'text': '', 'font': '', 'size': 0,
                'x0': float('inf'), 'y0': float('inf'), 'x1': 0, 'y1': 0,
                'element_type': 'paragraph'
            }
        
        return {
            'text': char.get('text', ''),
            'font': font_key,
            'size': char.get('size', 12),
            'x0': char.get('x0', 0),
            'y0': char.get('y0', 0),
            'x1': char.get('x1', 0),
            'y1': char.get('y1', 0),
            'element_type': 'paragraph'
        }
    
    def _update_element(self, element: Dict, char: Dict):
        """Update current element with new character"""
        element['text'] += char.get('text', '')
        element['x0'] = min(element['x0'], char.get('x0', 0))
        element['y0'] = min(element['y0'], char.get('y0', 0))
        element['x1'] = max(element['x1'], char.get('x1', 0))
        element['y1'] = max(element['y1'], char.get('y1', 0))

class TextChunker:
    """Break documents into semantic chunks"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_document(self, doc_structure: Dict[str, Any]) -> List[DocumentChunk]:
        """Convert document structure into chunks"""
        chunks = []
        doc_id = doc_structure.get('doc_id', 'unknown')
        
        for page in doc_structure.get('pages', []):
            page_num = page.get('page_num', 1)
            
            for element in page.get('elements', []):
                text = element.get('text', '').strip()
                if not text:
                    continue
                
                chunk_base = {
                    'doc_id': doc_id,
                    'page_num': page_num,
                    'font_info': {
                        'font': element.get('font', ''),
                        'size': element.get('size', 12)
                    },
                    'position': {
                        'x0': element.get('x0', 0),
                        'y0': element.get('y0', 0),
                        'x1': element.get('x1', 0),
                        'y1': element.get('y1', 0)
                    },
                    'chunk_type': element.get('element_type', 'paragraph')
                }
                
                if len(text) <= self.chunk_size:
                    # Single chunk for short text
                    chunks.append(DocumentChunk(
                        text=text,
                        chunk_id=f"{doc_id}_p{page_num}_{len(chunks)}",
                        **chunk_base
                    ))
                else:
                    # Split long text into overlapping chunks
                    words = text.split()
                    for i in range(0, len(words), self.chunk_size - self.overlap):
                        chunk_text = ' '.join(words[i:i + self.chunk_size])
                        chunks.append(DocumentChunk(
                            text=chunk_text,
                            chunk_id=f"{doc_id}_p{page_num}_{len(chunks)}",
                            **chunk_base
                        ))
        
        logger.info(f"Created {len(chunks)} chunks for document {doc_id}")
        return chunks

class VectorEmbedder:
    """Generate and manage vector embeddings"""
    
    def __init__(self, model_path: str = "models/all-MiniLM-L6-v2"):
        self.model_path = model_path
        self.model = None
        self.faiss_index = None
        self.chunk_store = []
        
    def _load_model(self):
        """Load the embedding model"""
        if self.model is not None:
            return
        
        try:
            self.model = SentenceTransformer(self.model_path)
            logger.info(f"Loaded embedding model from {self.model_path}")
        except Exception as e:
            logger.warning(f"Loading online model due to local error: {e}")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed_chunks(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Create embeddings for chunks and build FAISS index"""
        self._load_model()
        
        texts = [chunk.text for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create and populate FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings.astype('float32'))
        
        self.chunk_store = chunks
        
        logger.info(f"Built FAISS index with {self.faiss_index.ntotal} vectors")
        return embeddings
    
    def save_index(self, db_path: Path):
        """Save FAISS index and chunk store"""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        index_data = {
            'faiss_index': faiss.serialize_index(self.faiss_index),
            'chunks': [asdict(chunk) for chunk in self.chunk_store],
            'model_path': self.model_path
        }
        
        with open(db_path, 'wb') as f:
            pickle.dump(index_data, f)
        logger.info(f"Saved FAISS index to {db_path}")
    
    def load_index(self, db_path: Path):
        """Load FAISS index and chunk store"""
        with open(db_path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.faiss_index = faiss.deserialize_index(index_data['faiss_index'])
        self.chunk_store = [DocumentChunk(**chunk_dict) for chunk_dict in index_data['chunks']]
        self.model_path = index_data.get('model_path', self.model_path)
        
        logger.info(f"Loaded FAISS index with {len(self.chunk_store)} chunks")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve top-k most similar chunks for a query"""
        self._load_model()
        
        if not self.faiss_index:
            raise ValueError("FAISS index not loaded")
        
        # Embed and search
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunk_store):
                results.append(RetrievalResult(
                    chunk=self.chunk_store[idx],
                    score=float(score)
                ))
        
        logger.info(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
        return results

class OutputWriter:
    """Handle structured output writing"""
    
    @staticmethod
    def save_json(data: Dict[str, Any], output_path: Path):
        """Save data as JSON"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    @staticmethod
    def load_json(input_path: Path) -> Dict[str, Any]:
        """Load JSON data"""
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)

# Factory functions for easy import
def pdf_processor() -> PDFProcessor:
    return PDFProcessor()

def chunker(chunk_size: int = 512, overlap: int = 50) -> TextChunker:
    return TextChunker(chunk_size, overlap)

def embedder(model_path: str = "models/all-MiniLM-L6-v2") -> VectorEmbedder:
    return VectorEmbedder(model_path)

def output_writer() -> OutputWriter:
    return OutputWriter()