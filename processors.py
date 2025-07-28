import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
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
    section_title: Optional[str] = None  # Added for better section tracking
    importance_rank: Optional[int] = None  # Added for importance ranking

@dataclass
class RetrievalResult:
    """Result from FAISS retrieval"""
    chunk: DocumentChunk
    score: float

@dataclass
class ProcessedDocument:
    """Represents a fully processed document"""
    filename: str
    title: str
    doc_id: str
    chunks: List[DocumentChunk]
    total_pages: int
    extracted_at: str

class PDFProcessor:
    """Extract layout-aware text from PDFs"""
    
    def __init__(self):
        self.processed_docs: List[ProcessedDocument] = []
    
    def process_pdf_collection(self, pdf_paths: List[Path]) -> List[ProcessedDocument]:
        """Process a collection of PDFs and return structured documents"""
        processed_docs = []
        
        for pdf_path in pdf_paths:
            if not pdf_path.exists():
                logger.warning(f"PDF file not found: {pdf_path}")
                continue
                
            try:
                doc_structure = self.extract_text_with_layout(pdf_path)
                if doc_structure:
                    # Create chunks from the document structure
                    chunker_instance = TextChunker()
                    chunks = chunker_instance.chunk_document(doc_structure)
                    
                    # Extract title from first heading or use filename
                    title = self._extract_title(doc_structure) or pdf_path.stem
                    
                    processed_doc = ProcessedDocument(
                        filename=pdf_path.name,
                        title=title,
                        doc_id=doc_structure['doc_id'],
                        chunks=chunks,
                        total_pages=doc_structure['total_pages'],
                        extracted_at=doc_structure['extracted_at']
                    )
                    processed_docs.append(processed_doc)
                    logger.info(f"Successfully processed {pdf_path.name}")
                    
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                continue
        
        self.processed_docs = processed_docs
        return processed_docs
    
    def _extract_title(self, doc_structure: Dict[str, Any]) -> Optional[str]:
        """Extract document title from first page headings"""
        if not doc_structure.get('pages'):
            return None
            
        first_page = doc_structure['pages'][0]
        for element in first_page.get('elements', []):
            if element.get('element_type') == 'heading':
                title = element.get('text', '').strip()
                if title and len(title) < 200:  # Reasonable title length
                    return title
        return None
    
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
        self.current_section = None  # Track current section for better chunking
    
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
                
                # Update current section if this is a heading
                if element.get('element_type') == 'heading':
                    self.current_section = text[:100]  # Limit section title length
                
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
                    'chunk_type': element.get('element_type', 'paragraph'),
                    'section_title': self.current_section
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
        if not db_path.exists():
            raise FileNotFoundError(f"Index file not found: {db_path}")
            
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
            raise ValueError("FAISS index not loaded. Please embed chunks first or load existing index.")
        
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

class ImportanceRanker:
    """Rank document sections by importance based on persona and job"""
    
    def __init__(self, embedder: VectorEmbedder):
        self.embedder = embedder
    
    def rank_sections(self, chunks: List[DocumentChunk], persona: str, job_to_be_done: str, 
                     top_k: int = 10) -> List[DocumentChunk]:
        """Rank chunks by importance for given persona and job"""
        # Create a combined query from persona and job
        ranking_query = f"Role: {persona}. Task: {job_to_be_done}"
        
        # Get all chunks with scores
        retrieval_results = self.embedder.retrieve(ranking_query, top_k=len(chunks))
        
        # Assign importance ranks
        ranked_chunks = []
        for rank, result in enumerate(retrieval_results[:top_k], 1):
            chunk = result.chunk
            chunk.importance_rank = rank
            ranked_chunks.append(chunk)
        
        logger.info(f"Ranked {len(ranked_chunks)} sections by importance")
        return ranked_chunks

class DocumentRetriever:
    """Enhanced retriever with persona-based filtering"""
    
    def __init__(self, embedder: VectorEmbedder):
        self.embedder = embedder
        self.ranker = ImportanceRanker(embedder)
    
    def retrieve_for_persona(self, query: str, persona: str, job_to_be_done: str, 
                           top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve documents with persona-based relevance"""
        # Enhanced query with persona context
        enhanced_query = f"As a {persona} working on {job_to_be_done}, I need: {query}"
        
        results = self.embedder.retrieve(enhanced_query, top_k)
        
        # Add importance ranking to results
        for result in results:
            if not result.chunk.importance_rank:
                # Calculate importance on the fly if not pre-computed
                importance_query = f"{persona} {job_to_be_done}"
                temp_results = self.embedder.retrieve(importance_query, top_k=100)
                for i, temp_result in enumerate(temp_results, 1):
                    if temp_result.chunk.chunk_id == result.chunk.chunk_id:
                        result.chunk.importance_rank = i
                        break
        
        return results

class OutputWriter:
    """Handle structured output writing"""
    
    @staticmethod
    def save_json(data: Dict[str, Any], output_path: Path):
        """Save data as JSON with proper formatting"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Saved JSON output to {output_path}")
    
    @staticmethod
    def load_json(input_path: Path) -> Dict[str, Any]:
        """Load JSON data"""
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded JSON data from {input_path}")
        return data
    
    @staticmethod
    def create_input_json(challenge_id: str, test_case_name: str, documents: List[ProcessedDocument],
                         persona: Dict[str, str], job_to_be_done: Dict[str, str]) -> Dict[str, Any]:
        """Create input JSON structure"""
        return {
            "challenge_info": {
                "challenge_id": challenge_id,
                "test_case_name": test_case_name
            },
            "documents": [
                {"filename": doc.filename, "title": doc.title} 
                for doc in documents
            ],
            "persona": persona,
            "job_to_be_done": job_to_be_done
        }
    
    @staticmethod
    def create_output_json(input_documents: List[str], persona: str, job_to_be_done: str,
                          extracted_sections: List[Dict], subsection_analysis: List[Dict]) -> Dict[str, Any]:
        """Create output JSON structure"""
        return {
            "metadata": {
                "input_documents": input_documents,
                "persona": persona,
                "job_to_be_done": job_to_be_done
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }

# Factory functions for easy import
def pdf_processor() -> PDFProcessor:
    """Create PDF processor instance"""
    return PDFProcessor()

def chunker(chunk_size: int = 512, overlap: int = 50) -> TextChunker:
    """Create text chunker instance"""
    return TextChunker(chunk_size, overlap)

def embedder(model_path: str = "models/all-MiniLM-L6-v2") -> VectorEmbedder:
    """Create vector embedder instance"""
    return VectorEmbedder(model_path)

def retriever(embedder_instance: VectorEmbedder) -> DocumentRetriever:
    """Create document retriever instance"""
    return DocumentRetriever(embedder_instance)

def output_writer() -> OutputWriter:
    """Create output writer instance"""
    return OutputWriter()

def importance_ranker(embedder_instance: VectorEmbedder) -> ImportanceRanker:
    """Create importance ranker instance"""
    return ImportanceRanker(embedder_instance)