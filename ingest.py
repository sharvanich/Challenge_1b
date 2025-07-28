import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import sys
import json
import re

from processors import pdf_processor, chunker, embedder, output_writer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IngestionPipeline:
    """Ingestion pipeline that takes persona and job from user and processes PDFs"""
    
    def __init__(self, 
                 collection_dir: Path = Path("data/collection"),
                 input_dir: Path = Path("data/input"),
                 db_path: Path = Path("db/faiss_store.pkl"),
                 model_path: str = "models/all-MiniLM-L6-v2"):
        
        self.collection_dir = collection_dir
        self.input_dir = input_dir
        self.db_path = db_path
        
        # Initialize processors
        self.pdf_proc = pdf_processor()
        self.text_chunker = chunker(chunk_size=512, overlap=50)
        self.vector_embedder = embedder(model_path)
        self.writer = output_writer()
        
        # Title extraction patterns
        self.title_patterns = self._initialize_title_patterns()
        
        self._ensure_directories()
    
    def _initialize_title_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for title extraction"""
        return {
            'title_indicators': [
                r'^title\s*[:\-]?\s*(.+)$',
                r'^(.{10,80})\s*$',  # Lines of reasonable title length
                r'^\s*([A-Z][^.!?]*[^.!?\s])\s*$',  # Capitalized, no ending punctuation
                r'^\s*([^a-z]*[A-Z][^a-z]*)\s*$',  # All caps or title case
            ],
            'exclude_patterns': [
                r'^\s*(page|chapter|section|table|figure)\s*\d+',
                r'^\s*\d+\s*$',  # Just numbers
                r'^[^\w]*$',  # No alphanumeric characters
                r'^\s*(continued|cont\.|see also|references?)\s*',
                r'^\s*(abstract|summary|conclusion|introduction)\s*$',
            ]
        }
    
    def _ensure_directories(self) -> None:
        """Ensure required directories exist"""
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def find_pdfs(self) -> List[Path]:
        """Find all PDF files in collection directory"""
        if not self.collection_dir.exists():
            logger.error(f"Collection directory does not exist: {self.collection_dir}")
            return []
        
        pdf_files = list(self.collection_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        return pdf_files
    
    def get_user_input(self) -> Dict[str, Any]:
        """Get persona and job to be done from user"""
        print("\n" + "="*60)
        print("DOCUMENT INGESTION CONFIGURATION")
        print("="*60)
        
        # Get persona information
        print("\n1. PERSONA CONFIGURATION:")
        persona_role = input("Enter the persona : ").strip()
        if not persona_role:
            persona_role = "Document Analyst"
        
        # Get job/task information
        print("\n2. JOB TO BE DONE:")
        job_task = input("Enter the job to be done : ").strip()
        if not job_task:
            job_task = "Analyze and extract information from documents"
        
        # Get challenge information
        print("\n3. COLLECTION DETAILS:")
        challenge_id = input("Challenge ID (default: 'round_1b_001'): ").strip() or "round_1b_001"
        
        return {
            "persona": {"role": persona_role},
            "job_to_be_done": {"task": job_task},
            "challenge_id": challenge_id
        }
    
    def extract_title_from_pdf(self, pdf_path: Path) -> str:
        """Extract title from PDF using layout analysis"""
        try:
            logger.debug(f"Extracting title from {pdf_path.name}")
            
            # Extract document structure
            doc_structure = self.pdf_proc.extract_text_with_layout(pdf_path)
            if not doc_structure or not doc_structure.get('pages'):
                logger.warning(f"No content extracted from {pdf_path.name}")
                return self._generate_title_from_filename(pdf_path.stem)
            
            # Look for title in first few pages
            title_candidates = []
            
            for page in doc_structure['pages'][:3]:  # Check first 3 pages
                for element in page.get('elements', []):
                    text = element.get('text', '').strip()
                    if not text or len(text) < 5:
                        continue
                    
                    # Score this element as potential title
                    score = self._score_title_candidate(text, element)
                    if score > 0:
                        title_candidates.append({
                            'text': text,
                            'score': score,
                            'element_type': element.get('element_type', 'paragraph')
                        })
            
            # Select best title candidate
            if title_candidates:
                title_candidates.sort(key=lambda x: x['score'], reverse=True)
                best_title = title_candidates[0]['text']
                return self._clean_title(best_title)
            
            return self._generate_title_from_filename(pdf_path.stem)
            
        except Exception as e:
            logger.warning(f"Title extraction error for {pdf_path.name}: {e}")
            return self._generate_title_from_filename(pdf_path.stem)
    
    def _score_title_candidate(self, text: str, element: Dict) -> float:
        """Score text as potential title"""
        score = 0.0
        text_lower = text.lower()
        
        # Element type scoring
        if element.get('element_type') == 'heading':
            score += 5.0
        
        # Font size scoring (larger fonts more likely to be titles)
        font_size = element.get('size', 12)
        if font_size > 16:
            score += 3.0
        elif font_size > 14:
            score += 2.0
        
        # Length scoring (titles are usually 10-80 characters)
        length = len(text)
        if 15 <= length <= 60:
            score += 3.0
        elif 10 <= length <= 80:
            score += 1.0
        elif length > 100:
            score -= 2.0
        
        # Pattern matching for titles
        for pattern in self.title_patterns['title_indicators']:
            if re.match(pattern, text, re.IGNORECASE):
                score += 2.0
                break
        
        # Exclude patterns (negative scoring)
        for pattern in self.title_patterns['exclude_patterns']:
            if re.match(pattern, text, re.IGNORECASE):
                score -= 5.0
        
        # Capitalization scoring
        if text.istitle():
            score += 2.0
        elif text.isupper() and length > 5:
            score += 1.0
        
        # Avoid common non-title content
        if any(word in text_lower for word in ['page', 'copyright', 'published', 'abstract']):
            score -= 3.0
        
        return max(score, 0.0)
    
    def _clean_title(self, title: str) -> str:
        """Clean and format extracted title"""
        # Remove extra whitespace
        title = ' '.join(title.split())
        
        # Remove common prefixes
        prefixes_to_remove = ['title:', 'subject:', 'document:', 'report:']
        for prefix in prefixes_to_remove:
            if title.lower().startswith(prefix):
                title = title[len(prefix):].strip()
        
        # Ensure proper capitalization
        if title.islower() or title.isupper():
            title = title.title()
        
        # Limit length
        if len(title) > 80:
            title = title[:77] + "..."
        
        return title
    
    def _generate_title_from_filename(self, filename: str) -> str:
        """Generate a readable title from filename"""
        # Clean filename
        title = filename.replace('_', ' ').replace('-', ' ')
        
        # Remove common extensions and patterns
        title = re.sub(r'\.(pdf|doc|docx)$', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\d{4}-\d{2}-\d{2}', '', title)  # Remove dates
        title = re.sub(r'v\d+(\.\d+)?', '', title)  # Remove version numbers
        
        # Capitalize properly
        title = ' '.join(word.capitalize() for word in title.split())
        
        return title or "Document"
    
    def generate_test_case_info(self, persona: Dict, job_to_be_done: Dict, pdf_files: List[Path]) -> Dict[str, str]:
        """Generate test case name and description based on persona and job"""
        try:
            # Extract key information
            role = persona.get('role', 'Analyst')
            task = job_to_be_done.get('task', 'Document analysis')
            
            # Generate test case name from task
            test_case_name = self._generate_test_case_name(task)
            
            # Generate description
            description = f"{role} working with {len(pdf_files)} documents to {task.lower()}"
            
            return {
                "test_case_name": test_case_name,
                "description": description
            }
            
        except Exception as e:
            logger.error(f"Error generating test case info: {e}")
            return {
                "test_case_name": "document_analysis_task",
                "description": f"Document analysis task with {len(pdf_files)} files"
            }
    
    def _generate_test_case_name(self, task: str) -> str:
        """Generate test case name from task description"""
        task_lower = task.lower()
        
        # Extract key action words
        action_words = []
        common_actions = ['analyze', 'review', 'extract', 'summarize', 'compare', 'evaluate', 'process', 'research']
        
        for action in common_actions:
            if action in task_lower:
                action_words.append(action)
        
        # If no actions found, use generic terms
        if not action_words:
            if 'document' in task_lower:
                action_words.append('document')
            action_words.append('analysis')
        
        # Clean and join (limit to 3 words)
        test_case_name = '_'.join(action_words[:3])
        return test_case_name if test_case_name else 'document_analysis_task'
    
    def process_pdf_collection(self, pdf_files: List[Path], user_config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Process collection of PDFs using processors.py"""
        logger.info("Processing PDF collection...")
        
        # Use the corrected PDF processor to handle collection
        processed_docs = self.pdf_proc.process_pdf_collection(pdf_files)
        
        # Convert to required format
        documents = []
        for doc in processed_docs:
            documents.append({
                "filename": doc.filename,
                "title": doc.title
            })
            logger.info(f"✓ Processed {doc.filename} -> '{doc.title}'")
        
        return documents
    
    def create_input_json_structure(self, pdf_files: List[Path], user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create input JSON structure as specified in requirements"""
        
        # Process PDFs to get documents with titles
        documents = self.process_pdf_collection(pdf_files, user_config)
        
        # Generate test case information
        test_case_info = self.generate_test_case_info(
            user_config["persona"], 
            user_config["job_to_be_done"], 
            pdf_files
        )
        
        # Create the required input JSON structure
        input_structure = {
            "challenge_info": {
                "challenge_id": user_config["challenge_id"],
                "test_case_name": test_case_info["test_case_name"]
            },
            "documents": documents,
            "persona": user_config["persona"],
            "job_to_be_done": user_config["job_to_be_done"]
        }
        
        return input_structure
    
    def build_vector_index(self) -> bool:
        """Build vector index from processed PDFs"""
        try:
            # Find all PDFs in collection
            pdf_files = self.find_pdfs()
            if not pdf_files:
                logger.error("No PDF files found for indexing")
                return False
            
            logger.info(f"Building vector index from {len(pdf_files)} PDFs")
            
            # Process all PDFs and collect chunks
            all_chunks = []
            for pdf_path in pdf_files:
                try:
                    # Extract document structure
                    doc_structure = self.pdf_proc.extract_text_with_layout(pdf_path)
                    if doc_structure:
                        # Create chunks from document
                        chunks = self.text_chunker.chunk_document(doc_structure)
                        all_chunks.extend(chunks)
                        logger.debug(f"Added {len(chunks)} chunks from {pdf_path.name}")
                except Exception as e:
                    logger.error(f"Error processing {pdf_path.name} for indexing: {e}")
                    continue
            
            if not all_chunks:
                logger.error("No chunks created from PDFs")
                return False
            
            logger.info(f"Embedding {len(all_chunks)} chunks...")
            
            # Generate embeddings and build FAISS index
            self.vector_embedder.embed_chunks(all_chunks)
            self.vector_embedder.save_index(self.db_path)
            
            logger.info(f"✓ Vector index saved with {len(all_chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error building vector index: {e}")
            return False
    
    def display_configuration_summary(self, structure: Dict[str, Any]):
        """Display configuration summary"""
        print(f"\n{'='*60}")
        print("INGESTION CONFIGURATION SUMMARY")
        print(f"{'='*60}")
        
        challenge_info = structure.get('challenge_info', {})
        persona = structure.get('persona', {})
        job = structure.get('job_to_be_done', {})
        documents = structure.get('documents', [])
        
        print(f"Challenge ID: {challenge_info.get('challenge_id', 'N/A')}")
        print(f"Test Case: {challenge_info.get('test_case_name', 'N/A')}")
        print(f"Persona Role: {persona.get('role', 'N/A')}")
        print(f"Task: {job.get('task', 'N/A')}")
        
        print(f"\nDocuments ({len(documents)}):")
        for i, doc in enumerate(documents, 1):
            print(f"  {i}. {doc['filename']} -> '{doc['title']}'")
        
        print(f"{'='*60}\n")
    
    def run_pipeline(self, single_file: Optional[Path] = None, rebuild_only: bool = False) -> bool:
        """Run the complete ingestion pipeline"""
        try:
            if rebuild_only:
                logger.info("Rebuilding vector index only")
                return self.build_vector_index()
            
            # Get user input for persona and job
            user_config = self.get_user_input()
            
            # Find PDF files
            if single_file:
                if not single_file.exists():
                    logger.error(f"Single file not found: {single_file}")
                    return False
                pdf_files = [single_file]
            else:
                pdf_files = self.find_pdfs()
                if not pdf_files:
                    logger.error("No PDF files found")
                    return False
            
            # Create input JSON structure
            input_structure = self.create_input_json_structure(pdf_files, user_config)
            
            # Save input JSON
            input_json_path = self.input_dir / "collection_input.json"
            self.writer.save_json(input_structure, input_json_path)
            
            # Display summary
            self.display_configuration_summary(input_structure)
            
            # Build vector index for retrieval
            if not self.build_vector_index():
                logger.warning("Vector index build failed, but input JSON created successfully")
            
            logger.info("✓ Ingestion pipeline completed successfully!")
            logger.info(f"✓ Input JSON saved to: {input_json_path}")
            
            return True
            
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="PDF Ingestion Pipeline with Persona and Job Configuration")
    parser.add_argument("--collection-dir", type=Path, default=Path("data/collection"),
                       help="Directory containing PDF files")
    parser.add_argument("--input-dir", type=Path, default=Path("data/input"),
                       help="Directory to save input JSON files")
    parser.add_argument("--db-path", type=Path, default=Path("db/faiss_store.pkl"),
                       help="Path to save FAISS index")
    parser.add_argument("--model-path", type=str, default="models/all-MiniLM-L6-v2",
                       help="Path to embedding model")
    parser.add_argument("--single-file", type=Path,
                       help="Process only a single PDF file")
    parser.add_argument("--rebuild-index", action="store_true",
                       help="Only rebuild the vector index from existing PDFs")
    parser.add_argument("--interactive", action="store_true", default=True,
                       help="Run in interactive mode (default)")
    
    args = parser.parse_args()
    
    try:
        pipeline = IngestionPipeline(
            collection_dir=args.collection_dir,
            input_dir=args.input_dir,
            db_path=args.db_path,
            model_path=args.model_path
        )
        
        success = pipeline.run_pipeline(
            single_file=args.single_file,
            rebuild_only=args.rebuild_index
        )
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()