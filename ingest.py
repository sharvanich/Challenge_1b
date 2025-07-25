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

class EnhancedIngestionPipeline:
    """Enhanced ingestion pipeline with improved title extraction and persona-aware configuration"""
    
    def __init__(self, 
                 collection_dir: Path = Path("data/collection"),
                 input_dir: Path = Path("data/input"),
                 db_path: Path = Path("db/faiss_store.pkl"),
                 model_path: str = "models/all-MiniLM-L6-v2",
                 challenge_config: Optional[Dict[str, Any]] = None):
        
        self.collection_dir = collection_dir
        self.input_dir = input_dir
        self.db_path = db_path
        self.challenge_config = challenge_config or {}
        
        # Initialize processors
        self.pdf_proc = pdf_processor()
        self.text_chunker = chunker(chunk_size=512, overlap=50)
        self.vector_embedder = embedder(model_path)
        self.writer = output_writer()
        
        # Enhanced title extraction components
        self.title_patterns = self._initialize_title_patterns()
        
        self._ensure_directories()
    
    def _initialize_title_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for better title extraction"""
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
    
    def _extract_title_with_enhanced_ai(self, pdf_path: Path) -> str:
        """Enhanced title extraction using improved patterns and AI analysis"""
        try:
            logger.debug(f"Enhanced AI title extraction for {pdf_path.name}")
            
            # Extract document structure
            doc_structure = self.pdf_proc.extract_text_with_layout(pdf_path)
            if not doc_structure:
                logger.warning(f"No content extracted from {pdf_path.name}")
                return self._generate_title_from_filename(pdf_path.stem)
            
            # Extract and rank title candidates
            candidates = self._extract_enhanced_title_candidates(doc_structure)
            if not candidates:
                return self._generate_title_from_filename(pdf_path.stem)
            
            # Apply enhanced scoring and selection
            best_title = self._select_best_title_candidate(candidates, pdf_path.stem)
            
            logger.debug(f"Selected title: '{best_title}' for {pdf_path.name}")
            return best_title
        
        except Exception as e:
            logger.warning(f"Enhanced title extraction error for {pdf_path.name}: {e}")
            return self._generate_title_from_filename(pdf_path.stem)
    
    def _extract_enhanced_title_candidates(self, doc_structure: Any) -> List[Dict[str, Any]]:
        """Extract enhanced title candidates with scoring"""
        candidates = []
        
        try:
            # Extract text content
            content_text = self._extract_content_text(doc_structure)
            if not content_text:
                return candidates
            
            # Get first 20 lines for title analysis
            lines = content_text.strip().split('\n')[:20]
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or len(line) < 5:
                    continue
                
                # Score this line as a potential title
                score = self._score_title_candidate(line, i, lines)
                
                if score > 0:
                    candidates.append({
                        'text': line,
                        'score': score,
                        'position': i,
                        'length': len(line)
                    })
            
            # Sort by score descending
            candidates.sort(key=lambda x: x['score'], reverse=True)
            return candidates[:10]  # Top 10 candidates
        
        except Exception as e:
            logger.debug(f"Error extracting enhanced title candidates: {e}")
            return candidates
    
    def _extract_content_text(self, doc_structure: Any) -> str:
        """Extract plain text content from document structure"""
        try:
            if isinstance(doc_structure, dict):
                # Try different content keys
                for key in ['content', 'text', 'body', 'pages']:
                    if key in doc_structure:
                        content = doc_structure[key]
                        break
                else:
                    content = str(doc_structure)
                
                if isinstance(content, list):
                    # Join list items
                    text_parts = []
                    for item in content[:5]:  # First 5 pages/sections
                        if isinstance(item, dict):
                            text_parts.append(str(item.get('text', item.get('content', item))))
                        else:
                            text_parts.append(str(item))
                    return '\n'.join(text_parts)
                else:
                    return str(content)
            
            elif isinstance(doc_structure, str):
                return doc_structure
            else:
                return str(doc_structure)
        
        except Exception as e:
            logger.debug(f"Error extracting content text: {e}")
            return ""
    
    def _score_title_candidate(self, line: str, position: int, all_lines: List[str]) -> float:
        """Score a line as a potential title candidate"""
        score = 0.0
        line_lower = line.lower()
        
        # Position scoring (earlier lines more likely to be titles)
        if position == 0:
            score += 5.0
        elif position <= 2:
            score += 3.0
        elif position <= 5:
            score += 1.0
        
        # Length scoring (titles are usually 10-80 characters)
        length = len(line)
        if 15 <= length <= 60:
            score += 4.0
        elif 10 <= length <= 80:
            score += 2.0
        elif length > 80:
            score -= 2.0
        
        # Pattern matching
        for pattern in self.title_patterns['title_indicators']:
            if re.match(pattern, line, re.IGNORECASE):
                score += 3.0
                break
        
        # Exclude patterns (negative scoring)
        for pattern in self.title_patterns['exclude_patterns']:
            if re.match(pattern, line, re.IGNORECASE):
                score -= 5.0
        
        # Capitalization scoring
        if line.istitle():
            score += 2.0
        elif line.isupper() and length > 5:
            score += 1.0
        elif line[0].isupper():
            score += 0.5
        
        # Avoid body text indicators
        body_indicators = ['the ', 'this ', 'here ', 'in order', 'step ']
        if any(line_lower.startswith(indicator) for indicator in body_indicators):
            score -= 2.0
        
        # Avoid common non-title content
        if any(word in line_lower for word in ['page', 'copyright', 'all rights', 'published']):
            score -= 3.0
        
        return max(score, 0.0)
    
    def _select_best_title_candidate(self, candidates: List[Dict[str, Any]], filename: str) -> str:
        """Select the best title from candidates"""
        if not candidates:
            return self._generate_title_from_filename(filename)
        
        # Get highest scoring candidate
        best_candidate = candidates[0]
        
        # Validate the best candidate
        if best_candidate['score'] >= 5.0:
            title = best_candidate['text']
            # Clean and format the title
            title = self._clean_title(title)
            if len(title) >= 5:
                return title
        
        # If no good candidate, try to generate from filename
        return self._generate_title_from_filename(filename)
    
    def _clean_title(self, title: str) -> str:
        """Clean and format extracted title"""
        # Remove extra whitespace
        title = ' '.join(title.split())
        
        # Remove common prefixes/suffixes
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
    
    def _get_user_input(self) -> Dict[str, Any]:
        """Enhanced user input collection"""
        print("\n" + "="*60)
        print("DOCUMENT COLLECTION CONFIGURATION")
        print("="*60)
        
        # Get persona information
        print("\n1. PERSONA CONFIGURATION:")
        persona_role = input("Enter the persona/role (e.g., 'Analyst', 'Researcher', 'Specialist'): ").strip()
        if not persona_role:
            persona_role = "Document Analyst"
        
        # Optional: Get additional persona details
        persona_details = input("Additional persona context (optional): ").strip()
        
        # Get job/task information
        print("\n2. JOB TO BE DONE:")
        job_task = input("Enter the specific task (e.g., 'Analyze documents for key insights'): ").strip()
        if not job_task:
            job_task = "Analyze and work with the provided documents"
        
        # Optional: Get additional requirements
        additional_requirements = input("Additional requirements or constraints (optional): ").strip()
        
        # Get challenge information
        print("\n3. COLLECTION DETAILS:")
        challenge_id = input("Collection ID (default: 'collection_001'): ").strip() or "collection_001"
        
        return {
            "persona": {
                "role": persona_role,
                "details": persona_details if persona_details else None
            },
            "job_to_be_done": {
                "task": job_task,
                "requirements": additional_requirements if additional_requirements else None
            },
            "challenge_id": challenge_id
        }
    
    def _generate_enhanced_challenge_info(self, persona_role: str, job_task: str, pdf_files: List[Path]) -> Dict[str, str]:
        """Generate enhanced challenge info using improved analysis"""
        try:
            logger.info("Generating challenge information...")
            
            # Generate test case name based on task
            test_case_name = self._generate_test_case_name(job_task)
            
            # Generate comprehensive description
            description = self._generate_challenge_description(persona_role, job_task, len(pdf_files))
            
            return {
                "challenge_id": "collection_001",
                "test_case_name": test_case_name,
                "description": description
            }
            
        except Exception as e:
            logger.error(f"Error generating challenge info: {e}")
            return {
                "challenge_id": "collection_001", 
                "test_case_name": "document_analysis_task",
                "description": f"{persona_role} - {job_task}"
            }
    
    def _generate_test_case_name(self, job_task: str) -> str:
        """Generate a descriptive test case name from the task"""
        task_lower = job_task.lower()
        
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
            action_words.append('task')
        
        # Clean and join
        return '_'.join(action_words[:3]) if action_words else 'document_analysis_task'
    
    def _generate_challenge_description(self, persona_role: str, job_task: str, doc_count: int) -> str:
        """Generate comprehensive challenge description"""
        return f"{persona_role} working with {doc_count} documents - {job_task}"
    
    def _create_enhanced_challenge_structure(self, pdf_files: List[Path], user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create enhanced challenge structure with proper format"""
        
        # Get user input if not provided
        if not user_config:
            user_config = self._get_user_input()
        
        # Extract titles from PDFs with enhanced method
        documents = []
        logger.info("Extracting titles from PDF files with enhanced AI...")
        
        for pdf_path in pdf_files:
            title = self._extract_title_with_enhanced_ai(pdf_path)
            documents.append({
                "filename": pdf_path.name,
                "title": title
            })
            logger.info(f"✓ {pdf_path.name} -> '{title}'")
        
        # Generate enhanced challenge info
        persona_role = user_config["persona"]["role"]
        job_task = user_config["job_to_be_done"]["task"]
        challenge_info = self._generate_enhanced_challenge_info(persona_role, job_task, pdf_files)
        
        # Add user-provided challenge_id if available
        if user_config.get("challenge_id"):
            challenge_info["challenge_id"] = user_config["challenge_id"]
        
        # Build the complete structure
        structure = {
            "challenge_info": challenge_info,
            "documents": documents,
            "persona": user_config["persona"],
            "job_to_be_done": user_config["job_to_be_done"]
        }
        
        return structure
    
    def process_single_pdf(self, pdf_path: Path, user_config: Optional[Dict[str, Any]] = None) -> bool:
        """Process a single PDF file with enhanced extraction"""
        try:
            logger.info(f"Processing with enhanced extraction: {pdf_path.name}")
            
            # Extract and save structured text
            doc_structure = self.pdf_proc.extract_text_with_layout(pdf_path)
            if not doc_structure:
                logger.error(f"Failed to extract text from {pdf_path.name}")
                return False
            
            # Create the target JSON structure for this single document
            single_doc_structure = self._create_enhanced_challenge_structure([pdf_path], user_config)
            
            # Add the extracted content to the structure
            single_doc_structure["document_content"] = doc_structure
            
            json_path = self.input_dir / f"{pdf_path.stem}.json"
            self.writer.save_json(single_doc_structure, json_path)
            
            logger.info(f"✓ Enhanced processing complete for {pdf_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Error in enhanced processing {pdf_path.name}: {e}")
            return False
    
    def create_master_json(self, pdf_files: List[Path], user_config: Optional[Dict[str, Any]] = None) -> bool:
        """Create enhanced master JSON file"""
        try:
            master_structure = self._create_enhanced_challenge_structure(pdf_files, user_config)
            master_json_path = self.input_dir / "master_collection.json"
            
            self.writer.save_json(master_structure, master_json_path)
            logger.info(f"✓ Created enhanced master collection JSON with {len(pdf_files)} documents")
            
            # Print the enhanced configuration
            self._display_configuration_summary(master_structure)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating enhanced master JSON: {e}")
            return False
    
    def _display_configuration_summary(self, structure: Dict[str, Any]):
        """Display comprehensive configuration summary"""
        print(f"\n{'='*70}")
        print("CONFIGURATION SUMMARY")
        print(f"{'='*70}")
        
        challenge_info = structure.get('challenge_info', {})
        persona = structure.get('persona', {})
        job = structure.get('job_to_be_done', {})
        documents = structure.get('documents', [])
        
        print(f"Collection ID: {challenge_info.get('challenge_id', 'N/A')}")
        print(f"Test Case: {challenge_info.get('test_case_name', 'N/A')}")
        print(f"Description: {challenge_info.get('description', 'N/A')}")
        print(f"\nPersona Role: {persona.get('role', 'N/A')}")
        
        if persona.get('details'):
            print(f"Persona Details: {persona['details']}")
        
        print(f"\nPrimary Task: {job.get('task', 'N/A')}")
        
        if job.get('requirements'):
            print(f"Requirements: {job['requirements']}")
        
        print(f"\nDocuments ({len(documents)}):")
        for i, doc in enumerate(documents, 1):
            print(f"  {i}. {doc['filename']}")
            print(f"     Title: {doc['title']}")
        
        print(f"{'='*70}\n")
    
    def build_enhanced_vector_index(self) -> bool:
        """Build enhanced vector index with better chunking"""
        try:
            json_files = list(self.input_dir.glob("*.json"))
            if not json_files:
                logger.error("No JSON files found to build index")
                return False
            
            logger.info(f"Building enhanced vector index from {len(json_files)} documents")
            
            # Process all documents with enhanced chunking
            all_chunks = []
            for json_path in json_files:
                try:
                    doc_data = self.writer.load_json(json_path)
                    
                    # Handle the enhanced structure
                    if "document_content" in doc_data:
                        doc_structure = doc_data["document_content"]
                        
                        # Add document metadata for better retrieval
                        doc_metadata = {
                            'filename': json_path.stem,
                            'title': self._get_document_title(doc_data),
                            'persona': doc_data.get('persona', {}).get('role', ''),
                            'task': doc_data.get('job_to_be_done', {}).get('task', '')
                        }
                        
                        chunks = self.text_chunker.chunk_document(doc_structure, metadata=doc_metadata)
                        all_chunks.extend(chunks)
                    else:
                        # Fallback for old format
                        chunks = self.text_chunker.chunk_document(doc_data)
                        all_chunks.extend(chunks)
                        
                except Exception as e:
                    logger.error(f"Error processing {json_path.name}: {e}")
                    continue
            
            if not all_chunks:
                logger.error("No chunks created from documents")
                return False
            
            logger.info(f"Embedding {len(all_chunks)} chunks with enhanced metadata...")
            
            # Generate embeddings and build FAISS index
            self.vector_embedder.embed_chunks(all_chunks)
            self.vector_embedder.save_index(self.db_path)
            
            logger.info(f"✓ Enhanced vector index saved with {len(all_chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error building enhanced vector index: {e}")
            return False
    
    def _get_document_title(self, doc_data: Dict[str, Any]) -> str:
        """Extract document title from structure"""
        documents = doc_data.get('documents', [])
        if documents and len(documents) > 0:
            return documents[0].get('title', 'Unknown Document')
        return 'Unknown Document'
    
    def run_enhanced_pipeline(self, single_file: Optional[Path] = None, rebuild_only: bool = False) -> bool:
        """Run the enhanced ingestion pipeline"""
        if rebuild_only:
            logger.info("Rebuilding enhanced vector index only")
            return self.build_enhanced_vector_index()
        
        if single_file:
            logger.info(f"Processing single file with enhanced pipeline: {single_file}")
            if not self.process_single_pdf(single_file):
                return False
            return self.build_enhanced_vector_index()
        
        # Full enhanced pipeline
        logger.info("Starting enhanced ingestion pipeline")
        pdf_files = self.find_pdfs()
        
        if not pdf_files:
            logger.error("No PDF files found")
            return False
        
        # Create enhanced master JSON
        if not self.create_master_json(pdf_files):
            logger.error("Failed to create enhanced master JSON")
            return False
        
        # Process all PDFs with enhanced extraction
        success_count = sum(1 for pdf in pdf_files if self.process_single_pdf(pdf))
        
        logger.info(f"Enhanced processing: {success_count}/{len(pdf_files)} PDFs")
        
        if success_count == 0:
            logger.error("No PDFs were successfully processed")
            return False
        
        # Build enhanced vector index
        if not self.build_enhanced_vector_index():
            logger.error("Failed to build enhanced vector index")
            return False
        
        logger.info("✓ Enhanced ingestion pipeline completed successfully!")
        return True

def main():
    parser = argparse.ArgumentParser(description="Enhanced PDF Ingestion Pipeline")
    parser.add_argument("--collection-dir", type=Path, default=Path("data/collection"),
                       help="Directory containing PDF files")
    parser.add_argument("--input-dir", type=Path, default=Path("data/input"),
                       help="Directory to save processed JSON files")
    parser.add_argument("--db-path", type=Path, default=Path("db/faiss_store.pkl"),
                       help="Path to save FAISS index")
    parser.add_argument("--model-path", type=str, default="models/all-MiniLM-L6-v2",
                       help="Path to embedding model")
    parser.add_argument("--single-file", type=Path,
                       help="Process only a single PDF file")
    parser.add_argument("--rebuild-index", action="store_true",
                       help="Only rebuild the vector index from existing JSONs")
    parser.add_argument("--challenge-config", type=Path,
                       help="Path to JSON file with challenge configuration")
    parser.add_argument("--auto-config", action="store_true",
                       help="Auto-generate configuration from document analysis")
    
    args = parser.parse_args()
    
    try:
        # Load challenge configuration if provided
        challenge_config = {}
        if args.challenge_config and args.challenge_config.exists():
            with open(args.challenge_config, 'r') as f:
                challenge_config = json.load(f)
        
        pipeline = EnhancedIngestionPipeline(
            collection_dir=args.collection_dir,
            input_dir=args.input_dir,
            db_path=args.db_path,
            model_path=args.model_path,
            challenge_config=challenge_config
        )
        
        success = pipeline.run_enhanced_pipeline(
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