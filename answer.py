import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import sys
import json
from datetime import datetime

from processors import (
    pdf_processor, chunker, embedder, retriever, 
    importance_ranker, output_writer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnswerPipeline:
    """
    Answer pipeline that processes PDFs, creates input JSON, then generates output JSON
    Following the workflow: PDFs → input JSON → output JSON
    """
    
    def __init__(self,
                 collection_dir: Path = Path("data/collection"),
                 input_dir: Path = Path("data/input"),
                 output_dir: Path = Path("data/output"),
                 db_path: Path = Path("db/faiss_store.pkl"),
                 model_path: str = "models/all-MiniLM-L6-v2",
                 retrieval_k: int = 8):
        
        self.collection_dir = collection_dir
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.db_path = db_path
        self.model_path = model_path
        self.retrieval_k = retrieval_k
        
        # Ensure directories exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processors
        self.pdf_proc = pdf_processor()
        self.text_chunker = chunker(chunk_size=512, overlap=50)
        self.vector_embedder = embedder(model_path)
        self.writer = output_writer()
        
        # Initialize retrieval components (will be set up after embedding)
        self.doc_retriever = None
        self.imp_ranker = None
    
    def get_user_input(self) -> Dict[str, Any]:
        """Get persona and job to be done from user"""
        print("\n" + "="*60)
        print("ANSWER PIPELINE CONFIGURATION")
        print("="*60)
        
        # Get persona information
        print("\n1. PERSONA CONFIGURATION:")
        persona_role = input("Enter the persona/role (e.g., 'Data Analyst', 'Research Scientist'): ").strip()
        if not persona_role:
            persona_role = "Document Analyst"
        
        # Get job/task information
        print("\n2. JOB TO BE DONE:")
        job_task = input("Enter the specific task (e.g., 'Extract key insights from research papers'): ").strip()
        if not job_task:
            job_task = "Analyze and extract information from documents"
        
        # Get challenge information
        print("\n3. CHALLENGE DETAILS:")
        challenge_id = input("Challenge ID (default: 'round_1b_001'): ").strip() or "round_1b_001"
        test_case_name = input("Test case name (optional): ").strip() or self._generate_test_case_name(job_task)
        
        return {
            "persona": {"role": persona_role},
            "job_to_be_done": {"task": job_task},
            "challenge_id": challenge_id,
            "test_case_name": test_case_name
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
        
        return '_'.join(action_words[:3]) if action_words else 'document_analysis_task'
    
    def find_pdfs(self) -> List[Path]:
        """Find all PDF files in collection directory"""
        if not self.collection_dir.exists():
            logger.error(f"Collection directory does not exist: {self.collection_dir}")
            return []
        
        pdf_files = list(self.collection_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        return pdf_files
    
    def process_pdfs_and_create_input(self, pdf_files: List[Path], user_config: Dict[str, Any]) -> Path:
        """Step 1: Process PDFs using processors.py and create input JSON"""
        logger.info("Step 1: Processing PDFs and creating input JSON...")
        
        # Process PDFs using the corrected processor
        processed_docs = self.pdf_proc.process_pdf_collection(pdf_files)
        
        if not processed_docs:
            raise ValueError("No PDFs were successfully processed")
        
        # Create input JSON structure as specified in requirements
        input_structure = self.writer.create_input_json(
            challenge_id=user_config["challenge_id"],
            test_case_name=user_config["test_case_name"],
            documents=processed_docs,
            persona=user_config["persona"],
            job_to_be_done=user_config["job_to_be_done"]
        )
        
        # Save input JSON to data/input directory
        input_json_path = self.input_dir / f"{user_config['challenge_id']}_input.json"
        self.writer.save_json(input_structure, input_json_path)
        
        logger.info(f"✓ Input JSON created: {input_json_path}")
        return input_json_path
    
    def setup_retrieval_system(self, processed_docs: List, persona: str, job_to_be_done: str) -> None:
        """Setup the retrieval and ranking system"""
        logger.info("Setting up retrieval system...")
        
        # Collect all chunks from processed documents
        all_chunks = []
        for doc in processed_docs:
            all_chunks.extend(doc.chunks)
        
        if not all_chunks:
            raise ValueError("No chunks available for retrieval system")
        
        # Create embeddings and build FAISS index
        logger.info(f"Creating embeddings for {len(all_chunks)} chunks...")
        self.vector_embedder.embed_chunks(all_chunks)
        
        # Save the index
        self.vector_embedder.save_index(self.db_path)
        
        # Initialize retrieval components
        self.doc_retriever = retriever(self.vector_embedder)
        self.imp_ranker = importance_ranker(self.vector_embedder)
        
        logger.info("✓ Retrieval system setup complete")
    
    def generate_challenge_info(self, persona: str, job_task: str, processed_docs: List) -> Dict[str, Any]:
        """Generate challenge information from processed documents"""
        logger.info("Generating challenge information...")
        
        # Create challenge description
        doc_count = len(processed_docs)
        doc_titles = [doc.title for doc in processed_docs[:3]]  # First 3 titles
        
        challenge_desc = f"{persona} analyzing {doc_count} documents including: {', '.join(doc_titles)}"
        if doc_count > 3:
            challenge_desc += f" and {doc_count - 3} more documents"
        
        challenge_desc += f" to {job_task.lower()}"
        
        return {
            "description": challenge_desc,
            "document_count": doc_count,
            "primary_task": job_task
        }
    
    def extract_sections_by_importance(self, persona: str, job_to_be_done: str, 
                                     processed_docs: List, top_k: int = 10) -> List[Dict[str, Any]]:
        """Extract and rank sections by importance for persona and job"""
        logger.info(f"Extracting top {top_k} sections by importance...")
        
        # Collect all chunks
        all_chunks = []
        for doc in processed_docs:
            all_chunks.extend(doc.chunks)
        
        # Rank sections by importance
        ranked_chunks = self.imp_ranker.rank_sections(all_chunks, persona, job_to_be_done, top_k)
        
        # Convert to required format
        extracted_sections = []
        for chunk in ranked_chunks:
            section = {
                "document": chunk.doc_id + ".pdf",
                "section_title": chunk.section_title or self._extract_section_title(chunk.text),
                "importance_rank": chunk.importance_rank,
                "page_number": chunk.page_num
            }
            extracted_sections.append(section)
        
        logger.info(f"✓ Extracted {len(extracted_sections)} important sections")
        return extracted_sections
    
    def _extract_section_title(self, text: str) -> str:
        """Extract a section title from chunk text"""
        # Take first meaningful line as section title
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 5 and len(line) < 100:
                return line
        
        # Fallback: first 50 characters
        return text[:50].strip() + "..." if len(text) > 50 else text.strip()
    
    def perform_subsection_analysis(self, persona: str, job_to_be_done: str, 
                                  extracted_sections: List[Dict], processed_docs: List) -> List[Dict[str, Any]]:
        """Perform detailed analysis of subsections"""
        logger.info("Performing subsection analysis...")
        
        subsection_analysis = []
        
        # Create analysis query based on persona and job
        analysis_query = f"As a {persona}, analyze these documents to {job_to_be_done.lower()}"
        
        # Retrieve relevant chunks for analysis
        retrieval_results = self.doc_retriever.retrieve_for_persona(
            analysis_query, persona, job_to_be_done, top_k=self.retrieval_k
        )
        
        # Process each retrieval result
        for result in retrieval_results:
            chunk = result.chunk
            
            # Generate refined analysis text
            refined_text = self._generate_refined_analysis(chunk.text, persona, job_to_be_done)
            
            analysis = {
                "document": chunk.doc_id + ".pdf",
                "section_title": chunk.section_title or self._extract_section_title(chunk.text),
                "refined_text": refined_text,
                "page_number": chunk.page_num,
                "relevance_score": result.score,
                "chunk_type": chunk.chunk_type
            }
            subsection_analysis.append(analysis)
        
        logger.info(f"✓ Completed analysis of {len(subsection_analysis)} subsections")
        return subsection_analysis
    
    def _generate_refined_analysis(self, text: str, persona: str, job_to_be_done: str) -> str:
        """Generate refined analysis text for a chunk"""
        # Simple analysis generation (can be enhanced with LLM)
        analysis_prefix = f"From a {persona} perspective for {job_to_be_done.lower()}: "
        
        # Extract key points from text
        sentences = text.split('.')
        key_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20][:3]
        
        if key_sentences:
            refined = analysis_prefix + '. '.join(key_sentences) + '.'
        else:
            refined = analysis_prefix + text[:200] + "..." if len(text) > 200 else text
        
        return refined
    
    def create_output_json(self, input_json_path: Path) -> Path:
        """Step 2: Create output JSON from input JSON using processing pipeline"""
        logger.info("Step 2: Creating output JSON from input JSON...")
        
        # Load input JSON
        input_data = self.writer.load_json(input_json_path)
        
        # Extract information from input
        persona_role = input_data["persona"]["role"]
        job_task = input_data["job_to_be_done"]["task"]
        challenge_id = input_data["challenge_info"]["challenge_id"]
        
        # Find and process PDFs
        pdf_files = self.find_pdfs()
        if not pdf_files:
            raise ValueError("No PDF files found for processing")
        
        # Process PDFs again to get detailed chunks
        processed_docs = self.pdf_proc.process_pdf_collection(pdf_files)
        
        # Setup retrieval system
        self.setup_retrieval_system(processed_docs, persona_role, job_task)
        
        # Generate challenge information
        challenge_info = self.generate_challenge_info(persona_role, job_task, processed_docs)
        
        # Extract important sections
        extracted_sections = self.extract_sections_by_importance(
            persona_role, job_task, processed_docs, top_k=10
        )
        
        # Perform subsection analysis
        subsection_analysis = self.perform_subsection_analysis(
            persona_role, job_task, extracted_sections, processed_docs
        )
        
        # Create output JSON structure as specified in requirements
        input_documents = [doc["filename"] for doc in input_data["documents"]]
        
        output_structure = self.writer.create_output_json(
            input_documents=input_documents,
            persona=persona_role,
            job_to_be_done=job_task,
            extracted_sections=extracted_sections,
            subsection_analysis=subsection_analysis
        )
        
        # Add additional metadata
        output_structure["metadata"].update({
            "challenge_id": challenge_id,
            "processing_timestamp": datetime.now().isoformat(),
            "total_chunks_processed": sum(len(doc.chunks) for doc in processed_docs),
            "retrieval_k": self.retrieval_k,
            "model_path": self.model_path
        })
        
        # Save output JSON to data/output directory
        output_json_path = self.output_dir / f"{challenge_id}_output.json"
        self.writer.save_json(output_structure, output_json_path)
        
        logger.info(f"✓ Output JSON created: {output_json_path}")
        return output_json_path
    
    def load_existing_input_json(self, input_path: Path) -> Dict[str, Any]:
        """Load existing input JSON file"""
        if not input_path.exists():
            raise FileNotFoundError(f"Input JSON not found: {input_path}")
        
        input_data = self.writer.load_json(input_path)
        logger.info(f"✓ Loaded input JSON: {input_path}")
        return input_data
    
    def display_processing_summary(self, input_path: Path, output_path: Path):
        """Display processing summary"""
        input_data = self.writer.load_json(input_path)
        output_data = self.writer.load_json(output_path)
        
        print(f"\n{'='*70}")
        print("PROCESSING SUMMARY")
        print(f"{'='*70}")
        
        # Input info
        persona = input_data["persona"]["role"]
        task = input_data["job_to_be_done"]["task"]
        challenge_id = input_data["challenge_info"]["challenge_id"]
        input_docs = len(input_data["documents"])
        
        print(f"Challenge ID: {challenge_id}")
        print(f"Persona: {persona}")
        print(f"Task: {task}")
        print(f"Input Documents: {input_docs}")
        
        # Output info
        sections = len(output_data.get("extracted_sections", []))
        analyses = len(output_data.get("subsection_analysis", []))
        total_chunks = output_data["metadata"].get("total_chunks_processed", 0)
        
        print(f"\nOutput Results:")
        print(f"  Extracted Sections: {sections}")
        print(f"  Subsection Analyses: {analyses}")
        print(f"  Total Chunks Processed: {total_chunks}")
        
        print(f"\nFiles:")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_path}")
        
        # Top sections
        if output_data.get("extracted_sections"):
            print(f"\nTop Extracted Sections:")
            for i, section in enumerate(output_data["extracted_sections"][:3], 1):
                print(f"  {i}. {section['section_title'][:60]}...")
                print(f"     Document: {section['document']}, Page: {section['page_number']}")
        
        print(f"{'='*70}\n")
    
    def run_interactive_mode(self):
        """Run in interactive mode"""
        # print("\n" + "="*70)
        # print("ANSWER PIPELINE - INTERACTIVE MODE")
        # print("="*70)
        # print("Commands:")
        # print("  'process' - Process PDFs and create both input and output JSON")
        # print("  'input-only' - Process PDFs and create input JSON only")
        # print("  'output <input_file>' - Create output JSON from existing input JSON")
        # print("  'batch' - Process all JSON files in input directory")
        # print("  'quit' - Exit")
        # print("="*70)
        
        while True:
            try:
                command = 'batch'
                
                if command.lower() in ['quit', 'exit', 'q']:
                    break
                
                elif command == 'process':
                    # Full pipeline: PDFs → input JSON → output JSON
                    user_config = self.get_user_input()
                    pdf_files = self.find_pdfs()
                    
                    if not pdf_files:
                        print("✗ No PDF files found")
                        continue
                    
                    # Step 1: Create input JSON
                    input_path = self.process_pdfs_and_create_input(pdf_files, user_config)
                    print(f"✓ Step 1 complete: {input_path}")
                    
                    # Step 2: Create output JSON
                    output_path = self.create_output_json(input_path)
                    print(f"✓ Step 2 complete: {output_path}")
                    
                    # Display summary
                    self.display_processing_summary(input_path, output_path)
                
                elif command == 'input-only':
                    # Create input JSON only
                    user_config = self.get_user_input()
                    pdf_files = self.find_pdfs()
                    
                    if not pdf_files:
                        print("✗ No PDF files found")
                        continue
                    
                    input_path = self.process_pdfs_and_create_input(pdf_files, user_config)
                    print(f"✓ Input JSON created: {input_path}")
                
                elif command.startswith('output '):
                    # Create output JSON from existing input JSON
                    input_filename = command[7:].strip()
                    input_path = self.input_dir / input_filename
                    
                    if not input_path.exists():
                        print(f"✗ Input file not found: {input_path}")
                        continue
                    
                    try:
                        output_path = self.create_output_json(input_path)
                        print(f"✓ Output JSON created: {output_path}")
                        self.display_processing_summary(input_path, output_path)
                    except Exception as e:
                        print(f"✗ Error creating output: {e}")
                
                elif command == 'batch':
                    # Process all JSON files in input directory
                    json_files = list(self.input_dir.glob("*_input.json"))
                    if not json_files:
                        print("✗ No input JSON files found")
                        continue
                    
                    print(f"Processing {len(json_files)} input files...")
                    success_count = 0
                    
                    for json_file in json_files:
                        try:
                            output_path = self.create_output_json(json_file)
                            print(f"✓ {json_file.name} → {output_path.name}")
                            success_count += 1
                        except Exception as e:
                            print(f"✗ Error processing {json_file.name}: {e}")
                    
                    print(f"\nBatch processing complete: {success_count}/{len(json_files)} successful")
                
                else:
                    print("Unknown command. Use 'process', 'input-only', 'output <file>', 'batch', or 'quit'.")
            
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def run_pipeline(self, config_path: Optional[Path] = None, input_file: Optional[str] = None, 
                    output_only: bool = False) -> bool:
        """Run the complete answer pipeline"""
        try:
            if output_only and input_file:
                # Create output JSON from existing input JSON
                input_path = self.input_dir / input_file
                output_path = self.create_output_json(input_path)
                self.display_processing_summary(input_path, output_path)
                return True
            
            # Get configuration
            if config_path and config_path.exists():
                user_config = self.writer.load_json(config_path)
            else:
                user_config = self.get_user_input()
            
            # Find PDFs
            pdf_files = self.find_pdfs()
            if not pdf_files:
                logger.error("No PDF files found")
                return False
            
            # Step 1: Process PDFs and create input JSON
            input_path = self.process_pdfs_and_create_input(pdf_files, user_config)
            
            # Step 2: Create output JSON from input JSON
            output_path = self.create_output_json(input_path)
            
            # Display summary
            self.display_processing_summary(input_path, output_path)
            
            logger.info("✓ Answer pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Answer Pipeline: PDFs → Input JSON → Output JSON")
    parser.add_argument("--config", type=Path, help="Path to configuration JSON")
    parser.add_argument("--config-dir", type=Path, help="Directory containing multiple configuration files")
    parser.add_argument("--task", type=str, help="Custom task description")
    parser.add_argument("--db-path", type=Path, default=Path("db/faiss_store.pkl"),
                       help="Path to FAISS index")
    parser.add_argument("--input-dir", type=Path, default=Path("data/input"),
                       help="Directory for input JSON files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/output"),
                       help="Directory for output JSON files")
    parser.add_argument("--model-path", type=str, default="models/all-MiniLM-L6-v2",
                       help="Path to embedding model")
    parser.add_argument("--retrieval-k", type=int, default=8,
                       help="Number of chunks to retrieve")
    parser.add_argument("--output-file", type=str,
                       help="Custom output filename (without extension)")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--batch", action="store_true",
                       help="Process all input JSON files")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save the result to file")
    parser.add_argument("--input-file", type=str,
                       help="Specific input JSON file to process")
    parser.add_argument("--output-only", action="store_true",
                       help="Only create output JSON from existing input JSON")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not any([args.interactive, args.config, args.config_dir, args.batch, args.input_file]):
        parser.error("Must specify one of: --interactive, --config, --config-dir, --batch, or --input-file")
    
    try:
        pipeline = AnswerPipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            db_path=args.db_path,
            model_path=args.model_path,
            retrieval_k=args.retrieval_k
        )
        
        if args.interactive:
            pipeline.run_interactive_mode()
        else:
            success = pipeline.run_pipeline(
                config_path=args.config,
                input_file=args.input_file,
                output_only=args.output_only
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