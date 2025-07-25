import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import json
from datetime import datetime

# Import the generalized RAG flow
from langgraph_flow import create_rag_flow
from processors import output_writer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GeneralizedAnswerPipeline:
    """Generalized answer generation pipeline that works with any documents and personas"""
    
    def __init__(self,
                 db_path: Path = Path("db/faiss_store.pkl"),
                 input_dir: Path = Path("data/input"),
                 output_dir: Path = Path("data/output"),
                 model_path: str = "models/tinyllama",
                 retrieval_k: int = 8):
        
        self.db_path = db_path
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_path = model_path
        self.retrieval_k = retrieval_k
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.writer = output_writer()
        self.rag_flow = self._initialize_rag_flow()
    
    def _initialize_rag_flow(self):
        """Initialize the generalized RAG flow"""
        try:
            logger.info("Initializing generalized persona-based RAG flow...")
            rag_flow = create_rag_flow(
                db_path=self.db_path,
                model_path=self.model_path,
                retrieval_k=self.retrieval_k
            )
            logger.info("✓ Generalized RAG flow initialized")
            return rag_flow
        except Exception as e:
            logger.error(f"Failed to initialize RAG flow: {e}")
            raise
    
    def load_challenge_config(self, config_path: Path) -> Dict[str, Any]:
        """Load challenge configuration from JSON file"""
        if not config_path.exists():
            raise FileNotFoundError(f"Challenge config not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Validate required fields
            self._validate_config(config)
            
            logger.info(f"✓ Loaded challenge config from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load challenge config: {e}")
            raise
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate that the configuration has required fields"""
        required_top_level = ['documents', 'persona', 'job_to_be_done']
        for field in required_top_level:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate persona has role
        if 'role' not in config['persona']:
            raise ValueError("Persona must have a 'role' field")
        
        # Validate job_to_be_done has task
        if 'task' not in config['job_to_be_done']:
            raise ValueError("job_to_be_done must have a 'task' field")
        
        # Validate documents is not empty
        if not config['documents']:
            raise ValueError("Documents list cannot be empty")
    
    def execute_job_to_be_done(self, input_data: Dict[str, Any], custom_task: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the job_to_be_done task using persona-based analysis
        
        Args:
            input_data: Dictionary containing challenge_info, documents, persona, and job_to_be_done
            custom_task: Optional custom task to override the one in input_data
        
        Returns:
            Dictionary with analysis results in the specified format
        """
        try:
            # Use custom task if provided
            if custom_task:
                input_data = input_data.copy()
                input_data['job_to_be_done'] = {'task': custom_task}
            
            task = input_data['job_to_be_done']['task']
            persona_role = input_data['persona']['role']
            document_count = len(input_data.get('documents', []))
            
            logger.info(f"Executing job as {persona_role}: {task}")
            logger.info(f"Working with {document_count} document(s)")
            
            # Process using the generalized RAG flow
            result = self.rag_flow.process_with_persona(input_data)
            
            # Enhance with job execution metadata
            result.update({
                "job_execution_info": {
                    "original_task": task,
                    "persona_role": persona_role,
                    "model_path": self.model_path,
                    "retrieval_k": self.retrieval_k,
                    "processing_type": "job_to_be_done_execution",
                    "execution_timestamp": datetime.now().isoformat()
                }
            })
            
            # Log execution results
            if result['success']:
                sections_found = len(result.get('extracted_sections', []))
                analyses_created = len(result.get('subsection_analysis', []))
                logger.info(f"✓ Job executed successfully - {sections_found} sections, {analyses_created} analyses")
            else:
                logger.error(f"✗ Job execution failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing job_to_be_done: {e}")
            return self._create_error_result(input_data, str(e))
    
    def _create_error_result(self, input_data: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            "metadata": {
                "input_documents": [doc.get('filename', 'unknown') for doc in input_data.get('documents', [])],
                "persona": input_data.get('persona', {}).get('role', ''),
                "job_to_be_done": input_data.get('job_to_be_done', {}).get('task', ''),
                "error_details": error
            },
            "extracted_sections": [],
            "subsection_analysis": [],
            "final_answer": "",
            "job_execution_info": {
                "processing_type": "job_to_be_done_execution",
                "execution_timestamp": datetime.now().isoformat()
            },
            "error": error,
            "success": False,
            "timestamp": datetime.now().isoformat()
        }
    
    def save_job_result(self, result: Dict[str, Any], custom_filename: Optional[str] = None) -> Path:
        """Save job execution result in the specified output format"""
        try:
            if custom_filename:
                filename = f"{custom_filename}.json"
            else:
                # Create filename from job execution info
                persona_role = result.get('metadata', {}).get('persona', 'unknown').replace(' ', '_').lower()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                challenge_id = result.get('metadata', {}).get('challenge_id', 'general')
                filename = f"job_result_{challenge_id}_{persona_role}_{timestamp}.json"
            
            output_path = self.output_dir / filename
            
            # Format output according to specifications
            formatted_output = {
                "metadata": result.get('metadata', {}),
                "extracted_sections": result.get('extracted_sections', []),
                "subsection_analysis": result.get('subsection_analysis', []),
                "final_answer": result.get('final_answer', ''),
                "job_execution_info": result.get('job_execution_info', {}),
                "timestamp": result.get('timestamp', datetime.now().isoformat())
            }
            
            # Include error info if there was an error
            if not result.get('success', True):
                formatted_output['error'] = result.get('error', '')
                formatted_output['success'] = False
            else:
                formatted_output['success'] = True
            
            # Save using the writer
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(formatted_output, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Job result saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving job result: {e}")
            raise
    
    def run_batch_jobs(self, config_dir: Path) -> List[Dict[str, Any]]:
        """Execute jobs for all JSON configs in directory"""
        json_files = list(config_dir.glob("*.json"))
        if not json_files:
            logger.warning(f"No JSON config files found in {config_dir}")
            return []
        
        results = []
        logger.info(f"Executing batch jobs for {len(json_files)} configurations")
        
        for json_file in json_files:
            try:
                logger.info(f"Processing job from {json_file.name}...")
                
                # Load configuration
                config = self.load_challenge_config(json_file)
                
                # Execute the job
                result = self.execute_job_to_be_done(config)
                
                # Save result
                output_filename = f"job_result_{json_file.stem}"
                output_path = self.save_job_result(result, output_filename)
                
                # Collect summary info
                job_summary = {
                    'config_file': json_file.name,
                    'output_file': output_path.name,
                    'success': result['success'],
                    'persona_role': result.get('metadata', {}).get('persona', 'unknown'),
                    'task': result.get('metadata', {}).get('job_to_be_done', 'unknown')[:100],
                    'sections_found': len(result.get('extracted_sections', [])),
                    'subsections_analyzed': len(result.get('subsection_analysis', [])),
                    'has_final_answer': bool(result.get('final_answer', '').strip())
                }
                
                if not result['success']:
                    job_summary['error'] = result.get('error', 'Unknown error')
                
                results.append(job_summary)
                
                logger.info(f"✓ Job completed for {json_file.name} - {len(result.get('extracted_sections', []))} sections")
                
            except Exception as e:
                logger.error(f"✗ Error processing job from {json_file.name}: {e}")
                results.append({
                    'config_file': json_file.name,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def run_interactive_job_mode(self):
        """Run in interactive job execution mode"""
        print("\n" + "="*70)
        print("INTERACTIVE JOB-TO-BE-DONE EXECUTION MODE")
        print("="*70)
        print("Commands:")
        print("  'load <file>' - Load challenge configuration")
        print("  'task <description>' - Execute custom task with loaded config")
        print("  'execute' - Execute the original job_to_be_done task")
        print("  'quit' - Exit")
        print("="*70)
        
        current_config = None
        
        while True:
            try:
                command = input("\n> ").strip()
                
                if command.lower() in ['quit', 'exit', 'q']:
                    break
                
                if command.startswith('load '):
                    config_path = Path(command[5:].strip())
                    if not config_path.is_absolute():
                        config_path = self.input_dir / config_path
                    
                    try:
                        current_config = self.load_challenge_config(config_path)
                        persona = current_config['persona']['role']
                        task = current_config['job_to_be_done']['task']
                        docs = len(current_config['documents'])
                        
                        print(f"✓ Loaded configuration:")
                        print(f"  Persona: {persona}")
                        print(f"  Original Task: {task}")
                        print(f"  Documents: {docs}")
                        
                    except Exception as e:
                        print(f"✗ Error loading config: {e}")
                        continue
                
                elif command.startswith('task '):
                    if not current_config:
                        print("✗ No configuration loaded. Use 'load <file>' first.")
                        continue
                    
                    custom_task = command[5:].strip()
                    if not custom_task:
                        print("✗ Please provide a task description.")
                        continue
                    
                    print(f"Executing custom task: {custom_task}")
                    result = self.execute_job_to_be_done(current_config, custom_task)
                    self._display_job_result(result)
                    
                    # Save option
                    if input("\nSave result? (y/n): ").strip().lower() in ['y', 'yes']:
                        output_path = self.save_job_result(result)
                        print(f"✓ Saved to: {output_path}")
                
                elif command == 'execute':
                    if not current_config:
                        print("✗ No configuration loaded. Use 'load <file>' first.")
                        continue
                    
                    print("Executing original job_to_be_done task...")
                    result = self.execute_job_to_be_done(current_config)
                    self._display_job_result(result)
                    
                    if input("\nSave result? (y/n): ").strip().lower() in ['y', 'yes']:
                        output_path = self.save_job_result(result)
                        print(f"✓ Saved to: {output_path}")
                
                elif not command:
                    if current_config:
                        print("Use 'execute' to run the original task or 'task <description>' for a custom task.")
                    else:
                        print("No configuration loaded. Use 'load <file>' to start.")
                
                else:
                    print("Unknown command. Use 'load <file>', 'task <description>', 'execute', or 'quit'.")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _display_job_result(self, result: Dict[str, Any]):
        """Display job execution result"""
        print("\n" + "="*80)
        print("JOB-TO-BE-DONE EXECUTION RESULT")
        print("="*80)
        
        metadata = result.get('metadata', {})
        job_info = result.get('job_execution_info', {})
        
        print(f"Persona Role: {metadata.get('persona', 'N/A')}")
        print(f"Task: {metadata.get('job_to_be_done', 'N/A')}")
        print(f"Documents Processed: {len(metadata.get('input_documents', []))}")
        print(f"Execution Status: {'✓ SUCCESS' if result.get('success', False) else '✗ FAILED'}")
        
        if not result.get('success', False):
            print(f"\n✗ ERROR: {result.get('error', 'Unknown error')}")
            print("="*80)
            return
        
        # Display job execution summary
        sections = result.get('extracted_sections', [])
        subsections = result.get('subsection_analysis', [])
        final_answer = result.get('final_answer', '')
        
        print(f"\n📊 EXECUTION SUMMARY:")
        print("-" * 50)
        print(f"Sections Extracted: {len(sections)}")
        print(f"Subsections Analyzed: {len(subsections)}")
        print(f"Final Answer Generated: {'Yes' if final_answer.strip() else 'No'}")
        
        # Display top extracted sections
        if sections:
            print(f"\n📄 TOP EXTRACTED SECTIONS:")
            print("-" * 50)
            for i, section in enumerate(sections[:5], 1):
                print(f"{i}. {section['section_title']}")
                print(f"   📁 Document: {section['document']}")
                print(f"   📄 Page: {section['page_number']}")
                print(f"   ⭐ Importance: {section['importance_rank']}")
                print()
            
            if len(sections) > 5:
                print(f"   ... and {len(sections) - 5} more sections")
        
        # Display key subsection analyses
        if subsections:
            print(f"\n🔍 KEY SUBSECTION ANALYSES:")
            print("-" * 50)
            for i, sub in enumerate(subsections[:3], 1):
                print(f"{i}. {sub['section_title']}")
                refined = sub['refined_text']
                print(f"   Analysis: {refined[:150]}{'...' if len(refined) > 150 else ''}")
                if 'relevance_score' in sub:
                    print(f"   Relevance: {sub['relevance_score']:.3f}")
                print()
            
            if len(subsections) > 3:
                print(f"   ... and {len(subsections) - 3} more analyses")
        
        # Display final answer
        if final_answer.strip():
            print(f"\n💡 FINAL ANSWER:")
            print("-" * 50)
            print(final_answer[:600] + "..." if len(final_answer) > 600 else final_answer)
        
        # Display execution metadata
        retrieval_stats = metadata.get('retrieval_stats', {})
        if retrieval_stats:
            print(f"\n📈 PROCESSING STATISTICS:")
            print("-" * 50)
            print(f"Chunks Retrieved: {retrieval_stats.get('total_chunks', 0)}")
            print(f"Unique Documents: {retrieval_stats.get('unique_documents', 0)}")
            print(f"Avg Relevance Score: {retrieval_stats.get('avg_relevance_score', 0):.3f}")
        
        execution_time = job_info.get('execution_timestamp', 'N/A')
        print(f"Execution Time: {execution_time}")
        
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Generalized Job-to-be-Done Execution Pipeline")
    parser.add_argument("--config", type=Path, help="Path to challenge configuration JSON")
    parser.add_argument("--config-dir", type=Path, help="Directory containing multiple configuration files")
    parser.add_argument("--task", type=str, help="Custom task to execute (overrides config task)")
    parser.add_argument("--db-path", type=Path, default=Path("db/faiss_store.pkl"),
                       help="Path to FAISS index")
    parser.add_argument("--input-dir", type=Path, default=Path("data/input"),
                       help="Directory containing JSON configurations")
    parser.add_argument("--output-dir", type=Path, default=Path("data/output"),
                       help="Directory to save job results")
    parser.add_argument("--model-path", type=str, default="models/tinyllama",
                       help="Path to TinyLlama model")
    parser.add_argument("--retrieval-k", type=int, default=8,
                       help="Number of chunks to retrieve")
    parser.add_argument("--output-file", type=str,
                       help="Custom output filename (without extension)")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive job execution mode")
    parser.add_argument("--batch", action="store_true",
                       help="Execute jobs for all configs in input directory")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save the result to file")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not any([args.interactive, args.config, args.config_dir, args.batch]):
        parser.error("Must specify one of: --interactive, --config, --config-dir, or --batch")
    
    if not args.db_path.exists():
        logger.error(f"FAISS index not found at {args.db_path}")
        logger.error("Run 'python ingest.py' first to create the index")
        sys.exit(1)
    
    # Initialize pipeline
    try:
        pipeline = GeneralizedAnswerPipeline(
            db_path=args.db_path,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_path=args.model_path,
            retrieval_k=args.retrieval_k
        )
        
        if args.interactive:
            pipeline.run_interactive_job_mode()
        
        elif args.batch or args.config_dir:
            # Batch job execution
            config_dir = args.config_dir or args.input_dir
            results = pipeline.run_batch_jobs(config_dir)
            
            # Display batch summary
            successful = sum(1 for r in results if r.get('success', False))
            print(f"\n✓ Batch Job Execution Complete:")
            print(f"  Configurations Processed: {len(results)}")
            print(f"  Successful Executions: {successful}")
            print(f"  Failed Executions: {len(results) - successful}")
            
            if successful > 0:
                total_sections = sum(r.get('sections_found', 0) for r in results if r.get('success', False))
                total_analyses = sum(r.get('subsections_analyzed', 0) for r in results if r.get('success', False))
                with_answers = sum(1 for r in results if r.get('success', False) and r.get('has_final_answer', False))
                
                print(f"  Total Sections Extracted: {total_sections}")
                print(f"  Total Subsections Analyzed: {total_analyses}")
                print(f"  Jobs with Final Answers: {with_answers}")
                
                print(f"\n📋 Job Summary:")
                for result in results:
                    status = "✓" if result.get('success', False) else "✗"
                    persona = result.get('persona_role', 'Unknown')
                    config_file = result.get('config_file', 'Unknown')
                    print(f"  {status} {config_file} ({persona})")
        
        else:
            # Single configuration job execution
            if not args.config:
                parser.error("Must specify --config for single job execution")
            
            config = pipeline.load_challenge_config(args.config)
            result = pipeline.execute_job_to_be_done(config, args.task)
            pipeline._display_job_result(result)
            
            # Save if requested
            if not args.no_save:
                output_path = pipeline.save_job_result(result, args.output_file)
                print(f"\n✓ Job result saved to: {output_path}")
            
            sys.exit(0 if result['success'] else 1)
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()