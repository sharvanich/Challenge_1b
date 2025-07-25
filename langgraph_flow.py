import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime

# LangGraph imports
from langgraph.graph import StateGraph, END

# Local imports
from processors import VectorEmbedder, RetrievalResult

logger = logging.getLogger(__name__)

@dataclass
class PersonaAnalysisState:
    """Enhanced state object for persona-based analysis"""
    # Input data
    query: str
    persona: Dict[str, str]
    job_to_be_done: Dict[str, str]
    input_documents: List[Dict[str, str]]
    
    # Processing state
    retrieved_chunks: List[RetrievalResult] = field(default_factory=list)
    raw_context: str = ""
    
    # Analysis results
    extracted_sections: List[Dict[str, Any]] = field(default_factory=list)
    subsection_analysis: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

class PersonaAwareLLM:
    """Enhanced LLM wrapper with persona-aware prompting"""
    
    def __init__(self, model_path: str = "models/tinyllama"):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the TinyLlama model"""
        try:
            logger.info(f"Loading TinyLlama model from {self.model_path}")
            
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                
                tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                model = AutoModelForCausalLM.from_pretrained(self.model_path)
                
                self.model = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=1024,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
                logger.info("✓ TinyLlama model loaded successfully")
                
            except ImportError:
                logger.warning("Transformers not available, using mock model")
                self.model = self._mock_model
            except Exception as e:
                logger.warning(f"Failed to load model, using mock: {e}")
                self.model = self._mock_model
                
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            self.model = self._mock_model
    
    def _mock_model(self, prompt: str, **kwargs) -> List[Dict[str, str]]:
        """Generalized mock model for any document type"""
        if "extract sections" in prompt.lower():
            mock_response = """
            Section 1: "Introduction and Overview" (Page 1, Importance: 1)
            Section 2: "Main Content Area" (Page 2, Importance: 2)
            Section 3: "Supporting Information" (Page 3, Importance: 3)
            Section 4: "Additional Details" (Page 4, Importance: 4)
            """
        elif "analyze importance" in prompt.lower():
            mock_response = """
            Based on the persona role and job requirements:
            1. Main Content Area - Most relevant to the specified task
            2. Introduction and Overview - Provides necessary context
            3. Supporting Information - Helpful supplementary material
            4. Additional Details - Secondary relevance
            """
        else:
            mock_response = f"Analysis response tailored to the specified persona and job requirements. Context analyzed from provided documents."
        
        return [{"generated_text": prompt + "\n\n" + mock_response}]
    
    def generate(self, prompt: str, max_length: int = 1024) -> str:
        """Generate response using TinyLlama"""
        try:
            if callable(self.model):
                result = self.model(prompt)
            else:
                result = self.model(prompt, max_length=max_length)
            
            generated_text = result[0]["generated_text"]
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def extract_sections_with_persona(self, context: str, persona: str, job_task: str) -> str:
        """Extract sections based on persona and job requirements - works with any document type"""
        prompt = f"""You are analyzing documents from the perspective of a {persona}.
Your specific task: {job_task}

Document Context:
{context}

Instructions:
1. Extract the most important sections relevant to your role as {persona} and your specific task
2. Rank sections by importance for accomplishing your task (1 = most important)
3. Include page numbers where available
4. Focus on content that directly helps accomplish the specified task
5. Consider the unique perspective and needs of your role

Format your response as:
Section Name: "Title" (Page X, Importance: Y)

Extract sections:"""
        
        return self.generate(prompt, max_length=512)
    
    def analyze_subsections(self, sections: List[Dict], persona: str, job_task: str) -> str:
        """Analyze and refine subsections based on persona needs - generalized for any content"""
        sections_text = "\n".join([f"- {s['section_title']} from {s['document']}" for s in sections])
        
        prompt = f"""As a {persona}, you need to: {job_task}

Available sections from the documents:
{sections_text}

Instructions:
1. Analyze each section's relevance to your specific role and task
2. Provide refined analysis focusing on actionable insights for your work
3. Rank by practical importance for your job requirements
4. Consider how each section helps you accomplish your specific goals
5. Focus on extracting value that aligns with your expertise and responsibilities

Provide detailed analysis for each section:"""
        
        return self.generate(prompt, max_length=768)

class GeneralizedRAGFlow:
    """Generalized RAG flow that works with any collection of PDFs and personas"""
    
    def __init__(self, 
                 db_path: Path = Path("db/faiss_store.pkl"),
                 model_path: str = "models/tinyllama",
                 retrieval_k: int = 8):
        
        self.db_path = db_path
        self.retrieval_k = retrieval_k
        
        # Initialize components
        self.embedder = VectorEmbedder()
        self.llm = PersonaAwareLLM(model_path)
        
        # Load index and build graph
        self._load_index()
        self.graph = self._build_graph()
    
    def _load_index(self):
        """Load the FAISS index"""
        if not self.db_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.db_path}")
        
        try:
            self.embedder.load_index(self.db_path)
            logger.info(f"✓ Loaded FAISS index from {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise
    
    def _build_graph(self) -> StateGraph:
        """Build the generalized LangGraph workflow"""
        
        def retrieve_documents(state: PersonaAnalysisState) -> PersonaAnalysisState:
            """Retrieve relevant documents using persona-aware query expansion"""
            try:
                persona_role = state.persona.get('role', '')
                job_task = state.job_to_be_done.get('task', '')
                
                # Create persona-enhanced query
                expanded_query = f"{state.query} {persona_role} perspective {job_task}"
                logger.info(f"Retrieving documents for {persona_role}")
                
                results = self.embedder.retrieve(expanded_query, self.retrieval_k)
                state.retrieved_chunks = results
                
                logger.info(f"✓ Retrieved {len(results)} relevant chunks")
                return state
                
            except Exception as e:
                logger.error(f"Error in document retrieval: {e}")
                state.error = f"Retrieval error: {str(e)}"
                return state
        
        def build_context(state: PersonaAnalysisState) -> PersonaAnalysisState:
            """Build structured context from retrieved chunks"""
            if state.error:
                return state
            
            try:
                context_parts = []
                doc_structure = {}
                
                # Organize chunks by document
                for result in state.retrieved_chunks:
                    chunk = result.chunk
                    doc_id = chunk.doc_id
                    
                    if doc_id not in doc_structure:
                        doc_structure[doc_id] = []
                    
                    doc_structure[doc_id].append({
                        'text': chunk.text,
                        'page_num': chunk.page_num,
                        'chunk_type': chunk.chunk_type,
                        'score': result.score
                    })
                
                # Build structured context preserving document organization
                for doc_id, chunks in doc_structure.items():
                    context_parts.append(f"\n=== Document: {doc_id} ===")
                    for chunk in sorted(chunks, key=lambda x: x['page_num']):
                        context_parts.append(
                            f"[Page {chunk['page_num']}] {chunk['text']}"
                        )
                
                state.raw_context = "\n".join(context_parts)
                
                # Store comprehensive metadata
                state.metadata.update({
                    'input_documents': [doc['filename'] for doc in state.input_documents],
                    'persona': state.persona.get('role', ''),
                    'job_to_be_done': state.job_to_be_done.get('task', ''),
                    'retrieval_stats': {
                        'total_chunks': len(state.retrieved_chunks),
                        'unique_documents': len(doc_structure),
                        'avg_relevance_score': sum(r.score for r in state.retrieved_chunks) / len(state.retrieved_chunks) if state.retrieved_chunks else 0
                    },
                    'processing_timestamp': datetime.now().isoformat()
                })
                
                logger.info(f"✓ Built context from {len(doc_structure)} documents")
                return state
                
            except Exception as e:
                logger.error(f"Error building context: {e}")
                state.error = f"Context building error: {str(e)}"
                return state
        
        def extract_sections(state: PersonaAnalysisState) -> PersonaAnalysisState:
            """Extract important sections based on any persona and job requirements"""
            if state.error:
                return state
            
            try:
                logger.info("Extracting sections with persona-specific analysis...")
                
                persona_role = state.persona.get('role', '')
                job_task = state.job_to_be_done.get('task', '')
                
                # Use LLM to extract sections relevant to this specific persona and task
                sections_response = self.llm.extract_sections_with_persona(
                    state.raw_context, persona_role, job_task
                )
                
                # Parse the response to extract structured sections
                extracted_sections = self._parse_sections_response(
                    sections_response, state.retrieved_chunks
                )
                
                state.extracted_sections = extracted_sections
                
                logger.info(f"✓ Extracted {len(extracted_sections)} relevant sections")
                return state
                
            except Exception as e:
                logger.error(f"Error extracting sections: {e}")
                state.error = f"Section extraction error: {str(e)}"
                return state
        
        def analyze_subsections(state: PersonaAnalysisState) -> PersonaAnalysisState:
            """Analyze and refine subsections for any persona and task"""
            if state.error:
                return state
            
            try:
                logger.info("Analyzing subsections with persona-specific context...")
                
                persona_role = state.persona.get('role', '')
                job_task = state.job_to_be_done.get('task', '')
                
                # Get detailed analysis tailored to this persona and task
                analysis_response = self.llm.analyze_subsections(
                    state.extracted_sections, persona_role, job_task
                )
                
                # Create refined subsection analysis
                subsection_analysis = self._create_subsection_analysis(
                    state.extracted_sections, analysis_response, state.retrieved_chunks
                )
                
                state.subsection_analysis = subsection_analysis
                
                logger.info(f"✓ Analyzed {len(subsection_analysis)} subsections")
                return state
                
            except Exception as e:
                logger.error(f"Error analyzing subsections: {e}")
                state.error = f"Subsection analysis error: {str(e)}"
                return state
        
        def generate_final_response(state: PersonaAnalysisState) -> PersonaAnalysisState:
            """Generate final structured response for any persona and task"""
            if state.error:
                return state
            
            try:
                persona_role = state.persona.get('role', '')
                job_task = state.job_to_be_done.get('task', '')
                
                # Build comprehensive answer prompt
                sections_summary = "\n".join([
                    f"- {s['section_title']} (Importance: {s['importance_rank']}, Document: {s['document']})" 
                    for s in state.extracted_sections
                ])
                
                answer_prompt = f"""As a {persona_role}, based on your analysis of the provided documents:

Your specific task: {job_task}

Key sections identified and analyzed:
{sections_summary}

Provide a comprehensive response that:
1. Directly addresses your specific task requirements
2. Leverages your expertise as a {persona_role}
3. Uses insights from the most relevant document sections
4. Provides actionable recommendations or conclusions

Your response:"""
                
                state.final_answer = self.llm.generate(answer_prompt, max_length=512)
                
                logger.info("✓ Generated persona-specific final response")
                return state
                
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                state.error = f"Response generation error: {str(e)}"
                return state
        
        # Build the graph
        workflow = StateGraph(PersonaAnalysisState)
        
        # Add nodes
        workflow.add_node("retrieve", retrieve_documents)
        workflow.add_node("build_context", build_context)
        workflow.add_node("extract_sections", extract_sections)
        workflow.add_node("analyze_subsections", analyze_subsections)
        workflow.add_node("generate_response", generate_final_response)
        
        # Set up flow
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "build_context")
        workflow.add_edge("build_context", "extract_sections")
        workflow.add_edge("extract_sections", "analyze_subsections")
        workflow.add_edge("analyze_subsections", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    def _parse_sections_response(self, response: str, chunks: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Parse LLM response to extract structured sections - works with any content"""
        sections = []
        lines = response.strip().split('\n')
        
        for i, line in enumerate(lines):
            if ':' in line and ('(' in line or 'Page' in line):
                try:
                    # Parse various formats: Section Name: "Title" (Page X, Importance: Y)
                    parts = line.split(':')
                    if len(parts) >= 2:
                        title = parts[1].strip().strip('"').split('(')[0].strip()
                        
                        # Extract page and importance with flexible parsing
                        page_num = 1
                        importance = i + 1
                        
                        # Look for page number
                        if 'Page' in line:
                            try:
                                page_part = line.split('Page')[1].split(',')[0].split(')')[0].strip()
                                page_num = int(''.join(filter(str.isdigit, page_part)))
                            except:
                                pass
                        
                        # Look for importance ranking
                        if 'Importance:' in line:
                            try:
                                imp_part = line.split('Importance:')[1].split(')')[0].strip()
                                importance = int(''.join(filter(str.isdigit, imp_part)))
                            except:
                                pass
                        
                        # Find best matching document
                        document = "document.pdf"  # default
                        best_match = None
                        best_score = 0
                        
                        for chunk in chunks:
                            # Score based on page proximity and content relevance
                            page_score = max(0, 5 - abs(chunk.chunk.page_num - page_num))
                            content_score = chunk.score
                            total_score = page_score + content_score
                            
                            if total_score > best_score:
                                best_score = total_score
                                best_match = chunk
                        
                        if best_match:
                            document = best_match.chunk.doc_id
                        
                        sections.append({
                            "document": document,
                            "section_title": title,
                            "importance_rank": importance,
                            "page_number": page_num
                        })
                
                except Exception as e:
                    logger.debug(f"Error parsing section line: {line}, error: {e}")
                    continue
        
        # Sort by importance and limit results
        sections.sort(key=lambda x: x['importance_rank'])
        return sections[:10]
    
    def _create_subsection_analysis(self, sections: List[Dict], analysis: str, chunks: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Create detailed subsection analysis for any content type"""
        subsection_analysis = []
        
        for section in sections:
            # Find most relevant chunks for this section
            relevant_chunks = []
            for chunk in chunks:
                if chunk.chunk.doc_id == section['document']:
                    # Score based on page proximity and relevance
                    page_proximity = max(0, 3 - abs(chunk.chunk.page_num - section['page_number']))
                    combined_score = chunk.score + (page_proximity * 0.1)
                    relevant_chunks.append((chunk, combined_score))
            
            if relevant_chunks:
                # Get the best matching chunk
                best_chunk, _ = max(relevant_chunks, key=lambda x: x[1])
                
                # Extract refined text from analysis
                refined_text = self._extract_refined_text_for_section(
                    section['section_title'], analysis, best_chunk.chunk.text
                )
                
                subsection_analysis.append({
                    "document": section['document'],
                    "section_title": section['section_title'],
                    "refined_text": refined_text,
                    "page_number": section['page_number'],
                    "relevance_score": best_chunk.score,
                    "importance_rank": section['importance_rank']
                })
        
        return subsection_analysis
    
    def _extract_refined_text_for_section(self, section_title: str, analysis: str, original_text: str) -> str:
        """Extract refined analysis for a specific section"""
        analysis_lines = analysis.split('\n')
        
        # Look for section-specific analysis
        for i, line in enumerate(analysis_lines):
            if any(word.lower() in line.lower() for word in section_title.lower().split()):
                # Get contextual lines around the match
                start_idx = max(0, i - 1)
                end_idx = min(len(analysis_lines), i + 4)
                refined_lines = analysis_lines[start_idx:end_idx]
                refined_text = ' '.join(line.strip() for line in refined_lines if line.strip())
                
                if len(refined_text) > 50:
                    return refined_text[:400] + "..." if len(refined_text) > 400 else refined_text
        
        # Fallback to truncated original text with context
        return original_text[:300] + "..." if len(original_text) > 300 else original_text
    
    def process_with_persona(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process any collection of documents with any persona and task"""
        try:
            logger.info("Starting generalized persona-based document analysis...")
            
            # Extract and validate input components
            challenge_info = input_data.get('challenge_info', {})
            documents = input_data.get('documents', [])
            persona = input_data.get('persona', {})
            job_to_be_done = input_data.get('job_to_be_done', {})
            
            # Validate required inputs
            if not documents:
                raise ValueError("No documents provided for analysis")
            if not persona.get('role'):
                raise ValueError("Persona role is required")
            if not job_to_be_done.get('task'):
                raise ValueError("Job task description is required")
            
            # Create query from job requirements
            query = job_to_be_done.get('task', 'Analyze documents')
            
            logger.info(f"Processing {len(documents)} documents for {persona.get('role')} role")
            
            # Initialize state
            initial_state = PersonaAnalysisState(
                query=query,
                persona=persona,
                job_to_be_done=job_to_be_done,
                input_documents=documents
            )
            
            # Run the workflow
            final_state = self.graph.invoke(initial_state)
            
            # Build standardized output
            result = {
                "metadata": final_state.metadata,
                "extracted_sections": final_state.extracted_sections,
                "subsection_analysis": final_state.subsection_analysis,
                "final_answer": final_state.final_answer,
                "success": final_state.error is None,
                "error": final_state.error,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✓ Analysis completed. Success: {result['success']}, Sections: {len(result['extracted_sections'])}")
            return result
            
        except Exception as e:
            logger.error(f"Error in generalized processing: {e}")
            return {
                "metadata": {
                    "error_details": str(e),
                    "input_documents": input_data.get('documents', []),
                    "persona": input_data.get('persona', {}),
                    "job_to_be_done": input_data.get('job_to_be_done', {})
                },
                "extracted_sections": [],
                "subsection_analysis": [],
                "final_answer": "",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

def create_rag_flow(db_path: Path = Path("db/faiss_store.pkl"),
                    model_path: str = "models/tinyllama",
                    retrieval_k: int = 8) -> GeneralizedRAGFlow:
    """Create a generalized RAG flow instance that works with any documents and personas"""
    return GeneralizedRAGFlow(db_path, model_path, retrieval_k)

# Example usage function (can be removed if not needed)
def process_documents(input_data: Dict[str, Any], 
                     db_path: Path = Path("db/faiss_store.pkl")) -> Dict[str, Any]:
    """
    Process any collection of documents with any persona and task
    
    Args:
        input_data: Dictionary containing challenge_info, documents, persona, and job_to_be_done
        db_path: Path to the FAISS database
    
    Returns:
        Dictionary with analysis results in the specified format
    """
    try:
        rag_flow = create_rag_flow(db_path)
        return rag_flow.process_with_persona(input_data)
    except Exception as e:
        logger.error(f"Error in document processing: {e}")
        return {
            "metadata": {},
            "extracted_sections": [],
            "subsection_analysis": [],
            "final_answer": "",
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }