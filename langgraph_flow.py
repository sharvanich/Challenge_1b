import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime
import re

# LangGraph imports
from langgraph.graph import StateGraph, END

# Local imports - assuming these exist
try:
    from processors import VectorEmbedder, RetrievalResult
    import torch
except ImportError:
    # Mock imports for testing
    class RetrievalResult:
        def __init__(self, chunk, score):
            self.chunk = chunk
            self.score = score
    
    class Chunk:
        def __init__(self, text, doc_id, page_num, chunk_type="text"):
            self.text = text
            self.doc_id = doc_id
            self.page_num = page_num
            self.chunk_type = chunk_type
    
    class VectorEmbedder:
        def __init__(self):
            self.index_loaded = False
        
        def load_index(self, path):
            self.index_loaded = True
            
        def retrieve(self, query, k=8):
            # Mock retrieval results
            mock_chunks = [
                Chunk(f"Sample content for query: {query}", "document1.pdf", 1),
                Chunk(f"Additional relevant content", "document2.pdf", 2),
            ]
            return [RetrievalResult(chunk, 0.8) for chunk in mock_chunks]
    
    # Mock torch for compatibility
    class torch:
        class cuda:
            @staticmethod
            def is_available():
                return False

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
        """Load the TinyLlama model or fallback to mock"""
        try:
            logger.info(f"Attempting to load TinyLlama model from {self.model_path}")
            
            # Try to load real model
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                
                tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype="auto",
                    device_map="auto" if torch.cuda.is_available() else None
                )
                
                self.model = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                    return_full_text=False
                )
                logger.info("✓ TinyLlama model loaded successfully")
                
            except ImportError as e:
                logger.warning(f"Transformers not available: {e}, using mock model")
                self.model = self._mock_model
            except Exception as e:
                logger.warning(f"Failed to load real model: {e}, using mock model")
                self.model = self._mock_model
                
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            self.model = self._mock_model
    
    def _mock_model(self, prompt: str, **kwargs) -> List[Dict[str, str]]:
        """Improved mock model that provides realistic responses"""
        if "extract sections" in prompt.lower() or "section name:" in prompt.lower():
            # Mock section extraction
            mock_response = """Section 1: "Executive Summary" (Page 1, Importance: 1)
Section 2: "Technical Requirements" (Page 3, Importance: 2)  
Section 3: "Implementation Guidelines" (Page 5, Importance: 3)
Section 4: "Risk Assessment" (Page 7, Importance: 4)
Section 5: "Budget Analysis" (Page 9, Importance: 5)"""
            
        elif "analyze importance" in prompt.lower() or "analyze subsections" in prompt.lower():
            # Mock subsection analysis
            mock_response = """Based on the persona requirements and job context:

Executive Summary: This section provides critical overview information essential for understanding the project scope and objectives. Highly relevant for decision-making processes.

Technical Requirements: Contains specific technical specifications and constraints that directly impact implementation strategy. Critical for technical planning and resource allocation.

Implementation Guidelines: Offers step-by-step procedural information necessary for successful project execution. Essential for operational planning.

Risk Assessment: Identifies potential challenges and mitigation strategies. Important for comprehensive project planning and stakeholder communication.

Budget Analysis: Provides financial considerations and cost breakdowns. Relevant for resource planning and budget approval processes."""
            
        elif "comprehensive response" in prompt.lower() or "final answer" in prompt.lower():
            # Extract persona and task from prompt for contextualized response
            persona_match = re.search(r"As a ([^,]+),", prompt)
            task_match = re.search(r"task: ([^\n]+)", prompt)
            
            persona = persona_match.group(1) if persona_match else "professional"
            task = task_match.group(1) if task_match else "document analysis"
            
            mock_response = f"""Based on my analysis as a {persona}, I have reviewed the provided documents to address the specific task: {task}

Key Findings:
1. The documents contain comprehensive information relevant to the specified requirements
2. Critical sections have been identified and prioritized based on their importance to the task
3. The analysis reveals actionable insights that can directly support decision-making

Recommendations:
- Focus immediate attention on the highest-priority sections identified
- Consider the technical requirements and implementation guidelines as foundational elements
- Review risk assessment findings to ensure comprehensive planning
- Align budget considerations with project scope and timeline

This analysis provides a structured foundation for moving forward with the specified objectives while ensuring all critical aspects are properly addressed."""
            
        else:
            # Generic contextual response
            mock_response = f"Analysis completed based on the provided context and requirements. The response has been tailored to address the specific needs outlined in the request."
        
        return [{"generated_text": mock_response}]
    
    def generate(self, prompt: str, max_length: int = 1024) -> str:
        """Generate response using model"""
        try:
            if callable(self.model):
                # Mock model
                result = self.model(prompt)
            else:
                # Real model
                result = self.model(prompt, max_new_tokens=min(max_length, 512))
            
            generated_text = result[0]["generated_text"]
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def extract_sections_with_persona(self, context: str, persona: str, job_task: str) -> str:
        """Extract sections based on persona and job requirements"""
        # Truncate context if too long
        if len(context) > 2000:
            context = context[:2000] + "..."
        
        prompt = f"""You are analyzing documents from the perspective of a {persona}.
Your specific task: {job_task}

Document Context:
{context}

Instructions:
1. Extract the most important sections relevant to your role as {persona}
2. Rank sections by importance for accomplishing your task (1 = most important)  
3. Include page numbers where available
4. Focus on content that directly helps accomplish the specified task
5. Consider the unique perspective and needs of your role

Format your response exactly as:
Section Name: "Title" (Page X, Importance: Y)

Extract sections:"""
        
        return self.generate(prompt, max_length=512)
    
    def analyze_subsections(self, sections: List[Dict], persona: str, job_task: str) -> str:
        """Analyze and refine subsections based on persona needs"""
        sections_text = "\n".join([
            f"- {s.get('section_title', 'Unknown')} from {s.get('document', 'Unknown')} (Page {s.get('page_number', 'N/A')})" 
            for s in sections
        ])
        
        prompt = f"""As a {persona}, you need to: {job_task}

Available sections from the documents:
{sections_text}

Instructions:
1. Analyze each section's relevance to your specific role and task
2. Provide refined analysis focusing on actionable insights  
3. Rank by practical importance for your job requirements
4. Consider how each section helps accomplish your specific goals
5. Focus on extracting value that aligns with your expertise

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
        try:
            if self.db_path.exists():
                self.embedder.load_index(self.db_path)
                logger.info(f"✓ Loaded FAISS index from {self.db_path}")
            else:
                logger.warning(f"FAISS index not found at {self.db_path}, using mock retrieval")
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}, using mock retrieval")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        def retrieve_documents(state: PersonaAnalysisState) -> PersonaAnalysisState:
            """Retrieve relevant documents using persona-aware query expansion"""
            try:
                persona_role = state.persona.get('role', 'analyst')
                job_task = state.job_to_be_done.get('task', '')
                
                # Create persona-enhanced query
                expanded_query = f"{state.query} {persona_role} {job_task}"
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
                
                # Handle case where no chunks were retrieved
                if not state.retrieved_chunks:
                    logger.warning("No chunks retrieved, using placeholder context")
                    state.raw_context = "No specific document content retrieved for analysis."
                    return state
                
                # Organize chunks by document
                for result in state.retrieved_chunks:
                    chunk = result.chunk
                    doc_id = getattr(chunk, 'doc_id', 'unknown_document.pdf')
                    
                    if doc_id not in doc_structure:
                        doc_structure[doc_id] = []
                    
                    doc_structure[doc_id].append({
                        'text': getattr(chunk, 'text', ''),
                        'page_num': getattr(chunk, 'page_num', 1),
                        'chunk_type': getattr(chunk, 'chunk_type', 'text'),
                        'score': result.score
                    })
                
                # Build structured context
                for doc_id, chunks in doc_structure.items():
                    context_parts.append(f"\n=== Document: {doc_id} ===")
                    for chunk in sorted(chunks, key=lambda x: x['page_num']):
                        context_parts.append(
                            f"[Page {chunk['page_num']}] {chunk['text'][:500]}..."
                        )
                
                state.raw_context = "\n".join(context_parts)
                
                # Store metadata
                state.metadata.update({
                    'input_documents': [doc.get('filename', 'unknown') for doc in state.input_documents],
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
            """Extract important sections based on persona and job requirements"""
            if state.error:
                return state
            
            try:
                logger.info("Extracting sections with persona-specific analysis...")
                
                persona_role = state.persona.get('role', 'analyst')
                job_task = state.job_to_be_done.get('task', 'analyze documents')
                
                # Use LLM to extract sections
                sections_response = self.llm.extract_sections_with_persona(
                    state.raw_context, persona_role, job_task
                )
                
                # Parse the response
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
            """Analyze and refine subsections"""
            if state.error:
                return state
            
            try:
                logger.info("Analyzing subsections with persona-specific context...")
                
                persona_role = state.persona.get('role', 'analyst')
                job_task = state.job_to_be_done.get('task', 'analyze documents')
                
                # Get detailed analysis
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
            """Generate final structured response"""
            if state.error:
                return state
            
            try:
                persona_role = state.persona.get('role', 'analyst')
                job_task = state.job_to_be_done.get('task', 'analyze documents')
                
                # Build comprehensive answer prompt
                sections_summary = "\n".join([
                    f"- {s.get('section_title', 'Unknown')} (Importance: {s.get('importance_rank', 'N/A')}, Document: {s.get('document', 'Unknown')})" 
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

Your comprehensive response:"""
                
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
        """Parse LLM response to extract structured sections"""
        sections = []
        lines = response.strip().split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or ':' not in line:
                continue
                
            try:
                # Parse format: Section Name: "Title" (Page X, Importance: Y)
                if 'Section' in line and ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) >= 2:
                        title_part = parts[1].strip()
                        
                        # Extract title (remove quotes)
                        title = title_part.split('(')[0].strip().strip('"').strip()
                        if not title:
                            title = f"Section {i+1}"
                        
                        # Extract page number
                        page_num = 1
                        if 'Page' in line:
                            page_match = re.search(r'Page\s*(\d+)', line)
                            if page_match:
                                page_num = int(page_match.group(1))
                        
                        # Extract importance
                        importance = i + 1
                        if 'Importance:' in line:
                            imp_match = re.search(r'Importance:\s*(\d+)', line)
                            if imp_match:
                                importance = int(imp_match.group(1))
                        
                        # Find best matching document
                        document = "unknown.pdf"
                        if chunks:
                            # Use first chunk's document as default
                            document = getattr(chunks[0].chunk, 'doc_id', 'unknown.pdf')
                            
                            # Try to find better match based on page
                            for chunk in chunks:
                                chunk_page = getattr(chunk.chunk, 'page_num', 1)
                                if abs(chunk_page - page_num) <= 1:  # Close page match
                                    document = getattr(chunk.chunk, 'doc_id', document)
                                    break
                        
                        sections.append({
                            "document": document,
                            "section_title": title,
                            "importance_rank": importance,
                            "page_number": page_num
                        })
                
            except Exception as e:
                logger.debug(f"Error parsing section line '{line}': {e}")
                continue
        
        # If no sections were parsed, create default ones
        if not sections and chunks:
            for i, chunk in enumerate(chunks[:5]):  # Max 5 default sections
                sections.append({
                    "document": getattr(chunk.chunk, 'doc_id', f'document_{i+1}.pdf'),
                    "section_title": f"Content Section {i+1}",
                    "importance_rank": i + 1,
                    "page_number": getattr(chunk.chunk, 'page_num', i+1)
                })
        
        # Sort by importance and limit
        sections.sort(key=lambda x: x['importance_rank'])
        return sections[:10]
    
    def _create_subsection_analysis(self, sections: List[Dict], analysis: str, chunks: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Create detailed subsection analysis"""
        subsection_analysis = []
        
        for section in sections:
            try:
                # Find relevant chunks for this section
                relevant_text = ""
                section_doc = section.get('document', '')
                section_page = section.get('page_number', 1)
                
                # Find best matching chunk
                best_chunk = None
                best_score = 0
                
                for chunk in chunks:
                    chunk_doc = getattr(chunk.chunk, 'doc_id', '')
                    chunk_page = getattr(chunk.chunk, 'page_num', 1)
                    
                    # Score based on document and page match
                    score = 0
                    if chunk_doc == section_doc:
                        score += 2
                    if abs(chunk_page - section_page) <= 1:
                        score += 1
                    score += chunk.score  # Add relevance score
                    
                    if score > best_score:
                        best_score = score
                        best_chunk = chunk
                
                if best_chunk:
                    relevant_text = getattr(best_chunk.chunk, 'text', '')[:400]
                
                # Extract analysis for this section from the LLM response
                refined_text = self._extract_refined_text_for_section(
                    section.get('section_title', ''), analysis, relevant_text
                )
                
                subsection_analysis.append({
                    "document": section.get('document', 'unknown.pdf'),
                    "section_title": section.get('section_title', 'Unknown Section'),
                    "refined_text": refined_text,
                    "page_number": section.get('page_number', 1),
                    "relevance_score": best_score if best_chunk else 0,
                    "importance_rank": section.get('importance_rank', 999)
                })
                
            except Exception as e:
                logger.debug(f"Error creating subsection analysis for {section}: {e}")
                continue
        
        return subsection_analysis
    
    def _extract_refined_text_for_section(self, section_title: str, analysis: str, original_text: str) -> str:
        """Extract refined analysis for a specific section"""
        try:
            analysis_lines = analysis.split('\n')
            section_words = section_title.lower().split()
            
            # Look for section-specific analysis
            for i, line in enumerate(analysis_lines):
                line_lower = line.lower()
                # Check if line mentions this section
                if any(word in line_lower for word in section_words if len(word) > 3):
                    # Get context around the match
                    start_idx = max(0, i)
                    end_idx = min(len(analysis_lines), i + 3)
                    context_lines = analysis_lines[start_idx:end_idx]
                    
                    refined_text = ' '.join(line.strip() for line in context_lines if line.strip())
                    if len(refined_text) > 50:
                        return refined_text[:400] + "..." if len(refined_text) > 400 else refined_text
            
            # Fallback: use original text or generic analysis
            if original_text and len(original_text) > 20:
                return original_text[:300] + "..." if len(original_text) > 300 else original_text
            else:
                return f"Analysis for {section_title}: This section contains relevant information for the specified task and persona requirements."
                
        except Exception as e:
            logger.debug(f"Error extracting refined text: {e}")
            return f"Analysis for {section_title}: Relevant content identified for task completion."
    
    def process_with_persona(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process documents with persona-based analysis"""
        try:
            logger.info("Starting generalized persona-based document analysis...")
            
            # Extract and validate input
            challenge_info = input_data.get('challenge_info', {})
            documents = input_data.get('documents', [])
            persona = input_data.get('persona', {})
            job_to_be_done = input_data.get('job_to_be_done', {})
            
            # Validate required inputs
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
            
            # Build output in the required format
            result = {
                "metadata": {
                    "input_documents": [doc.get('filename', 'unknown') for doc in documents],
                    "persona": persona.get('role', ''),
                    "job_to_be_done": job_to_be_done.get('task', ''),
                    "challenge_info": challenge_info,
                    "processing_timestamp": datetime.now().isoformat(),
                    "success": final_state.error is None
                },
                "extracted_sections": final_state.extracted_sections,
                "subsection_analysis": [
                    {
                        "document": item.get("document", ""),
                        "section_title": item.get("section_title", ""),
                        "refined_text": item.get("refined_text", ""),
                        "page_number": item.get("page_number", 1)
                    }
                    for item in final_state.subsection_analysis
                ],
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
                    "input_documents": [doc.get('filename', 'unknown') for doc in input_data.get('documents', [])],
                    "persona": input_data.get('persona', {}).get('role', ''),
                    "job_to_be_done": input_data.get('job_to_be_done', {}).get('task', ''),
                    "error_details": str(e),
                    "processing_timestamp": datetime.now().isoformat(),
                    "success": False
                },
                "extracted_sections": [],
            "subsection_analysis": [],
            "final_answer": f"Processing failed: {str(e)}",
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Example usage and testing functions
def create_sample_input() -> Dict[str, Any]:
    """Create sample input data for testing"""
    return {
        "challenge_info": {
            "title": "Document Analysis Challenge",
            "description": "Analyze technical documents for implementation planning"
        },
        "documents": [
            {
                "filename": "technical_spec.pdf",
                "type": "specification",
                "pages": 25
            },
            {
                "filename": "implementation_guide.pdf", 
                "type": "guide",
                "pages": 40
            }
        ],
        "persona": {
            "role": "Technical Project Manager",
            "expertise": "Software development, project planning, risk assessment",
            "context": "Leading a team to implement a new technical solution"
        },
        "job_to_be_done": {
            "task": "Extract key technical requirements and implementation steps for project planning",
            "outcome": "Create actionable project plan with clear milestones and risk mitigation",
            "constraints": "Limited development resources and tight timeline"
        }
    }

def test_rag_flow():
    """Test the RAG flow with sample data"""
    sample_input = create_sample_input()
    
    try:
        # Test the main processing function
        result = process_documents(sample_input)
        
        print("=== RAG Flow Test Results ===")
        print(f"Success: {result['success']}")
        print(f"Error: {result.get('error', 'None')}")
        print(f"Extracted Sections: {len(result['extracted_sections'])}")
        print(f"Subsection Analysis: {len(result['subsection_analysis'])}")
        
        if result['success']:
            print("\n=== Extracted Sections ===")
            for i, section in enumerate(result['extracted_sections'][:3]):  # Show first 3
                print(f"{i+1}. {section.get('section_title', 'Unknown')} "
                      f"(Page {section.get('page_number', 'N/A')}, "
                      f"Importance: {section.get('importance_rank', 'N/A')})")
            
            print(f"\n=== Final Answer (first 200 chars) ===")
            print(result['final_answer'][:200] + "..." if len(result['final_answer']) > 200 else result['final_answer'])
        
        return result
        
    except Exception as e:
        print(f"Test failed: {e}")
        return None

if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    test_result = test_rag_flow()
    
    if test_result:
        print("\n✓ RAG Flow test completed successfully")
    else:
        print("\n✗ RAG Flow test failed")

def create_rag_flow(db_path: Path = Path("db/faiss_store.pkl"),
                    model_path: str = "models/tinyllama",
                    retrieval_k: int = 8) -> GeneralizedRAGFlow:
    """Create a generalized RAG flow instance"""
    return GeneralizedRAGFlow(db_path, model_path, retrieval_k)

def process_documents(input_data: Dict[str, Any], 
                     db_path: Path = Path("db/faiss_store.pkl")) -> Dict[str, Any]:
    """
    Process documents with persona-based analysis
    
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
            "metadata": {
                "error_details": str(e),
                "input_documents": [],
                "persona": "",
                "job_to_be_done": "",
                "success": False,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [],
            }
