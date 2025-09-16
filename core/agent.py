"""
Core Agent with dual memory system (buffer + summary)
Made by Rodrigo de Sarasqueta
"""

import asyncio
from typing import Dict, Any, List, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from .llm_client import LLMClient
from .vector_store import VectorStore
from .tools import DocumentRetrievalTool, HTTPRequestTool


class MinimalAgent:
    """
    Minimal RAG Agent with dual memory system
    Features:
    - Multi-LLM support (Ollama, OpenAI, DeepSeek)
    - ChromaDB vector store for documents
    - Dual memory: recent buffer + summary for long context
    - Extensible tools system
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_client = LLMClient(config)
        self.vector_store = VectorStore(config)
        self.memory = None
        self.agent_executor = None
        self.tools = []
        
    async def initialize(self):
        """Initialize all agent components"""
        print("Initializing Minimal Agent...")
        
        # Initialize LLM client
        await self.llm_client.initialize()
        
        # Initialize vector store
        await self.vector_store.initialize()
        
        # Setup dual memory system
        await self._setup_memory()
        
        # Setup tools
        await self._setup_tools()
        
        # Setup agent
        await self._setup_agent()
        
        print("Agent initialized successfully")
    
    async def _setup_memory(self):
        """Setup dual memory: buffer + summary"""
        llm = self.llm_client.get_llm()
        memory_config = self.config.get("memory", {})
        
        # ConversationSummaryBufferMemory automatically handles:
        # - Recent messages in buffer (up to max_token_limit)
        # - Older messages compressed into summary
        self.memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=memory_config.get("max_token_limit", 2000),
            return_messages=True,
            memory_key="chat_history"
        )
        
        print(f"Dual memory initialized (max tokens: {memory_config.get('max_token_limit', 2000)})")
    
    async def _setup_tools(self):
        """Setup available tools for the agent"""
        # Document retrieval tool
        doc_tool = DocumentRetrievalTool(
            vector_store=self.vector_store,
            config=self.config.get("retrieval", {})
        )
        
        # HTTP request tool for external integrations
        http_tool = HTTPRequestTool(
            config=self.config.get("http", {})
        )
        
        self.tools = [doc_tool, http_tool]
        print(f"Initialized {len(self.tools)} tools")
    
    async def _setup_agent(self):
        """Setup LangChain ReAct agent"""
        system_instructions = self.config.get("system_instructions", 
                                             "You are a helpful AI assistant.")
        
        # Create prompt template
        template = f"""{system_instructions}

Available tools: {{tools}}
Tool names: {{tool_names}}

Use tools when you need specific information. Always provide helpful responses.

Chat History:
{{chat_history}}

User: {{input}}

Think step by step:
{{agent_scratchpad}}"""

        prompt = PromptTemplate.from_template(template)
        
        # Create ReAct agent
        agent = create_react_agent(
            llm=self.llm_client.get_llm(),
            tools=self.tools,
            prompt=prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=3,
            max_execution_time=30
        )
        
        print("ReAct agent initialized")
    
    async def chat(self, message: str) -> Dict[str, Any]:
        """Process a chat message and return response"""
        if not self.agent_executor:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        try:
            # Execute agent with timeout
            result = await asyncio.wait_for(
                self.agent_executor.ainvoke({"input": message}),
                timeout=25.0
            )
            
            return {
                "response": result["output"],
                "success": True
            }
            
        except asyncio.TimeoutError:
            return {
                "response": "Request timed out. Please try a simpler question.",
                "success": False,
                "error": "timeout"
            }
        except Exception as e:
            return {
                "response": f"Error processing message: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    async def add_document(self, file_path: str, metadata: Optional[Dict] = None) -> str:
        """Add a document to the vector store"""
        return await self.vector_store.add_document_from_file(file_path, metadata)
    
    async def clear_memory(self):
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            print("Memory cleared")
    
    async def get_memory_summary(self) -> str:
        """Get current memory summary"""
        if not self.memory:
            return "No memory initialized"
        
        # Get the current buffer contents
        buffer = self.memory.chat_memory.messages
        summary = getattr(self.memory, 'moving_summary_buffer', 'No summary yet')
        
        return f"Buffer messages: {len(buffer)}, Summary: {summary}"
    
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return [tool.name for tool in self.tools]
