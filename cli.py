"""
Ultra-minimal CLI for the agent
Made by Rodrigo de Sarasqueta
"""

import asyncio
import sys
from pathlib import Path

from config import Config
from core.agent import MinimalAgent


async def main():
    """Simple chat interface"""
    print("AgentAlpha - Minimalist AI Agent")
    print("Commands: 'quit' to exit, 'clear' to reset memory")
    print("=" * 50)
    
    # Initialize agent
    try:
        config = Config()
        agent = MinimalAgent(config.to_dict())
        await agent.initialize()
        print("Agent ready!\n")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
                
            if user_input.lower() == 'clear':
                await agent.clear_memory()
                print("Memory cleared.\n")
                continue
            
            # Get response
            result = await agent.chat(user_input)
            
            if result["success"]:
                print(f"Agent: {result['response']}\n")
            else:
                print(f"Error: {result['response']}\n")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def add_pdf(file_path: str):
    """Add PDF to knowledge base"""
    async def _add():
        config = Config()
        agent = MinimalAgent(config.to_dict())
        await agent.initialize()
        
        try:
            doc_id = await agent.add_document(file_path)
            print(f"Document added: {doc_id}")
        except Exception as e:
            print(f"Error: {e}")
    
    asyncio.run(_add())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "add-pdf":
        if len(sys.argv) < 3:
            print("Usage: python cli.py add-pdf <file.pdf>")
            sys.exit(1)
        add_pdf(sys.argv[2])
    else:
        asyncio.run(main())