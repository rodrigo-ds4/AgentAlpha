"""
RAGAS evaluation for AgentAlpha
Made by Rodrigo de Sarasqueta
"""

import asyncio
import json
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)

from config import Config
from core.agent import MinimalAgent


class AgentEvaluator:
    """Evaluate AgentAlpha performance with RAGAS"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = Config(config_path)
        self.agent = None
        
    async def initialize(self):
        """Initialize the agent"""
        self.agent = MinimalAgent(self.config.to_dict())
        await self.agent.initialize()
        print("Agent initialized for evaluation")
    
    async def evaluate_questions(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate agent on test questions
        
        test_data format:
        [
            {
                "question": "What is the capital of France?",
                "ground_truth": "Paris",
                "contexts": ["Paris is the capital of France..."] # Optional
            }
        ]
        """
        if not self.agent:
            await self.initialize()
        
        print(f"Evaluating {len(test_data)} questions...")
        
        # Collect agent responses
        evaluation_data = []
        
        for i, item in enumerate(test_data):
            question = item["question"]
            ground_truth = item["ground_truth"]
            
            print(f"Processing question {i+1}/{len(test_data)}: {question[:50]}...")
            
            # Get agent response
            result = await self.agent.chat(question)
            answer = result["response"]
            
            # Get retrieved contexts
            contexts = await self._get_contexts(question)
            
            evaluation_data.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth
            })
        
        # Create dataset for RAGAS
        dataset = Dataset.from_list(evaluation_data)
        
        # Evaluate with RAGAS metrics
        print("Running RAGAS evaluation...")
        results = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness
            ]
        )
        
        return dict(results)
    
    async def _get_contexts(self, question: str) -> List[str]:
        """Get retrieved contexts for a question"""
        try:
            # Use the agent's vector store to get contexts
            search_results = await self.agent.vector_store.search(question, n_results=5)
            contexts = [result["content"] for result in search_results]
            return contexts if contexts else ["No context retrieved"]
        except Exception as e:
            print(f"Error getting contexts: {e}")
            return ["No context available"]
    
    async def run_evaluation_from_file(self, test_file: str) -> Dict[str, float]:
        """Load test data from JSON file and evaluate"""
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            return await self.evaluate_questions(test_data)
            
        except Exception as e:
            print(f"Error loading test file: {e}")
            return {}
    
    def print_results(self, results: Dict[str, float]):
        """Print evaluation results in a nice format"""
        print("\n" + "="*50)
        print("📊 RAGAS EVALUATION RESULTS")
        print("="*50)
        
        for metric, score in results.items():
            percentage = score * 100
            print(f"{metric:20s}: {score:.3f} ({percentage:.1f}%)")
        
        # Overall assessment
        avg_score = sum(results.values()) / len(results) if results else 0
        print(f"\n{'Average Score':20s}: {avg_score:.3f} ({avg_score*100:.1f}%)")
        
        if avg_score >= 0.8:
            assessment = "🟢 Excellent"
        elif avg_score >= 0.7:
            assessment = "🟡 Good"
        elif avg_score >= 0.6:
            assessment = "🟠 Needs Improvement"
        else:
            assessment = "🔴 Poor"
        
        print(f"Assessment: {assessment}")
        print("="*50)


async def main():
    """CLI interface for evaluation"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <test_file.json>")
        print("\nExample test_file.json:")
        example = [
            {
                "question": "What is AgentAlpha?",
                "ground_truth": "A minimal RAG agent with multi-LLM support"
            }
        ]
        print(json.dumps(example, indent=2))
        return
    
    test_file = sys.argv[1]
    
    # Run evaluation
    evaluator = AgentEvaluator()
    results = await evaluator.run_evaluation_from_file(test_file)
    
    if results:
        evaluator.print_results(results)
    else:
        print("❌ Evaluation failed")


if __name__ == "__main__":
    asyncio.run(main())
