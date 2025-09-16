"""
Simple evaluation for AgentAlpha (Python 3.9 compatible)
Made by Rodrigo de Sarasqueta
"""

import asyncio
import json
from typing import List, Dict, Any
import time

from config import Config
from core.agent import MinimalAgent


class SimpleEvaluator:
    """Simple evaluation metrics for AgentAlpha"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = Config(config_path)
        self.agent = None
        
    async def initialize(self):
        """Initialize the agent"""
        self.agent = MinimalAgent(self.config.to_dict())
        await self.agent.initialize()
        print("Agent initialized for evaluation")
    
    async def evaluate_questions(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple evaluation on test questions"""
        if not self.agent:
            await self.initialize()
        
        print(f"Evaluating {len(test_data)} questions...")
        
        results = {
            "total_questions": len(test_data),
            "responses": [],
            "avg_response_time": 0,
            "successful_responses": 0
        }
        
        total_time = 0
        
        for i, item in enumerate(test_data):
            question = item["question"]
            expected = item.get("ground_truth", "")
            
            print(f"Question {i+1}/{len(test_data)}: {question[:60]}...")
            
            # Measure response time
            start_time = time.time()
            result = await self.agent.chat(question)
            response_time = time.time() - start_time
            
            total_time += response_time
            
            if result["success"]:
                results["successful_responses"] += 1
            
            # Simple keyword matching for relevance
            relevance_score = self._calculate_relevance(result["response"], expected)
            
            response_data = {
                "question": question,
                "answer": result["response"],
                "expected": expected,
                "success": result["success"],
                "response_time": round(response_time, 2),
                "relevance_score": relevance_score
            }
            
            results["responses"].append(response_data)
        
        results["avg_response_time"] = round(total_time / len(test_data), 2)
        results["success_rate"] = results["successful_responses"] / len(test_data)
        
        return results
    
    def _calculate_relevance(self, answer: str, expected: str) -> float:
        """Simple keyword-based relevance scoring"""
        if not expected:
            return 1.0
        
        answer_lower = answer.lower()
        expected_lower = expected.lower()
        
        # Extract keywords from expected answer
        keywords = expected_lower.split()
        matches = sum(1 for keyword in keywords if keyword in answer_lower)
        
        return matches / len(keywords) if keywords else 0.0
    
    async def evaluate_retrieval(self, questions: List[str]) -> Dict[str, Any]:
        """Evaluate document retrieval quality"""
        if not self.agent:
            await self.initialize()
        
        print(f"Evaluating retrieval for {len(questions)} questions...")
        
        retrieval_results = {
            "total_queries": len(questions),
            "avg_documents_found": 0,
            "queries_with_results": 0,
            "details": []
        }
        
        total_docs = 0
        
        for i, question in enumerate(questions):
            print(f"Retrieval {i+1}/{len(questions)}: {question[:60]}...")
            
            # Get retrieval results
            docs = await self.agent.vector_store.search(question, n_results=5)
            
            doc_count = len(docs)
            total_docs += doc_count
            
            if doc_count > 0:
                retrieval_results["queries_with_results"] += 1
            
            retrieval_results["details"].append({
                "question": question,
                "documents_found": doc_count,
                "top_similarity": docs[0]["similarity_score"] if docs else 0.0
            })
        
        retrieval_results["avg_documents_found"] = round(total_docs / len(questions), 2)
        retrieval_results["retrieval_success_rate"] = retrieval_results["queries_with_results"] / len(questions)
        
        return retrieval_results
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results"""
        print("\n" + "="*60)
        print("📊 AGENT EVALUATION RESULTS")
        print("="*60)
        
        # Overall metrics
        print(f"Total Questions: {results['total_questions']}")
        print(f"Success Rate: {results['success_rate']:.1%}")
        print(f"Avg Response Time: {results['avg_response_time']}s")
        
        # Calculate average relevance
        relevance_scores = [r['relevance_score'] for r in results['responses']]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        print(f"Avg Relevance: {avg_relevance:.1%}")
        
        # Performance assessment
        if avg_relevance >= 0.8 and results['success_rate'] >= 0.9:
            assessment = "🟢 Excellent"
        elif avg_relevance >= 0.6 and results['success_rate'] >= 0.8:
            assessment = "🟡 Good"
        elif avg_relevance >= 0.4 and results['success_rate'] >= 0.6:
            assessment = "🟠 Fair"
        else:
            assessment = "🔴 Needs Improvement"
        
        print(f"Assessment: {assessment}")
        
        # Detailed results
        print("\n📋 DETAILED RESULTS:")
        print("-" * 60)
        for i, resp in enumerate(results['responses'], 1):
            status = "✅" if resp['success'] else "❌"
            print(f"{i}. {status} {resp['relevance_score']:.0%} relevance | {resp['response_time']}s")
            print(f"   Q: {resp['question']}")
            print(f"   A: {resp['answer'][:100]}{'...' if len(resp['answer']) > 100 else ''}")
            print()
        
        print("="*60)


async def main():
    """CLI interface for simple evaluation"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_simple.py <test_file.json>")
        print("\nExample test_file.json:")
        example = [
            {
                "question": "What is AgentAlpha?",
                "ground_truth": "minimal RAG agent"
            }
        ]
        print(json.dumps(example, indent=2))
        return
    
    test_file = sys.argv[1]
    
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading test file: {e}")
        return
    
    # Run evaluation
    evaluator = SimpleEvaluator()
    results = await evaluator.evaluate_questions(test_data)
    
    evaluator.print_results(results)
    
    # Also evaluate retrieval if there are questions
    questions = [item["question"] for item in test_data]
    print("\n🔍 EVALUATING RETRIEVAL...")
    retrieval_results = await evaluator.evaluate_retrieval(questions)
    
    print(f"Retrieval Success Rate: {retrieval_results['retrieval_success_rate']:.1%}")
    print(f"Avg Documents Found: {retrieval_results['avg_documents_found']}")


if __name__ == "__main__":
    asyncio.run(main())
