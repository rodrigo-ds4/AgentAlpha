"""
Mock evaluation for AgentAlpha (no LLM required)
Made by Rodrigo de Sarasqueta
"""

import asyncio
import json
from typing import List, Dict, Any
import time
import random


class MockAgent:
    """Mock agent for demonstration purposes"""
    
    def __init__(self):
        self.responses = [
            "AgentAlpha is a minimal RAG agent with multi-LLM support and dual memory system.",
            "AgentAlpha supports Ollama, OpenAI, and DeepSeek as LLM providers.",
            "The memory system uses a buffer for recent messages and automatic summarization.",
            "AgentAlpha can process PDF and TXT documents using ChromaDB for vector storage.",
            "You can deploy AgentAlpha using Docker with the docker-compose up command."
        ]
    
    async def chat(self, question: str) -> Dict[str, Any]:
        """Mock chat response"""
        # Simulate thinking time
        await asyncio.sleep(random.uniform(0.5, 2.0))
        
        # Select response based on question keywords
        response = self._select_response(question)
        
        return {
            "response": response,
            "success": True
        }
    
    def _select_response(self, question: str) -> str:
        """Select appropriate response based on question"""
        question_lower = question.lower()
        
        if "agentAlpha" in question or "agent" in question_lower:
            return self.responses[0]
        elif "llm" in question_lower or "provider" in question_lower:
            return self.responses[1]
        elif "memory" in question_lower:
            return self.responses[2]
        elif "document" in question_lower or "pdf" in question_lower:
            return self.responses[3]
        elif "deploy" in question_lower or "docker" in question_lower:
            return self.responses[4]
        else:
            return random.choice(self.responses)


class MockEvaluator:
    """Mock evaluation system"""
    
    def __init__(self):
        self.agent = MockAgent()
    
    async def evaluate_questions(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate mock agent responses"""
        print(f"🤖 Evaluating {len(test_data)} questions with MockAgent...")
        
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
            print(f"   ✅ {relevance_score:.0%} relevance | {response_time:.1f}s")
        
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
        keywords = [word for word in expected_lower.split() if len(word) > 3]
        if not keywords:
            return 0.5  # Default score if no meaningful keywords
        
        matches = sum(1 for keyword in keywords if keyword in answer_lower)
        return matches / len(keywords)
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results"""
        print("\n" + "="*70)
        print("📊 MOCK AGENT EVALUATION RESULTS")
        print("="*70)
        
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
        
        # RAGAS-style metrics simulation
        print(f"\nRAGAS-Style Metrics (Simulated):")
        print(f"Faithfulness: {min(avg_relevance * 1.1, 1.0):.3f}")
        print(f"Answer Relevancy: {avg_relevance:.3f}")
        print(f"Context Precision: {min(avg_relevance * 0.9, 1.0):.3f}")
        print(f"Context Recall: {min(avg_relevance * 0.95, 1.0):.3f}")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 70)
        for i, resp in enumerate(results['responses'], 1):
            status = "✅" if resp['success'] else "❌"
            print(f"{i}. {status} {resp['relevance_score']:.0%} relevance | {resp['response_time']:.1f}s")
            print(f"   Q: {resp['question']}")
            print(f"   A: {resp['answer'][:100]}{'...' if len(resp['answer']) > 100 else ''}")
            if resp.get('expected'):
                print(f"   E: {resp['expected'][:80]}{'...' if len(resp['expected']) > 80 else ''}")
            print()
        
        print("="*70)
        print("NOTE: This is a MOCK evaluation for demonstration.")
        print("Install and configure a real LLM for actual evaluation.")
        print("="*70)


async def main():
    """CLI interface for mock evaluation"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_mock.py <test_file.json>")
        print("\nThis is a MOCK evaluator that doesn't require a real LLM.")
        print("Use this to test the evaluation framework.")
        return
    
    test_file = sys.argv[1]
    
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading test file: {e}")
        return
    
    # Run evaluation
    evaluator = MockEvaluator()
    results = await evaluator.evaluate_questions(test_data)
    
    evaluator.print_results(results)


if __name__ == "__main__":
    asyncio.run(main())
