# AgentAlpha Evaluation Guide

**Made by Rodrigo de Sarasqueta**

Comprehensive guide for evaluating AgentAlpha performance using RAGAS and custom metrics.

## Available Test Datasets

### 1. `test_questions.json` (Basic - 5 questions)
Simple questions for basic functionality testing.

**Example:**
```json
{
  "question": "What is AgentAlpha?",
  "ground_truth": "AgentAlpha is a minimal RAG agent..."
}
```

### 2. `test_questions_simple.json` (Simple - 8 questions)  
Short golden answers, good for quick validation.

**Categories covered:**
- Basic functionality
- LLM providers
- Document processing
- Deployment
- Technical details

**Performance with MockAgent:** ~56% relevance

### 3. `test_questions_comprehensive.json` (Advanced - 12 questions)
Detailed technical questions with comprehensive golden answers.

**Categories covered:**
- `basic_functionality` - Core concepts
- `technical_architecture` - Implementation details  
- `llm_providers` - Provider configuration
- `document_processing` - RAG system details
- `deployment` - Docker and production
- `tools_system` - Available tools
- `api_endpoints` - REST API details
- `configuration` - Settings and config
- `evaluation` - Evaluation capabilities
- `edge_case` - Error handling
- `performance` - Limitations and characteristics
- `troubleshooting` - Common issues

**Difficulty levels:** easy, medium, hard

**Performance with MockAgent:** ~10% relevance (expected - answers are very detailed)

## Evaluation Systems

### 1. Mock Evaluation (No LLM required)
```bash
python evaluate_mock.py test_questions_simple.json
```

**Features:**
- Works without active LLM
- Simulates RAGAS-style metrics
- Good for testing evaluation framework
- Keyword-based relevance scoring

### 2. Simple Evaluation (Real LLM required)
```bash
python evaluate_simple.py test_questions.json
```

**Features:**
- Uses actual AgentAlpha
- Measures response time
- Keyword-based relevance
- Retrieval quality assessment

### 3. Full RAGAS Evaluation (Python 3.10+ required)
```bash
python evaluate.py test_questions.json
```

**Features:**
- Complete RAGAS metrics
- Faithfulness scoring
- Answer relevancy
- Context precision/recall
- Professional evaluation report

## Creating Custom Test Data

### Basic Format
```json
[
  {
    "question": "Your question here?",
    "ground_truth": "Expected answer with key terms"
  }
]
```

### Advanced Format (Comprehensive)
```json
[
  {
    "category": "your_category",
    "question": "Technical question?",
    "ground_truth": "Detailed technical answer with specific terminology",
    "difficulty": "easy|medium|hard"
  }
]
```

### Golden Answer Guidelines

**For High Relevance Scores:**
- Include key terms that would appear in agent responses
- Use terminology consistent with the codebase
- Keep answers focused and specific

**For Comprehensive Testing:**
- Cover all major functionality areas
- Include edge cases and error scenarios
- Test different difficulty levels
- Use technical details from actual implementation

## Metrics Explanation

### Success Rate
Percentage of questions that received successful responses (no errors).

### Response Time
Average time for agent to process and respond to questions.

### Relevance Score
Keyword-based matching between agent response and golden answer:
- 1.0 (100%) = Perfect keyword match
- 0.7+ (70%+) = Good relevance  
- 0.5+ (50%+) = Fair relevance
- <0.5 (<50%) = Poor relevance

### RAGAS Metrics (when available)

**Faithfulness** - How well the answer sticks to retrieved context
**Answer Relevancy** - How relevant the answer is to the question
**Context Precision** - Quality of retrieved context chunks
**Context Recall** - Coverage of relevant information in context

## Interpreting Results

### Assessment Levels
- 🟢 **Excellent**: >80% relevance, >90% success rate
- 🟡 **Good**: >60% relevance, >80% success rate  
- 🟠 **Fair**: >40% relevance, >60% success rate
- 🔴 **Needs Improvement**: <40% relevance or <60% success rate

### Expected Performance

**MockAgent Results:**
- Simple dataset: ~50-60% relevance (good for mock)
- Comprehensive dataset: ~10-20% relevance (expected - very detailed answers)

**Real Agent Results (estimated):**
- Simple dataset: ~70-85% relevance
- Comprehensive dataset: ~45-65% relevance
- Basic dataset: ~60-75% relevance

## Best Practices

1. **Start with Simple Tests** - Use basic dataset first
2. **Gradual Complexity** - Move to comprehensive after basic validation
3. **Regular Testing** - Run evaluations after code changes
4. **Document Baselines** - Record initial performance metrics
5. **A/B Testing** - Compare different LLM providers or configurations
6. **Monitor Degradation** - Watch for performance drops over time

## Troubleshooting Evaluation

### Low Relevance Scores
- Check if golden answers match agent's vocabulary
- Verify agent has access to relevant documents  
- Ensure LLM is responding appropriately
- Consider adjusting similarity thresholds

### Evaluation Errors
- Verify LLM connectivity (for real evaluations)
- Check file paths and permissions
- Ensure required dependencies are installed
- Review configuration settings

### Performance Issues
- Reduce test dataset size for faster iteration
- Use mock evaluation for development
- Optimize vector store settings
- Monitor system resources

---

This evaluation framework provides comprehensive testing capabilities for AgentAlpha, enabling continuous improvement and quality assurance.
