# LLM Prompt Evolution

An automated system for iteratively improving LLM prompts through error analysis and performance evaluation.

## Overview

This project implements an intelligent prompt optimization system that:
- Evaluates prompt performance on classification tasks
- Analyzes prediction errors to identify patterns
- Automatically generates improved prompts based on error analysis
- Iteratively refines prompts until accuracy threshold is reached

## Features

- **Automated Prompt Evaluation**: Tests prompts against labeled data to measure accuracy
- **Error Analysis**: Identifies common patterns in misclassifications
- **Intelligent Prompt Revision**: Uses LLM to analyze errors and generate improved prompts
- **Iterative Improvement**: Continues refinement until desired accuracy is achieved
- **Result Tracking**: Saves all iterations and improvements to JSON for analysis

### Parameters

The `Main()` function accepts:
- `MaxIterations` (default: 5): Maximum number of improvement iterations
- `AccuracyThreshold` (default: 0.8): Target accuracy to stop iterations

### Example Output

```
Initial Accuracy: 0.625
--- Iteration 1 ---
Analyzing errors and generating revised prompt...
Revised Accuracy: 0.750
Improvement: 0.125
--- Iteration 2 ---
Revised Accuracy: 0.812
Improvement: 0.062
Accuracy threshold 0.800 reached! Stopping iterations.
Results Saved
```

## How It Works

1. **Initial Evaluation**: Tests the starting prompt on training data
2. **Error Analysis**: Examines misclassified examples to identify patterns
3. **Prompt Revision**: Uses LLM to generate improved prompt based on error analysis
4. **Re-evaluation**: Tests revised prompt and measures improvement
5. **Iteration**: Repeats until accuracy threshold is met or max iterations reached

## Output

Results are saved to `PromptTracing.json` containing:
- Iteration number
- Prompt text at each iteration
- Accuracy achieved

## Example Use Case

The included example demonstrates talent aspiration classification:
- Analyzes employee feedback statements
- Classifies as "has_aspiration" or "no_aspiration"
- Improves prompt to better distinguish between aspirational and non-aspirational statements
