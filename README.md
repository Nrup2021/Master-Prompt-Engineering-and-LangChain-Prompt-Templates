# Master Prompt Engineering and LangChain Prompt Templates

## Overview

This repository contains learning notes and practical examples covering Prompt Engineering, In-Context Learning, advanced prompting techniques, and LangChain Prompt Templates.

The goal is to understand how to design effective prompts and build reusable LLM-powered applications for tasks such as text summarization, question answering, text classification, and code generation.

## Learning Objectives

- Understand the fundamentals of Prompt Engineering.
- Learn zero-shot, one-shot, and few-shot prompting.
- Understand Chain-of-Thought (CoT) and self-consistency techniques.
- Create reusable prompt templates with LangChain.
- Build processing pipelines using LangChain Expression Language (LCEL).
- Understand how language models process inputs and generate outputs.
- Explore practical LLM applications using Python.

## Table of Contents

1. [Introduction to Prompt Engineering](#1-introduction-to-prompt-engineering)
2. [In-Context Learning](#2-in-context-learning)
3. [Prompting Techniques](#3-prompting-techniques)
4. [Important LLM Parameters](#4-important-llm-parameters)
5. [Introduction to LangChain](#5-introduction-to-langchain)
6. [Prompt Templates](#6-prompt-templates)
7. [LangChain Expression Language (LCEL)](#7-langchain-expression-language-lcel)
8. [Applications of Prompting](#8-applications-of-prompting)
9. [Key Technical Terms](#9-key-technical-terms)
10. [Example Workflow](#10-example-workflow)
11. [Best Practices](#11-best-practices)
12. [Key Takeaways](#12-key-takeaways)

---

## 1. Introduction to Prompt Engineering

**Prompt Engineering** is the process of designing and refining instructions given to a Large Language Model (LLM) to generate useful and relevant responses.

A prompt can contain:

- Instructions describing the task.
- Context or background information.
- Examples of expected input and output.
- Constraints and rules.
- The desired response format.

### Example

**Basic prompt:**

```text
The future of artificial intelligence is
```

**Structured prompt:**

```text
Explain the future of artificial intelligence
in simple English.

Provide:
1. Three important trends.
2. Real-world examples.
3. A short conclusion.
```

A structured prompt makes the task and expected output clearer.

## 2. In-Context Learning

**In-Context Learning** is the ability of an LLM to use instructions and examples provided directly in a prompt to perform a task.

It does not require updating the model's underlying weights.

For example, you can provide customer-support examples and ask the model to classify a new customer message into the same categories.

### Example

```text
Message: I want my money back.
Category: Refund

Message: Where is my package?
Category:
```

The model can use the demonstrated pattern to predict the category of the new message.

## 3. Prompting Techniques

### 3.1 Basic Prompting

A basic prompt provides a short phrase or simple instruction.

**Example:**

```text
Explain artificial intelligence.
```

**Use case:** Exploring a model's capabilities or asking straightforward questions.

### 3.2 Zero-Shot Prompting

Zero-shot prompting asks the model to perform a task without providing examples in the prompt.

**Example:**

```text
Classify the following customer message
as Refund, Shipping, or Product Info.

Message: Where is my package?

Return only the category.
```

**Expected output:**

```text
Shipping
```

**Use case:** Classification, translation, summarization, and straightforward question answering.

### 3.3 One-Shot Prompting

One-shot prompting provides one example to demonstrate the expected task or output format.

**Example:**

```text
Example:
English: Good morning.
French: Bonjour.

Now translate this sentence into French:

English: Where is the nearest supermarket?
French:
```

**Expected output:**

```text
Où se trouve le supermarché le plus proche ?
```

**Use case:** Teaching a model a desired format or style using one example.

### 3.4 Few-Shot Prompting

Few-shot prompting provides multiple examples to demonstrate a pattern.

**Example:**

```text
Message: I love this product.
Sentiment: Positive

Message: This product is terrible.
Sentiment: Negative

Message: The product arrived yesterday.
Sentiment: Neutral

Message: The battery lasts all day.
Sentiment:
```

**Expected output:**

```text
Positive
```

**Use case:** Classification tasks, structured extraction, and specialized output formats.

### 3.5 Chain-of-Thought (CoT) Prompting

Chain-of-Thought prompting encourages a model to approach a problem in a structured, step-by-step manner.

**Example:**

```text
A store has 22 apples.
It sells 15 apples and receives 8 more.

Calculate the final number of apples.
Explain the calculation briefly.
```

**Calculation:**

```text
22 - 15 = 7
7 + 8 = 15

Final answer: 15 apples.
```

**Use case:** Multi-step calculations, logical problems, and tasks that benefit from structured analysis.

### 3.6 Self-Consistency

Self-consistency involves generating multiple independent candidate solutions and comparing them to identify a consistent answer.

**Example workflow:**

1. Generate several candidate solutions.
2. Compare their final answers.
3. Check the assumptions and calculations.
4. Select the answer best supported by the evidence.

**Use case:** Tasks where checking multiple approaches may improve reliability.

Self-consistency does not guarantee correctness. Multiple answers can share the same error.

### 3.7 Role Prompting

Role prompting specifies the role or expertise the model should adopt.

**Example:**

```text
Act as an experienced Python developer.

Explain REST APIs to a beginner.
Use simple language and provide one example.
```

**Use case:** Technical tutoring, writing assistance, coding support, and task-oriented chatbots.

---

## 4. Important LLM Parameters

| Parameter | Meaning | Common purpose |
|---|---|---|
| `max_new_tokens` | Maximum number of new tokens generated in the response. | Control response length. |
| `temperature` | Influences the randomness of token generation. | Lower values often produce more predictable outputs. |
| `top_k` | Restricts sampling to the top K candidate tokens. | Limit the candidate pool. |
| `top_p` | Restricts sampling to candidates within a cumulative probability threshold. | Control sampling diversity. |

These settings influence generation behavior. Their effects depend on the model and its implementation.

For classification and other consistency-sensitive tasks, a lower temperature is often a useful starting point.

---

## 5. Introduction to LangChain

**LangChain** is a framework for developing applications powered by language models.

It provides components for defining prompts, calling models, processing outputs, and connecting application steps.

Common components used in this tutorial include:

- `PromptTemplate`
- `RunnableLambda`
- `StrOutputParser`
- Language model integrations
- LCEL processing chains

The tutorial uses an IBM watsonx.ai model integration for its examples. Other compatible model providers can be used with appropriate configuration.

## 6. Prompt Templates

A **Prompt Template** is a reusable prompt containing placeholders for dynamic values.

Instead of rewriting the prompt for every request, you define it once and provide different inputs.

### Example

```python
from langchain_core.prompts import PromptTemplate

template = """
Explain {topic} in {style} language.
Provide one practical example.
"""

prompt = PromptTemplate.from_template(template)

formatted_prompt = prompt.format(
    topic="REST APIs",
    style="simple"
)

print(formatted_prompt)
```

**Output:**

```text
Explain REST APIs in simple language.
Provide one practical example.
```

### Important concepts

| Concept | Meaning |
|---|---|
| Template | Reusable prompt structure. |
| Placeholder | A variable such as `{topic}`. |
| Formatting | Replacing placeholders with actual values. |
| Reusability | Using the same prompt for different inputs. |

### Benefits

- Reduces repeated code.
- Keeps instructions consistent.
- Makes prompts easier to maintain.
- Supports reusable application workflows.
- Separates prompt design from input data.

---

## 7. LangChain Expression Language (LCEL)

**LCEL** is LangChain's approach to composing runnable components into a processing pipeline.

It uses the pipe operator (`|`) to connect components.

### Typical processing flow

```text
Input Data
    |
    v
Prompt Template
    |
    v
Language Model
    |
    v
Output Parser
    |
    v
Final Response
```

### Example

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

template = """
Summarize the following content in two sentences:

{content}
"""

prompt = PromptTemplate.from_template(template)

# Assumes that `llm` is an initialized,
# supported LangChain language model.
chain = prompt | llm | StrOutputParser()

response = chain.invoke({
    "content": (
        "Artificial intelligence is transforming "
        "healthcare, education, and transportation."
    )
})

print(response)
```

### How it works

1. The input dictionary supplies the `content` variable.
2. The prompt template creates the complete instruction.
3. The language model generates a response.
4. `StrOutputParser()` returns the output as a string.
5. `invoke()` executes the chain and returns the result.

**Note:** The example assumes that the model has been initialized with valid provider credentials and configuration.

---

## 8. Applications of Prompting

### 8.1 Text Summarization

Summarization converts longer text into a shorter version while preserving the main points.

**Example prompt:**

```text
Summarize the following content in three bullet points.

Content:
{content}
```

**Use cases:**

- Document summarization.
- Meeting notes.
- Research summaries.
- News summaries.

### 8.2 Question Answering

Question answering generates responses to questions using supplied information.

**Example prompt:**

```text
Answer the question using only the provided context.

Context:
{content}

Question:
{question}

If the answer is not available in the context,
state that the information is insufficient.
```

**Use cases:**

- Documentation assistants.
- Customer-support bots.
- Internal knowledge assistants.
- Educational question-answering tools.

### 8.3 Text Classification

Text classification assigns text to predefined categories.

**Example prompt:**

```text
Classify the following message into exactly
one category:

Categories:
- Refund
- Shipping
- Product Info

Message:
{text}

Return only the category.
```

**Use cases:**

- Customer-support routing.
- Sentiment analysis.
- Feedback categorization.
- Email classification.

### 8.4 Code Generation

Code generation converts natural-language requirements into code.

**Example prompt:**

```text
Generate an SQL query to retrieve the names
and email addresses of customers who made
a purchase in the last 30 days.

Assume the database contains:
- customers(id, name, email)
- purchases(customer_id, purchase_date)

Use a join between the tables.
Return only the SQL query.
```

Generated code should be reviewed and tested before being used in production.

### 8.5 Role-Playing Applications

Role prompting configures a model to answer according to a specified role and tone.

**Example prompt:**

```text
Role: Mobile Development Mentor
Tone: Friendly and educational

Question:
{question}

Explain the answer with a practical example.
```

**Use cases:**

- Programming tutors.
- Interview preparation assistants.
- Technical documentation helpers.
- Interactive learning applications.

### 8.6 Product Review Analysis

A review-analysis prompt can request sentiment, features, and a concise summary.

**Example prompt:**

```text
Analyze the following product review:

{review}

Return:
- Sentiment: Positive, Negative, or Neutral
- Key Features: List the mentioned features
- Summary: One sentence
```

This pattern is useful for analyzing customer feedback at scale.

---

## 9. Key Technical Terms

| Term | Definition |
|---|---|
| AI | Artificial Intelligence; technology that performs tasks associated with human intelligence. |
| LLM | Large Language Model; a model that processes and generates language. |
| Prompt | Input or instruction supplied to a model. |
| Prompt Engineering | Designing prompts to guide model behavior and output. |
| In-Context Learning | Using instructions and examples in a prompt without updating model weights. |
| Model Weights | Learned numerical values used by a trained model. |
| Token | A unit of text processed by a language model. |
| Hallucination | Incorrect or unsupported information generated by an AI model. |
| Context | Information supplied to help the model answer a question. |
| Prompt Template | Reusable prompt with placeholders. |
| Runnable | A component that can be executed within a LangChain workflow. |
| `RunnableLambda` | A LangChain component that wraps a Python function as a runnable. |
| LCEL | LangChain Expression Language for composing processing pipelines. |
| Pipe Operator | Connects runnable components in sequence. |
| `invoke()` | Executes a runnable with supplied input. |
| Output Parser | Processes model output into a desired representation. |
| `StrOutputParser` | Returns model output as a string. |
| Temperature | A generation parameter influencing sampling variability. |
| Generalization | Applying learned patterns to new examples or tasks. |
| Model ID | Identifier used to select a particular model. |

---

## 10. Example Workflow

Consider a customer-support application that classifies incoming messages.

### Input

```text
Where is my package?
```

### Step 1: Define the task

```text
Classify the customer message into one category:
Refund, Shipping, or Product Info.
```

### Step 2: Add the message

```text
Customer message:
Where is my package?
```

### Step 3: Generate the response

```text
Shipping
```

### Step 4: Integrate into an application

A practical application could:

1. Receive a message from a customer.
2. Send the message and instructions to the model.
3. Validate the returned category.
4. Route the message to the appropriate support team.
5. Record the result for monitoring and evaluation.

In a production system, validate model outputs, handle API errors, protect sensitive data, and monitor classification quality.

---

## 11. Best Practices

### Write clear instructions

Specify the task, context, constraints, and expected output.

### Choose the appropriate prompting technique

- Use zero-shot prompting for straightforward tasks.
- Use one-shot or few-shot prompting when examples help clarify the expected result.
- Use structured prompts when consistent output is important.
- Consider multiple candidate solutions for tasks that benefit from verification.

### Control the output format

Request a specific format, such as JSON, a table, a list, or a short paragraph.

### Ground answers in context

For question-answering applications, provide relevant source information and instruct the model not to invent missing details.

### Validate the output

Do not assume every generated response is correct. Validate categories, JSON structures, SQL queries, and other outputs before using them.

### Separate prompt logic from application logic

Use prompt templates to organize instructions and application code to manage validation, error handling, and business rules.

### Test with different inputs

Evaluate expected cases, edge cases, ambiguous messages, and invalid inputs.

### Keep credentials secure

Store API keys and other secrets in environment variables or a suitable secrets manager rather than committing them to GitHub.

---

## 12. Key Takeaways

- Prompt Engineering helps guide LLM responses through well-designed instructions.
- In-Context Learning uses prompt examples without retraining the model.
- Zero-shot, one-shot, and few-shot prompting differ in the number of examples provided.
- Chain-of-Thought and self-consistency are techniques for approaching and checking more complex tasks.
- LangChain Prompt Templates make prompts reusable and maintainable.
- LCEL connects prompt, model, and output-processing components into a workflow.
- Common LLM applications include summarization, Q&A, classification, and code generation.
- Reliable applications require testing, output validation, error handling, and appropriate safeguards.

## Learning Resources

- [LangChain Documentation](https://docs.langchain.com/)
- [LangChain Prompt Templates](https://python.langchain.com/docs/concepts/prompt_templates/)
- [IBM Prompt Engineering](https://www.ibm.com/think/topics/prompt-engineering)
- [IBM Zero-Shot Prompting](https://www.ibm.com/think/topics/zero-shot-prompting)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

---

## Repository Purpose

This repository serves as a practical reference for learning Prompt Engineering and LangChain fundamentals, preparing for AI-related technical interviews, and developing LLM-powered applications using Python.

**Topics:** Prompt Engineering, Generative AI, LLM, In-Context Learning, LangChain, Prompt Templates, LCEL, Python, Text Summarization, Question Answering, Text Classification, Code Generation.
