# One-shot reasoning prompts

**User:** Anonymous (zazula@me.com)  
**Created:** 8/31/2025 20:01:53  
**Updated:** 9/1/2025 3:13:12  
**Exported:** 9/1/2025 3:13:24  
**Link:** [https://chatgpt.com/g/g-p-685204e87c308191aecc9582c309d0dc/c/68b50c9c-ca44-832a-92fa-3f64d395adc0](https://chatgpt.com/g/g-p-685204e87c308191aecc9582c309d0dc/c/68b50c9c-ca44-832a-92fa-3f64d395adc0)  

## Prompt:
we are adding some one-shot reasoning functions to our custom cli (https://github.com/zazula/library -- look at python/gpt_toolkit/cli).

develop industry leading prompts for the following one-shot reasoning commands:

/think <prompt> - One-shot high reasoning for this prompt only
/research <topic> - Deep research mode with maximum reasoning
/analyze <subject> - Analytical mode with structured thinking"""

## Response:
To craft industry-leading prompts for your one-shot reasoning commands, could you please clarify a few things:

1.  What is the expected output style or format for each command? For example, should `/think` return a concise answer, `/research` return a structured report, and `/analyze` return a breakdown with arguments and counterpoints?
2.  Are there specific topics, domains, or use cases that these commands are optimized for (e.g., product strategy, code review, scientific analysis)?
3.  Do you want the prompts to be general-purpose templates users can adapt, or prefilled with specific examples for demonstration?

Once I have this information, I can tailor the prompts for precision, clarity, and high-value reasoning performance.

## Prompt:
/think should be adaptive based on the query - if it deems it appropriate, it can shift into /research or /analyze...  should we have one that is a pipeline? or allow /research llm prompt generation /analyze our prompts and /think about a future design then /plan the work?

## Response:
Got it. I’ll explore industry-leading prompt strategies for your CLI one-shot reasoning commands—`/think`, `/research`, and `/analyze`—as well as evaluate whether it makes sense to introduce a pipeline or chained mode that intelligently transitions across commands (e.g., from `/research` to `/analyze` to `/plan`). I’ll also explore best practices for adaptive behavior in CLI tools like yours.

I’ll get back to you with a structured recommendation.

Research completed in 6m · 14 sources · 26 searches

Prompts and Workflow Strategy for One-Shot Reasoning CLI

Developing One-Shot Reasoning Prompts for CLI Commands
======================================================

**Overview:** We propose high-quality prompt templates for three new CLI reasoning commands – **`/think`**, **`/research`**, and **`/analyze`**. Each command guides the LLM to a different reasoning depth and style. Below, we detail the goal of each mode and present an “industry-leading” prompt that maximizes the model’s reasoning abilities in one shot. These prompts are designed as system or prefix instructions that the CLI can prepend to the user’s query.

`/think` – Adaptive High-Level Reasoning
----------------------------------------

**Purpose:** The `/think` command produces a one-shot answer with **high-level reasoning**, adapting its depth to the query complexity. For simple questions, it gives a concise direct answer. For complex or open-ended questions, it engages in deeper reasoning or even a mini-analysis, all within a single response.

**Prompt Objectives:** The prompt should instruct the model to:

*   **Adapt to Query Complexity:** Determine if the query is straightforward or complex. If straightforward, answer succinctly; if complex, **reason step-by-step internally** and provide a detailed answer.
*   **Encourage Internal Reasoning:** Think through the problem carefully before responding, ensuring the answer is well-founded. _(We want the model to simulate a chain-of-thought without always exposing it, unless explanation is needed.)_ Notably, prompting the model to “think step by step” can significantly improve reasoning on complex questions[rohan-paul.com](https://www.rohan-paul.com/p/how-to-improve-the-reasoning-ability#:~:text=Technical%20breakdown%3A%20In%20practice%2C%20CoT,a%20%E2%80%9Csurprising%20jump%20in%20performance%E2%80%9D).
*   **Stay Coherent and On-Point:** Deliver a single, coherent answer addressing the query. If the model did internal analysis, the final output should synthesize that reasoning into a clear response.

**Proposed Prompt Template (for system message or prefix):**

```text
You are a highly intelligent and adaptive reasoning AI. Carefully consider the user's question and solve it with the appropriate depth:

- If the question is straightforward or asks for a fact, provide a concise and direct answer.
- If the question is complex, open-ended, or ambiguous, **think through it step by step** (internally) and then provide a thorough, well-reasoned answer. Break down the problem into sub-problems if needed, analyze each part, and then synthesize a solution.
- Always ensure your final answer is clear, correct, and fully addresses the user's prompt. 

*(Think deeply about the problem before answering. Do not reveal your internal reasoning steps unless they are needed to explain your answer.)*
```

This prompt makes the `/think` command **adaptive**. By instructing the model to decide how much reasoning to apply, we leverage chain-of-thought techniques on harder queries (which research shows improves accuracy on complex tasks[rohan-paul.com](https://www.rohan-paul.com/p/how-to-improve-the-reasoning-ability#:~:text=Technical%20breakdown%3A%20In%20practice%2C%20CoT,a%20%E2%80%9Csurprising%20jump%20in%20performance%E2%80%9D)), without unnecessary verbosity for simple prompts. The emphasis on _internal_ step-by-step thinking guides the model to use higher-order reasoning covertly, yielding a well-founded answer.

`/research` – Deep Research Mode (Maximum Reasoning)
----------------------------------------------------

**Purpose:** The `/research` command triggers **deep-dive reasoning**. It puts the model in an extensive research mindset to produce a comprehensive, exploratory answer on the given topic. The model should use **maximum reasoning**, effectively utilizing a large token budget to explore the topic from multiple angles.

**Prompt Objectives:** The prompt should encourage the model to:

*   **Be Comprehensive:** Perform an in-depth exploration of the topic. This means covering background context, core details, relevant examples, and even multiple perspectives. The answer should read like a well-researched report or essay on the topic.
*   **Utilize Full Reasoning Capacity:** Take advantage of the maximum reasoning tokens available (if using a model/setting that supports this). Essentially, the model should not cut short its chain of thought – it should **exhaustively reason** about the topic. Expect a longer answer with rich detail.
*   **Stay Structured and Coherent:** Even though the response will be long and detailed, it should remain organized (e.g. use paragraphs or sections for subtopics) so that it’s readable and logical.

**Proposed Prompt Template:**

```text
You are now in "Deep Research" mode: a thorough expert researcher with unlimited time and knowledge. Your task is to deeply investigate and explain the given topic.

- **Depth & Breadth:** Provide a comprehensive analysis of the topic, covering all relevant background, details, and nuances. Include definitions of key terms, important data or facts, and multiple viewpoints if applicable. Leave no significant aspect unexplored.
- **Reasoning:** Use your full reasoning capabilities to examine the topic from different angles. If the topic involves a problem or question, consider various approaches or solutions and evaluate them. It’s okay to think step-by-step and discuss intermediate reasoning as part of the answer.
- **Evidence & Examples:** Where helpful, include evidence, examples, or brief case studies to support points. (If an exact source is not available, rely on logical justification or commonly accepted knowledge.)
- **Clarity:** Organize your response into coherent sections or paragraphs with clear explanations. Even though the answer will be detailed, ensure it remains well-structured and easy to follow.

Now, begin your deep research on the topic and present your findings:
```

When the user provides a topic to `/research`, this prompt pushes the model to produce an **exhaustive, well-reasoned essay-style answer**. By explicitly allowing step-by-step exploration in the answer, we invite the model to “show its work” as it builds the explanation (a strategy known to improve multi-hop reasoning and correctness[rohan-paul.com](https://www.rohan-paul.com/p/how-to-improve-the-reasoning-ability#:~:text=Technical%20breakdown%3A%20In%20practice%2C%20CoT,a%20%E2%80%9Csurprising%20jump%20in%20performance%E2%80%9D)). The model effectively acts like a scholarly assistant, delivering **maximum reasoning** output. The use of multiple angles and inclusion of examples ensures the answer isn’t just long, but also informative and evidence-backed.

`/analyze` – Analytical Mode with Structured Thinking
-----------------------------------------------------

**Purpose:** The `/analyze` command yields an **analytical breakdown** of a given subject. This mode is for structured, critical thinking – the model should dissect the subject into parts, examine each systematically, and provide a clear, organized analysis (often as lists or sections). The result should feel like a structured report or detailed outline.

**Prompt Objectives:** The prompt should instruct the model to:

*   **Adopt an Analytical Mindset:** Approach the subject methodically. This often means identifying key components, factors, or dimensions of the subject and addressing each one. For example, if asked to analyze a problem, the model might discuss its causes, effects, possible solutions, and their pros/cons.
*   **Produce Structured Output:** The answer must be well-organized. It should use **headings, bullet points, or numbered lists** to delineate parts of the analysis. This structured format makes it easy to follow complex reasoning. (Prompt engineering best practices suggest that asking for bullet points or subheadings yields more comprehensive and clearer answers[multimodal.dev](https://www.multimodal.dev/post/llm-prompting#:~:text=Structuring%20the%20prompts%20helps%20increase,each%20part%20of%20your%20input).)
*   **Be Detailed yet Focused:** Each part of the analysis should contain insightful details, but all details should relate to the overall subject. Avoid going off on tangents – maintain focus on analyzing the given subject deeply and objectively.

**Proposed Prompt Template:**

```text
You are in "Analytical" mode, acting as a critical analyst. Break down the user’s subject and examine it with structured, logical reasoning. 

Your analysis should include:
- **Key Components or Aspects:** Identify the main components, factors, or sections of the subject. For each component, provide a focused examination.
- **Structured Format:** Present the analysis in a clear structure. Use headings or bullet points for different sections or factors. For example, you might use bullet points, numbered lists, or subheadings to organize ideas (e.g., "1. Factor A – explanation... 2. Factor B – explanation...", or separate sections with titles).
- **Depth of Insight:** For each part of the subject, give insightful details. Explain the significance, implications, or cause-and-effect relationships. If comparing options or considering pros/cons, clearly list them under that section.
- **Conclusion/Synthesis:** End with a brief conclusion or summary of your analysis, synthesizing the insights from the parts into an overall understanding.

Remember to keep the analysis well-structured and systematic. Now, analyze the subject step by step:
```

Using this prompt, the `/analyze` command will generate a **well-structured dissection** of the topic. The explicit instructions to use headings and lists guide the model to output in a neatly organized format – an approach known to enhance clarity and completeness in LLM responses[multimodal.dev](https://www.multimodal.dev/post/llm-prompting#:~:text=Structuring%20the%20prompts%20helps%20increase,each%20part%20of%20your%20input). The user will see a breakdown of the topic into logical sections, each analyzed in depth. Such structured thinking not only improves readability but also ensures the model doesn’t skip key facets of the subject.

Adaptive Pipeline Consideration: Combining Modes vs Separate Commands
---------------------------------------------------------------------

You also asked whether a single pipeline command might be useful, where the AI could automatically shift between these modes (or sequentially invoke them) for complex tasks. This is an intriguing idea, essentially orchestrating a multi-step reasoning process within one command. Here are some thoughts:

*   **Unified Pipeline Command (e.g. `/solve` or `/plan`):** Instead of the user manually calling `/research` then `/analyze` then `/think`, a pipeline command could do it automatically. For example, given a broad problem, the tool could:
    1.  **Research Phase:** first use a deep research prompt to gather extensive information or brainstorm solutions (like internally calling the `/research` prompt).
    2.  **Analysis Phase:** then analyze or evaluate the gathered information (internally using an analytical prompt to compare options, find insights, etc.).
    3.  **Thinking/Planning Phase:** finally, synthesize a conclusion or a plan of action (similar to `/think`, or perhaps a dedicated `/plan` prompt to outline next steps).
    The output of the pipeline would be a composite result, e.g. a plan that’s backed by prior research and analysis done in the background. This could be presented either as a single integrated answer or as a sequence of sections.
*   **Benefits:** This approach mimics how a human expert might tackle a complex project – researching the topic, analyzing findings, then formulating a plan. It could yield very thorough results. In fact, recent research suggests letting the LLM generate and answer its own sub-questions can improve outcomes. For instance, _AutoReason_ (2024) automatically decomposes a query into sub-questions and solves those to build a better overall answer[rohan-paul.com](https://www.rohan-paul.com/p/how-to-improve-the-reasoning-ability#:~:text=There%20are%20also%20methods%20to,shot%20prompts%20on%20reasoning%20benchmarks). A pipeline command could leverage a similar idea, effectively doing a chain-of-thought _with_ intermediate queries and resolutions.
*   **Challenges:** Implementing a pipeline means the system must manage multi-step prompting behind the scenes. This requires careful design to pass context from one step to the next (possibly using the model’s output from research as input to analysis, etc.). It can also be time-consuming and token-expensive. Moreover, not every question needs the full pipeline – for many queries, a single-step `/think` (adaptive reasoning) might be sufficient. So the trigger for engaging the full pipeline needs consideration (perhaps the user explicitly calls it for very complex tasks).

**Recommendation:** Keep `/think`, `/research`, and `/analyze` as distinct one-shot commands for clarity and user control. However, you can certainly implement an advanced pipeline (maybe a command like `/solve` or `/plan`) that **orchestrates these modes** for the user. This pipeline command could be opt-in for complex problems where the user wants a comprehensive solution (research → analysis → plan). In designing it, re-use the prompts above in sequence. For example, the system might internally do something like:

1.  Run the `/research` prompt with the user’s topic to get a detailed information dump.
2.  Feed that output (or summary of it) into an `/analyze` prompt to get structured insights or comparisons.
3.  Finally, take those insights into a `/plan` prompt (which you could design similarly) to produce actionable steps or a conclusion.

Each step’s prompt can be crafted similarly to the ones we developed, possibly with slight adjustments to chain them smoothly. The end result to the user could either be a single combined answer or a clearly demarcated multi-part answer (e.g., _Research Findings_, _Analysis_, _Proposed Plan_ sections). This would indeed provide an **industry-leading approach** to complex queries, harnessing the full power of the LLM in a guided manner.

* * *

**Sources & Rationale:** We based the above prompt designs on known best practices in prompt engineering and the latest research on LLM reasoning:

*   Adaptive chain-of-thought prompting (“Let’s think step by step”) has been shown to significantly improve complex reasoning[rohan-paul.com](https://www.rohan-paul.com/p/how-to-improve-the-reasoning-ability#:~:text=Technical%20breakdown%3A%20In%20practice%2C%20CoT,a%20%E2%80%9Csurprising%20jump%20in%20performance%E2%80%9D). The `/think` prompt leverages this idea in a controlled way.
*   Deep reasoning with ample context (similar to few-shot or extended CoT) leads to more thorough answers[rohan-paul.com](https://www.rohan-paul.com/p/how-to-improve-the-reasoning-ability#:~:text=There%20are%20also%20methods%20to,shot%20prompts%20on%20reasoning%20benchmarks), which is the goal of `/research` mode.
*   Structured output improves clarity and completeness[multimodal.dev](https://www.multimodal.dev/post/llm-prompting#:~:text=Structuring%20the%20prompts%20helps%20increase,each%20part%20of%20your%20input), hence the emphasis on headings/lists in `/analyze` mode.
*   The pipeline concept draws on research like AutoReason for dynamic multi-step prompting[rohan-paul.com](https://www.rohan-paul.com/p/how-to-improve-the-reasoning-ability#:~:text=There%20are%20also%20methods%20to,shot%20prompts%20on%20reasoning%20benchmarks), aiming to combine the strengths of each mode.

By implementing these carefully crafted prompts, the CLI’s one-shot reasoning commands will encourage the LLM to perform at its best in each scenario – whether it requires a quick thoughtful answer, an in-depth exposition, or a well-structured analysis.

## Prompt:
could we leverage something like a HRM model to iterate a research->analysis->plan->REPEAT loop rather than having a waterfall approach?  what about meta deepconf?

## Response:
Great questions. To give you the most relevant insights, could you please clarify a couple of things:

1.  When you mention an "HRM model," are you referring to a Hierarchical Reinforcement Model, Human Reasoning Model, or something else entirely? And how do you envision it guiding or adapting the loop?
2.  For "meta deepconf," are you referencing DeepConf as in Facebook's data center network configuration automation framework, or are you using the term in another context (e.g., meta-configuration loops for research reasoning)?
3.  Is the goal to design an autonomous loop that revisits and refines the outputs of research → analysis → planning steps until a quality threshold is met? Or are you thinking of human-in-the-loop iterations?

Once I have a bit more detail on those terms and your goals, I can explore best practices or frameworks that match your vision.

## Prompt:
1. https://github.com/sapientinc/HRM, 2. https://ai.meta.com/research/publications/deep-think-with-confidence/, 3. autonomous

## Response:
Thanks for the clarification. I’ll explore how to use the HRM framework (from Sapient) and Meta’s DeepThink with Confidence strategy to design an autonomous research → analysis → planning loop that can iterate and refine itself intelligently, rather than follow a simple waterfall model.

I'll get back to you with a breakdown of how these approaches can be applied or adapted for your CLI reasoning workflows, including architectural and prompt design patterns.

Research cancelled

## Prompt:
could we leverage something like a HRM model to iterate a research->analysis->plan->REPEAT loop rather than having a waterfall approach?  what about meta deepconf?

## Response:
Just to confirm before I proceed—should the resulting CLI system be fully autonomous in driving the loop (research → analysis → planning → repeat) without user intervention, using internal evaluation and confidence to decide when to stop or iterate? Or would you like it to surface intermediate results for optional user approval at each stage?

## Prompt:
1. https://github.com/sapientinc/HRM, 2. https://ai.meta.com/research/publications/deep-think-with-confidence/, 3. autonomous. It would be great to have output along the way like we see "thinking" outputs in reasoning models

## Response:
Thanks, that's clear. I’ll now explore how to use Sapient's HRM and Meta's DeepThink with Confidence to create an autonomous reasoning engine that cycles through research → analysis → planning in a repeatable loop. I’ll include strategies to:

*   Trigger iterations based on internal confidence estimates or heuristic signals.
*   Generate interpretable "thinking" outputs during each stage.
*   Architect this loop for your CLI while preserving transparency.

I'll return with a detailed report outlining system behavior, prompt patterns, and control strategies.

Research completed in 4m · 8 sources · 18 searches

Designing Autonomous Reasoning Loops with HRM and DeepThink for CLI

Designing an Autonomous Reasoning Loop with HRM and DeepThink
=============================================================

**Introduction:**  
Modern AI reasoning systems often follow a fixed _waterfall_ approach – performing research, then analysis, then planning, in one pass. However, recent advances suggest we can build a more intelligent loop that **iteratively** refines its reasoning until confident in the result. This report explores how to leverage Sapient’s **Hierarchical Reasoning Model (HRM)** and Meta’s **Deep Think with Confidence (DeepConf)** to create an autonomous CLI tool that cycles through _research → analysis → plan_ repeatedly. The goal is a system that **self-decomposes tasks, evaluates its own progress, and decides when it’s confident enough to stop**, all while providing the user with transparent “thinking” updates.

Sapient’s Hierarchical Reasoning Model (HRM)
--------------------------------------------

**Hierarchical Planning and Execution:** Sapient’s HRM is a brain-inspired AI architecture that uses _two interdependent recurrent modules operating at different timescales_[medium.com](https://medium.com/@danny_54172/hierarchical-reasoning-model-9244a7ae4533#:~:text=The%20paper%20introduces%20the%20Hierarchical,that%20operate%20at%20different%20timescales). Specifically, it consists of:

*   **High-Level Module:** Handles slow, abstract planning (the “big-picture” reasoning).
*   **Low-Level Module:** Handles fast, detailed computations (the step-by-step problem solving).

This design lets HRM break complex tasks into a hierarchy of sub-tasks: the high-level planner sets objectives or subgoals, and the low-level module executes the fine-grained steps to achieve them[medium.com](https://medium.com/@danny_54172/hierarchical-reasoning-model-9244a7ae4533#:~:text=The%20paper%20introduces%20the%20Hierarchical,that%20operate%20at%20different%20timescales). The two modules work in tandem, exchanging information at each cycle. This **modular decomposition** of reasoning makes HRM’s approach fundamentally different from a single-step Chain-of-Thought; it is more structured and robust.

**Adaptive Reasoning Depth (ACT):** HRM can dynamically adjust how many thinking cycles it uses via _Adaptive Computation Time (ACT)_[medium.com](https://medium.com/@danny_54172/hierarchical-reasoning-model-9244a7ae4533#:~:text=,ACT)[medium.com](https://medium.com/@danny_54172/hierarchical-reasoning-model-9244a7ae4533#:~:text=%2A%20Hierarchical%20Convergence%3A%20Enables%20multi,%E2%80%9Cfast%20or%20slow%E2%80%9D%20reasoning%20dynamically). In essence, the model learns when to stop or continue its reasoning: simple problems may only need a few cycles, whereas harder ones trigger many cycles of the high-level and low-level loop. Internally, HRM’s high-level module produces a “halt” signal (learned through a Q-learning objective) indicating if the model believes it has reached a solution or should keep going. This means **HRM performs self-evaluation at runtime** – it doesn’t always run a fixed number of steps, but rather _adapts the depth of computation until the task is solved or a max limit is hit_. This adaptive halting mechanism lets HRM achieve _deep reasoning without collapse_ while being efficient[medium.com](https://medium.com/@danny_54172/hierarchical-reasoning-model-9244a7ae4533#:~:text=,ACT).

**Notable Results:** Despite its modest size (~27M parameters trained on only ~1K examples per task), HRM matches or outperforms much larger models on complex reasoning benchmarks (ARC-AGI, extreme Sudoku, mazes)[medium.com](https://medium.com/@danny_54172/hierarchical-reasoning-model-9244a7ae4533#:~:text=Despite%20its%20modest%20size%20,convergence%20strategy%20called%20hierarchical%20convergence)[medium.com](https://medium.com/@danny_54172/hierarchical-reasoning-model-9244a7ae4533#:~:text=%2A%20Top,dynamically%20using%20Adaptive%20Computation%20Time). It achieves this by **structuring the reasoning process** instead of relying on massive scale. These results underscore that _architectural intelligence_ – dividing tasks hierarchically and allocating “fast or slow” thinking as needed – can yield high reasoning performance[medium.com](https://medium.com/@danny_54172/hierarchical-reasoning-model-9244a7ae4533#:~:text=,ACT). For our purposes, HRM’s key contributions are: (1) **Modular task decomposition** (planner vs. solver roles), and (2) **Internal self-evaluation/halting** (deciding when it’s done). We will map these ideas onto the CLI reasoning loop design.

Meta’s Deep Think with Confidence (DeepConf)
--------------------------------------------

**Parallel Reasoning with Confidence Filtering:** DeepConf is a test-time inference method that improves LLM reasoning by combining _parallel thinking_ (generating multiple solution paths) with **confidence-based filtering**[arxiv.org](https://arxiv.org/html/2508.15260v1#:~:text=with%20Confidence%20,compared%20to%20full%20parallel)[arxiv.org](https://arxiv.org/html/2508.15260v1#:~:text=We%20introduce%20Deep%20Think%20with,or%20improving%20final%20answer%20accuracy). The core idea is to use the model’s own internal confidence signals to weed out poor reasoning traces and decide whether more thought is needed. In traditional _self-consistency_, an LLM might generate many answer paths and pick the most common answer (majority vote). DeepConf instead looks at _how confident the model was for each reasoning trace_ and uses that to make smarter decisions:

*   **Model-Internal Confidence Signals:** DeepConf leverages metrics like token probability distributions. For example, it monitors **token entropy** (how uncertain the model is at each step) and **token confidence** (essentially the inverse of entropy – how peaked the probability distribution is)[arxiv.org](https://arxiv.org/html/2508.15260v1#:~:text=We%20define%20token%20confidence%20as,tokens%20at%20position). By aggregating these at the trace level, one can compute an **overall confidence score** for a given reasoning path[arxiv.org](https://arxiv.org/html/2508.15260v1#:~:text=). Higher average confidence correlates strongly with correct solutions[arxiv.org](https://arxiv.org/html/2508.15260v1#:~:text=). Intuitively, if the model generated an answer with consistently high-confidence at each step (low uncertainty), that answer is likely correct; conversely, if it was “guessing” with high entropy in many steps, the answer is suspect.
*   **Filtering & Early Stopping:** Using these signals, DeepConf **filters out low-quality reasoning traces either during or after generation**[arxiv.org](https://arxiv.org/html/2508.15260v1#:~:text=We%20introduce%20Deep%20Think%20with,confidence%20reasoning%20traces%20either). In _offline mode_, the system can generate many complete solutions and then discard those with low confidence before voting or choosing a final answer. In _online mode_, it can even **stop a reasoning thread early** if confidence drops too much mid-way[arxiv.org](https://arxiv.org/html/2508.15260v1#:~:text=However%2C%20global%20confidence%20measures%20present,reasoning%20breakdowns%20that%20occur%20at)[arxiv.org](https://arxiv.org/html/2508.15260v1#:~:text=In%20this%20section%2C%20we%20present,performance%20by%20evaluating%20and%20aggregating). This prevents wasting time on a path that the model itself finds dubious. The result is a big boost in efficiency _and_ accuracy: for instance, DeepConf (with 512 parallel traces) achieved **99.9% accuracy on the challenging AIME 2025 math benchmark** while generating **84.7% fewer tokens** than naive parallel sampling[arxiv.org](https://arxiv.org/html/2508.15260v1#:~:text=hyperparameter%20tuning%20and%20can%20be,compared%20to%20full%20parallel%20thinking)[arxiv.org](https://arxiv.org/html/2508.15260v1#:~:text=In%20offline%20mode%20with%20access,1%20highlights%20our%20key%20results). In other words, it nearly saturates performance by focusing on _quality over quantity_ of reasoning paths.

**Implications for an Autonomous Loop:** DeepConf demonstrates that an AI can _judge its own uncertainty_ and use that to decide whether to think more. Rather than blindly doing exhaustive reasoning or stopping after a fixed effort, the process becomes **confidence-guided**: continue reasoning _only as long as needed_. For our CLI tool, we take inspiration from this: after each cycle of reasoning, the system should **estimate its confidence in the current answer/plan**. If confidence is low or uncertainty is high, the system knows it should “dig deeper” – perhaps research further or consider alternative reasoning. If confidence is high, it can move forward to finalize the answer. The key mechanism is that _the decision to continue or stop is dynamic and data-driven (here, driven by the model’s own assessment of its answer)_[arxiv.org](https://arxiv.org/html/2508.15260v1#:~:text=with%20Confidence%20,compared%20to%20full%20parallel).

Designing the Autonomous Reasoning Loop (Application Design)
------------------------------------------------------------

Using the principles above, we design a CLI reasoning system that **iteratively self-improves** instead of doing one-pass reasoning. The loop will explicitly mirror HRM’s hierarchical reasoning **within each iteration** and use DeepConf-style confidence checks **between iterations**:

*   **Hierarchical “Research–Analyze–Plan” Cycle:** Each iteration consists of three phased sub-tasks, analogous to HRM’s modules:  
    **(1) Research (Low-Level detail)** – The system gathers information needed for the problem. This could involve searching a knowledge base or recalling facts (in an automated agent, this might be a web search or database query). It’s akin to HRM’s fast module: detailed operations to fetch or compute specifics.  
    **(2) Analysis (High-Level reasoning)** – The system interprets the information and evaluates progress towards the goal. It combines new data with what’s already known, identifies any gaps or contradictions, and updates its understanding of the problem. This is a more abstract, reflective step, similar to HRM’s high-level planner reasoning about the big picture.  
    **(3) Plan (High-Level action)** – Based on the analysis, the system formulates the next step. This could be a plan for another research query, a decision to execute some solution step, or drafting the final answer if it believes it’s ready. Essentially, the plan is the output of the high-level module’s thinking for this cycle: either a next sub-goal for further investigation or a conclusion.
    These three phases collectively achieve **modular task decomposition**. The _research_ phase handles granular detail gathering, while _analysis/plan_ handle strategic reasoning. By separating them, the system can manage complexity more effectively – much like HRM separates concerns between two modules. Each cycle functions like a self-contained reasoning “chunk” that advances the solution.
*   **Self-Evaluation and Confidence Check:** After each research–analysis–plan cycle, the system performs a **self-evaluation**, inspired by DeepConf and HRM’s halting logic. At this point, the system asks itself: _“Am I confident in the solution or plan I have now? Have I resolved the question sufficiently?”_ This can be implemented in multiple ways. For example:
    *   The system could use an _LLM prompt to assess confidence_, having the model critique its current answer or identify if any unknowns remain.
    *   Alternatively, if using a model API that provides probabilities, the tool could analyze the model’s token-level confidence for signs of uncertainty (e.g. high entropy in its last answer)[arxiv.org](https://arxiv.org/html/2508.15260v1#:~:text=We%20define%20token%20confidence%20as,tokens%20at%20position).
    *   Another approach is running _multiple reasoning trials_ internally: if they all converge on the same answer, confidence is high; if they diverge, more work is needed (a simplified form of self-consistency check).
    Drawing from DeepConf, the system doesn’t treat every reasoning attempt equally – it weighs their quality. For instance, if the model’s answer was accompanied by a lot of hedging (“maybe”, “I think”) or the analysis phase noted unresolved questions, that indicates **low confidence**. The CLI should then decide to **“dig deeper”**, i.e. loop for another iteration to gather more info or consider alternative angles. Conversely, if the analysis finds the answer solid and the model’s internal confidence is high, the system can **proceed to finalize** the result. This dynamic loop continues until the stopping criteria are met: typically _“confidence >= threshold”_ or a max number of loops to avoid infinite cycling.
*   **Stopping Criterion (Termination):** The loop will terminate when either the system is sufficiently confident in its answer or an optional safety cap (like max iterations) is reached. “Sufficiently confident” could mean the model’s self-evaluation states nothing important is missing, or a computed confidence score exceeds a preset threshold (analogous to HRM’s learned halt signal reaching a certain level[GitHub](https://github.com/sapientinc/HRM/blob/05dd4ef795a98c20110e380a330d0b3ec159a46b/models/hrm/hrm_act_v1.py#L260-L269)). The final output is then presented to the user as the answer/solution, backed by all the reasoning done so far.

This design ensures the system **reflects and reevaluates at each step**, much like a human problem-solver checking their work. It replaces the linear waterfall with a feedback loop: each plan feeds into the next research phase if needed, gradually converging on a correct answer. It’s essentially an implementation of _“think until you’re sure”_. By modularizing the thinking (HRM style) and only moving on when sure (DeepConf style), the CLI can tackle complex queries more reliably than a single-pass approach.

CLI Implementation Recommendations
----------------------------------

To turn the above design into a working CLI tool, we propose the following architecture and strategies:

*   **Modular Agent Architecture:** Internally structure the CLI tool as an **agent with multiple skills or subroutines** corresponding to the phases. For example, implement separate functions or prompt templates for: `do_research(query)`, `analyze_context(context)`, `plan_next_step(context)`. Each function can invoke an LLM or other services as needed. This separation mirrors the HRM modules, making the system’s code easier to maintain and extend. The high-level controller (the “manager” loop) will invoke these in sequence for each iteration.
*   **State and Memory:** Maintain an **iteration state** that carries forward useful information. This state might include the original question, any facts gathered so far, intermediate conclusions, and the latest plan. In each cycle, the _research_ component can append new findings to the state; the _analysis_ component can update the understanding of the problem, and the _plan_ component can either produce the final answer or identify what to do next. Essentially, the CLI should build an internal “scratchpad” of notes – analogous to the hidden state HRM carries across cycles, but here implemented as an explicit memory in the program. This prevents losing information between iterations and allows the reasoning to accumulate.
*   **Confidence Evaluation Mechanism:** Implement a dedicated step or function for **confidence evaluation** after each cycle. For instance, after the `plan_next_step` function produces an outcome, run a quick check like `evaluate_confidence(current_answer, context)`. This could be a small LLM prompt asking: “Do we have high confidence in the above answer? Yes/No and why? If no, what is missing?” – effectively a self-critique. Another implementation could utilize a scoring heuristic: if the plan output contains uncertain language or if certain key aspects of the question haven’t been addressed in the analysis, mark confidence as low. If using an advanced LLM with logit access, compute an approximate confidence score (e.g., average token confidence as in DeepConf) for the answer[arxiv.org](https://arxiv.org/html/2508.15260v1#:~:text=). The CLI can then use this evaluation to decide looping.
*   **Decision Logic:** The control flow in the CLI loop will look something like:
    ```
    state = initialize(question)
    loop_count = 0
    max_loops = N  (to prevent infinite loops)
    do {
        loop_count += 1
        info = do_research(state)          # Low-level detail gathering
        update state with info
        insights = analyze_context(state)  # High-level reasoning on gathered info
        update state with insights
        outcome = plan_next_step(state)    # High-level plan or answer formulation
        update state with outcome
        confidence = evaluate_confidence(state)  # How good is our outcome?
    } while (confidence is low AND loop_count < max_loops)
    present outcome to user
    ```
    Each iteration uses the prior state (which includes all previous knowledge and partial conclusions) so that the reasoning deepens cumulatively. The loop breaks out once `confidence` is no longer low. This logic ensures **autonomy** – after the user poses the question, the system drives the entire process without further user input, until it’s ready to deliver an answer.
*   **Transparent “Thinking” Outputs:** To keep the user informed _without overwhelming them_, the CLI should produce structured interim messages at each stage of each iteration. Instead of dumping raw LLM thoughts (which could be verbose or confusing), format succinct updates. For example:
    *   After a research step, print something like: **`[Research]`** _“Found relevant data on topic X and Y.”_
    *   After analysis, **`[Analysis]`** _“It appears that X is a key factor, but Z remains uncertain.”_
    *   After planning, **`[Plan]`** _“Next, will look into Z to fill the gap.”_ (or, if final, _“Proposed solution ready.”_).
    Using clear tags or headings (Research/Analysis/Plan) helps the user follow the agent’s progress at a glance. Each message should be a concise summary (one to three sentences) highlighting the result of that phase, rather than the entire internal reasoning. This way, the user sees the **chain-of-thought _outline_** without the unnecessary verbosity. It builds trust, as the user can verify the agent’s direction, and it provides transparency as required. But by filtering out low-level detail, it keeps the output readable. This approach is analogous to how one might narrate their problem-solving steps to someone else: high-level updates, not every stray thought.
*   **Mapping HRM/DeepConf to Code:** We can draw a loose parallel between our implementation and the HRM/DeepConf concepts for clarity: The `do_research` (low-level action) and `analyze/plan` (high-level reasoning) mimic HRM’s two-tier loop, albeit implemented via sequential function calls rather than a single neural network. The `evaluate_confidence` step plays the role of HRM’s halting unit and DeepConf’s filter – it tells the system whether to continue (like HRM’s decision to do another cycle, or DeepConf spawning another reasoning trace) or halt. By structuring code this way, we make the system’s operation conceptually aligned with those proven ideas. In practice, this means easier debugging and extension: e.g., one could improve the confidence evaluator independently (perhaps by training a classifier to predict answer correctness) and plug it in without disturbing the rest of the loop.
*   **Prompt Engineering and Tool Use:** If the CLI integrates with external tools (web search APIs, calculators, etc.), those can be invoked during the _research_ or _analysis_ phases as needed. For instance, if the question needs factual lookup, the `do_research` step might call a search API and then have the LLM summarize the findings. The prompts for each phase should be designed to elicit the right behavior: the research prompt should instruct the model to generate search queries or request data; the analysis prompt should ask for reasoning given the collected facts; the planning prompt should ask “Given the analysis, what should we do next or conclude?”. (We detail example prompts in the next section.) By separating these intents, we avoid one giant prompt trying to do everything. This makes the model’s job easier and the outputs more focused at each step.
*   **Error Handling & Loop Limits:** In an autonomous loop, it’s important to handle failures gracefully. The CLI should detect if it’s looping without making progress (e.g. confidence never improves, or it runs into the same state repeatedly). Incorporating a max loop count (say, 5 iterations) prevents endless loops. If the max is hit, the system might present whatever answer it has with a note of uncertainty or ask the user for guidance. Logging each iteration’s outcome internally can help the system notice if it’s stuck in a cycle. These safeguards ensure the tool remains practical and does not hang indefinitely.

Prompt Strategy and Templates for Each Phase
--------------------------------------------

Designing effective prompts for each reasoning phase is crucial. Each phase has a distinct role, so we craft prompts to clearly delimit those roles. Below are suggested prompt structures for the _Research_, _Analysis_, _Plan_, and _Self-Evaluation_ steps, with notes on how they operate:

*   **Research Prompt:** This prompt should instruct the AI to gather facts or evidence related to the sub-problem at hand. It often will be combined with a tool (like a web search). For example:
    _Template:_ _“Search for information on **\[specific aspect\]**. Focus on finding data or answers for **\[question or gap\]**. Provide any relevant details.”_
    _Usage:_ If the main question is, say, solving a technical problem, and the analysis determined that we need a formula or definition, the research prompt narrows down that need. E.g., _“Find the official definition of the HRM framework and how it handles intermediate steps.”_ The AI (or an integrated search agent) then returns with snippets or facts. The CLI should then condense results to the key points. The output from this prompt populates the knowledge base for analysis. It’s important that the prompt keeps the search scoped (to avoid irrelevant info) and possibly asks for sources so analysis can verify them.
*   **Analysis Prompt:** This prompt asks the AI to reason about all the information gathered so far in context of the original question. It should synthesize and identify what the information means for the problem.
    _Template:_ _“Given the question **\[restate user’s problem\]** and the information gathered (listed below), analyze how this knowledge contributes to a solution. Identify any insights, and note if any questions remain unanswered.”_
    _Usage:_ The analysis step’s output might be a short summary of findings and their implications. For example, _“The HRM framework uses a high-level and low-level module to break down reasoning[medium.com](https://medium.com/@danny_54172/hierarchical-reasoning-model-9244a7ae4533#:~:text=The%20paper%20introduces%20the%20Hierarchical,that%20operate%20at%20different%20timescales), which implies our CLI agent should split planning and execution. However, we still need to determine how the agent will measure its confidence (perhaps using internal token probabilities as in DeepConf)[arxiv.org](https://arxiv.org/html/2508.15260v1#:~:text=We%20define%20token%20confidence%20as,tokens%20at%20position).”_ This shows an insight (the approach to split tasks) and a remaining gap (how to measure confidence). The prompt encourages the model to be both **integrative** (connect facts to the problem) and **critical** (spot what’s missing). By explicitly asking for remaining questions, we set up the next iteration’s goal.
*   **Plan Prompt:** The planning prompt directs the AI to decide on the next action or to articulate the solution if ready. It takes the analysis into account. There can be two modes for this prompt:
    *   _Next-Step Planning:_ If analysis found open issues, the prompt should formulate a strategy to resolve them (essentially preparing the next research target or reasoning step).
    *   _Final Answer Formulation:_ If analysis concludes all necessary info is in hand, the prompt should formulate the final answer or solution outline.
    _Template (next-step):_ _“Based on the analysis, outline a plan for the next step. What should we do next to move closer to the solution? (If information is missing, specify what to research or clarify.)”_
    _Template (final answer):_ _“Formulate the final answer to the original question based on the analysis. Ensure all parts of the question are addressed confidently.”_
    _Usage:_ Continuing our example, if analysis flagged low confidence in how to measure uncertainty, a next-step plan might say, _“Next, we should investigate methods to quantify model confidence during reasoning (perhaps from the DeepConf approach). I will research how DeepConf extracts confidence signals.”_ This planned action then feeds into the next loop’s research phase. On the other hand, if analysis concluded all pieces (HRM task splitting and DeepConf confidence measure) are understood, the plan might be a summary answer: _“Now I will compile the design: The CLI will have a loop with research, analysis, plan steps, using HRM’s hierarchy for task breakdown and DeepConf’s confidence metric to decide when to stop.”_ – which is essentially the final output content. The plan prompt ensures the system explicitly decides whether to continue or finalize at this juncture.
*   **Self-Evaluation (Confidence Check) Prompt:** Although this may be handled partly by code logic, it can be useful to have the AI reflect explicitly. A prompt for this could ask the model to rate or justify its confidence.
    _Template:_ _“Review the proposed solution/plan above. On a scale or in words, how confident are you that this answers the question completely and correctly? If not fully confident, what aspect should be examined next?”_
    _Usage:_ This can be given as an internal query to the model right after the planning step. For example, the model might respond, _“I am mostly confident, but not entirely. I haven’t addressed how exactly to implement the confidence threshold – that might need clarification.”_ The CLI can parse this and realize another iteration (to research how to implement a confidence threshold) is needed. If the model instead says, _“I am confident that all key points have been addressed,”_ the loop can terminate. This natural language self-critique is an analogue to DeepConf’s numeric confidence score, albeit less precise. It leverages the model’s own reasoning to detect gaps. (In practice, combining this with a numeric heuristic – e.g., checking if the plan prompt had uncertain language – can strengthen the decision.)

**Prompt Chaining and Context:** Each successive prompt should include necessary context from previous steps. For instance, the analysis prompt should include the question and a bulleted list of facts from research. The planning prompt should include a summary of the analysis conclusions. By feeding the right information into each phase’s prompt, we ensure the LLM has the context it needs, without exposing irrelevant details. This is similar to how HRM passes the state between high and low modules, but here we do it through prompt memory.

**Clarity in Outputs:** All prompts should encourage concise, structured answers (since we will turn those into the “\[Research\] …”, “\[Analysis\] …” messages). You might instruct the model in the prompt to answer with a brief list of key points or a short paragraph, depending on the phase. Keeping outputs short helps in presenting the interim results to the user cleanly. It also makes it easier to scan for confidence indicators (it’s hard to gauge confidence from a rambling paragraph). In essence, the prompts themselves contribute to **“hiding unnecessary verbosity”** by requesting focused responses.

Conclusion
----------

By blending HRM’s hierarchical approach and DeepConf’s confidence-driven control, our CLI tool will behave as an **autonomous reasoning agent**: it decomposes complex problems into manageable parts, iteratively solves them, and knows when it has done enough. The high-level/low-level division from HRM gives our system a powerful way to organize its thoughts (planning vs. executing), and the confidence evaluation from DeepConf ensures it doesn’t stop prematurely or go in circles without cause. The end result is a CLI that doesn’t just dump an answer, but actively _thinks_ in front of the user – showing a trail of insightful, structured “thinking” outputs – until it reaches a well-founded solution.

By implementing the design pattern and prompt strategies outlined above, developers can create a reasoning loop that is **modular, transparent, and self-correcting**. This not only improves the quality of answers (especially for difficult queries) but also builds user trust, as the system can explain how it arrived at an answer and when it’s sure of it. In summary, the combination of Sapient’s HRM and Meta’s DeepConf concepts leads to a next-generation CLI assistant that reasones more like an expert human: breaking problems down and only concluding once every important detail has been checked with confidence[arxiv.org](https://arxiv.org/html/2508.15260v1#:~:text=with%20Confidence%20,compared%20to%20full%20parallel)[medium.com](https://medium.com/@danny_54172/hierarchical-reasoning-model-9244a7ae4533#:~:text=The%20paper%20introduces%20the%20Hierarchical,that%20operate%20at%20different%20timescales).



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)