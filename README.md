# CASPER

## Preparation of Environment

## Executing the Code

CASPER process executor is a menu-driven program that takes the input of all required variables from the user and then calls the CASPER main program separately for summarization, improvement, and evaluation.

### Execution Snapshot

Below is a snapshot of the file in execution:

![Execution Screenshot](https://github.com/user-attachments/assets/14bb14f0-9620-419d-84a6-74ffa32586fc)

### Input Parameters

- **Events**: Can take values from **1-1 to 1-15**.
- **n (Number of Iterations)**: Specifies how many iterations the planning engine should run.
  - **n = 0**: Represents the **0-shot case**, where no improvement happens.
  - **n > 0**: The engine will iteratively improve the summaries.

### LLM Options

- **Summarization LLM Options**: 
  - Llama 3.1
  - GPT-4o Mini

- **Improvement LLM Options**:
  - Phi 4
  - GPT-4o Mini
  - Phi 3 Mini
  - Gemma 2
  - Deepseek

- **Evaluation LLM**: 
  - Fixed as **Phi 4**

**Note:** It is important to type the LLM names **exactly as given** in the options (case and space sensitive) to prevent exceptions.

### Output Files

#### Generated Summaries
- Summaries will be saved in:
