# LLM GANS 
## LLM GAN-based-Summarizer  
Witness testimonies are crucial in criminal investigations, providing context that physical evidence alone cannot. However, they are often verbose, fragmented, and inconsistent, making it difficult to extract a coherent and reliable account. Systematically distilling these testimonies into structured, fact-based summaries is essential for accurate and unbiased investigations.

This project processes interrogator-interrogatee conversations to generate three controlled summaries: event details (chronological sequence of events), facts (factual statements by the witness), and character description (physical details of the perpetrator). We employ an LLM-based framework for efficient summarization.

A zero-shot LLM first generates attribute-specific summaries from raw testimonies. Then, a GAN-inspired refinement process iteratively improves them. An evaluator LLM assesses summaries based on custom criteria—checking for hallucinations, relevance, and attribute-specific accuracy—assigning a score. This score, along with evaluation feedback, guides an improver LLM, which refines the summaries. This adversarial setup enhances summary quality without requiring training or fine-tuning, enabling domain-specific, controlled summarization for any dataset.

## Preparation of Environment

Create a conda environment GANS using the `environment.yml` file provided using the following command:
```
conda env create -f environment.yml --name GANS
```

Activate the environment:
```
conda activate GANS
```

Install PyTorch according to the CUDA version of the GPU you are using:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Now you are ready to execute the code.

## Executing the Code

GANS process executor is a menu-driven program that takes the input of all required variables from the user and then calls the GANS main program separately for summarization, improvement, and evaluation.

### Execution Snapshot

Below is a snapshot of the file in execution:

![Screenshot 2025-03-30 130851](https://github.com/user-attachments/assets/79f0b8e4-3d7d-4d13-a3e9-2aba5dcbbf1b)


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
  ```
  ./Generated_summaries_{sum_prefix}/{testimony_code}_{attribute_for_saving_file_name}.txt
  ```
  **Example:**
  ```
  ./Generated_summaries_Llama_3.1/E1T1_Event_Details.txt
  ```

#### Improved Summaries
- Improved summaries will be saved in:
  ```
  ./Generated_summaries_{sum_prefix}_sum_{improve_prefix}_eval_and_imp/{summary_path_prefix}_p{i+1}.txt
  ```
  **Example:**
  ```
  ./Generated_summaries_Llama_3.1_sum_GPT-4o_Mini_eval_and_imp/E1T1_Event_Details_p1.txt
  ```

#### Evaluation Files
- Evaluation logs will be saved in:
  ```
  ./evaluation_logs_{sum_prefix}_sum_{improve_prefix}_eval_and_imp/{summary_path_prefix}_eval.txt
  ```
  **Example:**
  ```
  ./evaluation_logs_Llama_3.1_sum_GPT-4o_Mini_eval_and_imp/E1T1_Event_Details_eval.txt
  ```

