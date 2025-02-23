# CASPER

## Preparation of Environment

Create a conda environment CASPER using the `environment.yml` file provided using the following command:
```
conda env create -f environment.yml --name CASPER
```

Activate the environment:
```
conda activate CASPER
```

Install PyTorch according to the CUDA version of the GPU you are using:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Now you are ready to execute the code.

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

