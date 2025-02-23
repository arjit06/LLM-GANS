# CASPER

## Preparation of environent 

## Executing the code
CASPER process executor is a menu driven program which takes the input of all the required variables from the user and then calls the CASPER main program seperately for summarization, improvement and evaluation. <br><br>
The following is a snapshot of the file in execution:- <br>
![Screenshot 2025-02-22 234241](https://github.com/user-attachments/assets/14bb14f0-9620-419d-84a6-74ffa32586fc)
<br> 
Events can take values from (1-1 to 1-15) <br>
n is the number of iterations for which the planning engine is supoosed to work. It can take values from n=0 (the 0-shot case, where no improvement happens) to any number.  <br>
The summarization LLM options are Llama 3.1/GPT-4o Mini <br>
The improvement LLM optinos are Phi 4/GPT-4o Mini/Phi 3 Mini/Gemma 2/Deepseek <br>
The evaluation LLM is fixed as Phi 4 <br>
It is important to type the name of the LLMs exactly as given in the options (case and space sensitive) to prevent any exceptions <br> 
The Generated summaries will be saved in the file './Generated_summaries_{sum_prefix}/{testimony_code}_{attribute_for_saving_file_name}.txt' (eg: './Generated_summaries_Llama_3.1/E1T1_Event_Details.txt' <br> 
The Improved summaries will be saved in the file './Generated_summaries_{sum_prefix}_sum_{improve_prefix}_eval_and_imp/{summary_path_prefix}_p{i+1}.txt' (eg: 'Generated_summaries_Llama_3.1_sum_GPT-4o_Mini_eval_and_imp/E1T1_Event_Details_p1.txt' <br> 
The evaluation log will be saved in the file './evaluation_log_{sum_prefix}_sum_{improve_prefix}_eval_and_imp.txt' (eg: './evaluation_Llama_3.1_sum_GPT-4o_Mini_eval_and_imp.txt')



