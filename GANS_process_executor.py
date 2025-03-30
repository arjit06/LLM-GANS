import time
import subprocess
import argparse

def execute_processes(events_list=list(range(1,16)),attribute_list=['Event Details','Facts','Character Description'],max_n=3,cuda_id=0,summarization_model='Llama 3.1',improvement_model='Phi 4'): 
    testimony_codes=[]
    for event_num in events_list:
        for testimony_num in [1,2,3]:
                testimony_codes.append(f'E{event_num}T{testimony_num}')
                
    
    #run summarization
    mode='summarization'  
    subprocess.run(["python", "GANS_main.py", f"--events_list={events_list}" ,f"--cuda_id={cuda_id}", f"--mode={mode}", f"--summarization_model={summarization_model}", f"--improvement_model={improvement_model}"]) 
    
    #run improvement
    for testimony_code in testimony_codes:
        print('testimony_code:',testimony_code)
        subprocess.run(["python", "GANS_main.py", f"--testimony_code={testimony_code}", f"--attribute_list={attribute_list}", f"--max_n={max_n}", f"--cuda_id={cuda_id}", f"--summarization_model={summarization_model}" ,f"--improvement_model={improvement_model}"]) 
        
    #run evaluation   
    mode='evaluation'
    subprocess.run(["python", "GANS_main.py", f"--max_n={max_n}", f"--events_list={events_list}", f"--cuda_id={cuda_id}", f"--mode={mode}", f"--summarization_model={summarization_model}", f"--improvement_model={improvement_model}"]) 

print('Enter the value of all the following variables to execute the model:-')
events=input('Enter the list of events to be processed (e.g. 1-15): ')
max_n=int(input('Enter value of n for planning engine (e.g. 3): '))
cuda_id=int(input('Enter the GPU number (e.g. 0): '))
summarization_model=input('Enter the summarization model to be used (one out of Llama 3.1,GPT-4o Mini): ')
improvement_model=input('Enter the improvement model to be used (one out of Phi 4,GPT-4o Mini,Phi 3 mini,Gemma 2,Deepseek): ')

events_list=[int(i) for i in range(int(events.split('-')[0]),int(events.split('-')[1])+1)]
attribute_list=['Event Details','Facts','Character Description']
execute_processes(events_list,attribute_list,max_n,cuda_id,summarization_model,improvement_model)