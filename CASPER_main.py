from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
import transformers
import torch
import ast
import argparse
import json
import os

model_name,tokenizer,pipeline,=None,None,None
model_name2,tokenizer2,pipeline2=None,None,None
client=None
#default values of the models 
summarization_llm='Llama 3.1'
improvement_llm='GPT-4o Mini'
scoring_llm='Phi 4'
cuda_id=1
remove_labels=False

def load_llama_3_1():
    global model_name, tokenizer,pipeline,cuda_id
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16,
        device_map=f"cuda:{cuda_id}",
    )

def load_phi_4():
    global model_name2,tokenizer2,pipeline2,cuda_id
    model_name2="microsoft/phi-4"
    tokenizer2 = AutoTokenizer.from_pretrained(model_name2)
    pipeline2 = transformers.pipeline(
        "text-generation",
        model=model_name2,
        model_kwargs={"torch_dtype": "auto"},
        device_map=f"cuda:{cuda_id}",
    )
    
def load_phi_3_mini(): 
    global model_name2,tokenizer2,pipeline2,cuda_id
    model_name2="microsoft/Phi-3-mini-128k-instruct"
    tokenizer2 = AutoTokenizer.from_pretrained(model_name2) 
    model2 = AutoModelForCausalLM.from_pretrained( 
        model_name2,  
        device_map=f"cuda:{cuda_id}",  
        torch_dtype="auto",  
        trust_remote_code=True,  
    ) 
    pipeline2 = transformers.pipeline( 
        "text-generation", 
        model=model2, 
        tokenizer=tokenizer2
    )

def load_deepseek_distill_llama(): 
    global model_name2,tokenizer2,pipeline2,cuda_id
    model_name2="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    tokenizer2 = AutoTokenizer.from_pretrained(model_name2) 
    pipeline2 = transformers.pipeline("text-generation", model=model_name2,device=f'cuda:{cuda_id}')
    
def load_gpt_4o_mini():
    global client
    client = OpenAI(
        api_key="your api key"
    )
    
def load_gemma_2():
    global model_name2,tokenizer2,pipeline2,cuda_id
    model_name2="google/gemma-2-9b"
    tokenizer2 = AutoTokenizer.from_pretrained(model_name2) 
    pipeline2 = transformers.pipeline( 
        "text-generation", 
        model=model_name2, 
        device=f'cuda:{cuda_id}'
    )

def load_models(task='summarization'):
    global summarization_llm,improvement_llm,scoring_llm
    llm_loader_dict={'Llama 3.1':load_llama_3_1,'GPT-4o Mini':load_gpt_4o_mini,'Phi 4':load_phi_4,'Phi 3 mini':load_phi_3_mini,'Gemma 2':load_gemma_2, 'Deepseek':load_deepseek_distill_llama}
    
    if task=='summarization':
        llm_loader_dict[summarization_llm]()
    elif task=='improvement':
        llm_loader_dict[improvement_llm]()
    elif task=='scoring':
        llm_loader_dict[scoring_llm]()
      
def openai_generate(prompt, model="gpt-4o-mini", role="evaluator"):
    """Generates a response from OpenAI GPT-4o Mini"""
    system_content="You are a member of the Police Department tasked with evaluating the summary based on the evaluation criteria given to you"
    if role=="modifier":
        system_content="You are now an officer whose task is to modify and rewrite the provided summary using the evaluation from the Police Captain and the context."
    elif role=='summarizer': 
        system_content="You are now an analyzer, whose task can be to answer questions based on the context, to extract relevant content from the context or to generate a summary based on the context."
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_content},
                  {"role": "user", "content": prompt}],
        # temperature=1
    )
    return response.choices[0].message.content.strip()

# generate responses from [GPT-4o Mini, Phi 4, Deepseek] using system and user messages
def llm_generate(prompt, role="evaluator"):
    system_content="You are a member of the Police Department tasked with evaluating the summary based on the evaluation criteria given to you"
    if role=="modifier":
        system_content="You are now an officer whose task is to modify and rewrite the provided summary using the evaluation from the Police Captain and the context."
    messages=[{"role": "system", "content": system_content},
                {"role": "user", "content": prompt}]
    police_officer_verdict_prompt_output = pipeline2(
                messages,
                max_new_tokens=800, #400 words
    )
    return police_officer_verdict_prompt_output

# retrieve the dialogue from the testimony files 
def get_dialogue(testimony_code='E1T1'):
    global remove_labels
    if remove_labels==False:
        f=open('./Testimonies_json/'+testimony_code+'.json')
        dialogue=f.read()
        f.close()
    else: 
        def remove_annotations(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            
            cleaned_text = ""
            
            for entry in data:
                interrogator = entry["Interrogator"]
                interrogee = entry["Interrogee"]
                cleaned_text += f"Interrogator: {interrogator}\nInterrogee: {interrogee}\n\n"
            
            return cleaned_text

        file_path = './Testimonies_json/'+testimony_code+'.json'
        dialogue = remove_annotations(file_path)
    return dialogue

def create_guiding_prompt(dialogue): 
    witness_testimony_prompt=dialogue+'''\n\nWhat are the important events that happened in this witness' testimony? 
Who all are involved in this incident according to the witness' testimony? 
When did the incident happen, according to the witness' testimony? 
Where did the incident happen, according to the witness' testimony?
How did the incident take place, according to the witness' testimony?
Please answer the above questions in one paragraph each and put an <end> after the last answer:\n\n'''
    return witness_testimony_prompt

def get_event_details_summarization_prompt(context): 
    new_prompt=context+'''\n\nExtract the Event Details and generate 7 questions which will be related to Event_Details. The definition of the Event Details is given as the chronological sequence of events that took place. Answer those questions and generate the summary of the answers in not more than 150 words. Please follow the framework given below:

Questions Based on Event Details:
{Your generated questions}

Answers
{Your generated answers}

Summary
{Your generated summary of the answers}

Always start the summary with "According to the witness" and end with <eos> '''
    return new_prompt 

def get_facts_summarization_prompt(context):
    new_prompt=context+'''\n\nExtract the Facts and generate 7 questions which will be related to Facts. The definition of the Facts is given as all the information that the witness is claiming or states. Answer those questions and generate the summary of the answers in not more than 150 words. Please follow the framework given below:

Questions Based on Facts:
{Your generated questions}

Answers
{Your generated answers}

Summary
{Your generated summary of the answers}

Always start the summary with "According to the witness" and end with <eos> '''
    return new_prompt 

def get_character_description_summarization_prompt(dialogue): 
    new_prompt='''The following is the testimony of an interviewee on an event that has happened. Extract the Character Description from the testimony, which is defined to be the names of the characters involved in the events, as well as their appearance described by the interviewee. It will also describe the relationship between the character if it is stated by the interviewee. In case you don't find such descriptions from the eyewitness, then you have to simply state the following: "No description was provided in the witness's testimony". 

Testimony:\n'''+dialogue+'''\n\nFollow the template given below:

Character Description:
{your generated text}

The Character Description should not be more than 50 words and should end with <eos>'''
    return new_prompt

#modify this acc to attribute
def create_attribute_specific_summarization_prompt(context, dialogue, attribute='Event Details'):
    if attribute=='Event Details':
        new_prompt=get_event_details_summarization_prompt(context)
    elif attribute=='Facts':
        new_prompt=get_facts_summarization_prompt(context)
    elif attribute=='Character Description':
        new_prompt=get_character_description_summarization_prompt(dialogue)
    return new_prompt

def get_event_details_criteria():
    criteria='''1. Coverage of Key Events
Evaluation Points:
The extent to which the summary captures all the critical events associated with the Event_Details attribute from the source dialogue.

2. Accuracy of Event Sequencing
Evaluation Points:
The correctness of the chronological order of events as described in the source dialogue.
Does the summary maintain logical progression without skipping or misordering key moments?

3. Clarity
Evaluation Points:
The clarity of the narrative and how well the events are conveyed without unnecessary details or verbosity.

4. Attribute Relevance
Evaluation Points:
How well the summary adheres to the Event_Details attribute, focusing exclusively on events rather than other aspects like character traits or factual details.
Are the described details primarily event-related, avoiding unnecessary character descriptions or broader factual context?
Does the summary maintain fidelity to the "what happened" aspect of the dialogue without introducing unrelated information?

5. Faithfulness to Source
Evaluation Points:
The alignment of the summary with the factual content of the annotated source dialogue.
Does the summary accurately represent the events described in the source?
Are there any factual distortions, omissions, or hallucinations?'''
    l=['Coverage of Key Events','Accuracy of Event Sequencing','Clarity','Attribute Relevance','Faithfulness to Source']
    return criteria,l

def get_facts_criteria():
    criteria='''1. Fact Coverage
Evaluation Points:
The extent to which the summary includes all relevant factual information from the source dialogue related to the Facts attribute.
Does the summary include all factual claims made by the characters?

2. Factual Consistency
Evaluation Points:
The alignment of factual details with the source content.
Are the facts presented accurate according to the dialogue?
Are there any facts which are missing according to the dialogue ?
Are any fabricated or incorrect facts included?

3. Clarity 
Evaluation Points:
The clarity and coherence with which the facts are presented.
Are the facts clearly and logically organized?
Does the summary avoid ambiguity or redundancy in presenting the facts?

4. Attribute Relevance
Evaluation Points:
How well the summary focuses on the Facts attribute, avoiding unrelated details such as event sequences or character descriptions.
Does the summary stick to factual assertions made by the characters?
Are unnecessary details about the sequence of events or actions minimized?

5. Factual Completeness 
Evaluation Points:
The balance between including all necessary facts and maintaining a concise narrative.
Are all critical facts included without omitting any key details?
Is the summary concise, avoiding over-explanation or repetition of facts?''' 
    l=['Fact Coverage','Factual Consistency','Clarity','Attribute Relevance','Factual Completeness']
    return criteria,l

def get_character_description_criteria():
    criteria='''1. Character Feature Coverage
Evaluation Points:
The extent to which the generated summary captures all relevant physical and distinguishing features of the characters described in the source dialogue.

2. Clarity 
Evaluation Points:
The clarity and precision with which the character descriptions are conveyed in the summary.
Is the relationship between characters clear where relevant ?

3. Attribute Focus
Evaluation Points:
The degree to which the summary exclusively focuses on physical and descriptive characteristics, avoiding unrelated details or actions.
Does the generated summary focus solely on character descriptions?
Does it avoid blending in actions, events, or behavioral traits unrelated to physical descriptions?

4. Factual Completeness
Evaluation Points:
The alignment of character descriptions in the summary with the source dialogue.
How well the generated summary includes all key descriptive details compared to the gold summary.
Are any fabricated details (e.g., incorrect shirt color or body type) included?
Does the generated summary omit any key descriptive features present in the gold summary?
'''
    l=['Character Feature Coverage','Clarity','Attribute Focus','Factual Completeness']
    return criteria,l

def get_evaluation_criteria(attribute='Event Details'):
    if attribute=='Event Details':
        return get_event_details_criteria()
    elif attribute=='Facts': 
        return get_facts_criteria()
    elif attribute=='Character Description':
        return get_character_description_criteria()
    
def get_closing_string_template(attribute='Event Details'):
    global improvement_llm
    l=get_evaluation_criteria(attribute)[1]
    if improvement_llm=='Phi 3 mini': closing_line='''Strictly follow the template given below, and always and put an <end> after the last evaluation:\n\n'''
    else: closing_line='''Follow the template given below:\n\n''' # [GPT 4o Mini, Phi 4, Deepseek]
    for i in range(len(l)):
        closing_line+=f'''{i+1}) {l[i]}:
Score:Your answer
Reasoning:Your answer\n\n'''
    if improvement_llm=='Phi 3 mini': closing_line+='''<end>'''
    else: closing_line+='''<eos>'''
    if improvement_llm=='Gemma 2': closing_line='''Evaluation with Score and Reasoning:\n\n'''
    return closing_line

def create_role_specific_verdict_prompt(summary,previous_verdict,testimony_code='E1T1',attribute='Event Details',role='Police Officer'): 
    previous_verdict_mention_string='You are also given the verdict of a Subordinate Officer which you can refer to while evaluating the summary, but you should not be influenced by it while providing your own verdict.'
    if role=='Police Officer': previous_verdict_mention_string=''
    role_assignment_string=f'You are now a {role} of the Police Department tasked with evaluating the summary based on the evaluation criteria given to you. {previous_verdict_mention_string}\n\n'
    attribute_assignment_string=f'Evaluate the summary given below using the evaluation criteria for {attribute} and the context for evaluation. The answer for the questions in evaluation criteria present in evaluation point should only be in the scale of 0 to 5, 0 being the least relevant and 5 being the highest. Here is the summary that you need to evaluate:\n\n'
    summary_string=f'Summary:\n{summary}\n\n'
    evaluation_criteria_string=f'Evaluation Criteria for {attribute}:\n'+get_evaluation_criteria(attribute)[0]+'\n\n'
    context_string=f'Context:\n'+get_dialogue(testimony_code)+'\n\n'
    previous_verdict_string=''
    if role!='Police Officer': previous_verdict_string=f'Verdict of a Subordinate Officer:\n{previous_verdict}\n\n'
    closing_string=get_closing_string_template(attribute)
    return role_assignment_string+attribute_assignment_string+summary_string+evaluation_criteria_string+context_string+previous_verdict_string+closing_string

def get_last_word(attribute): 
    if attribute=='Event Details':last_word="5. Faithfulness to Source"
    elif attribute=='Facts':last_word="5. Factual Completeness"
    elif attribute=='Character Description':last_word="4. Factual Completeness"
    return last_word

def get_first_word(attribute): 
    if attribute=='Event Details':first_word="Coverage"
    elif attribute=='Facts':first_word="Fact Coverage"
    elif attribute=='Character Description':first_word="Character Feature Coverage"
    return first_word

def get_police_officer_verdict(summary,previous_verdict,testimony_code='E1T1',attribute='Event Details',task='improvement'): 
    police_officer_verdict_prompt=create_role_specific_verdict_prompt(summary,previous_verdict,testimony_code=testimony_code,attribute=attribute,role='Police Officer')
  
    def run_police_officer_llm(llm):
        if llm in ['Phi 4','Deepseek']: 
            police_officer_verdict_prompt_output = llm_generate(police_officer_verdict_prompt)
            if llm=='Deepseek': police_officer_verdict=police_officer_verdict_prompt_output[0]['generated_text'][2]['content'].split('</think>')[1].split('<end>')[0].strip('\n')
            else: police_officer_verdict=police_officer_verdict_prompt_output[0]['generated_text'][2]['content'].split('<end>')[0].strip('\n')
            
        elif llm=='Gemma 2': 
            police_officer_verdict_prompt_output = pipeline2(police_officer_verdict_prompt,max_new_tokens=800)
            last_word=get_last_word(attribute)
            police_officer_verdict_split_list=police_officer_verdict_prompt_output[0]['generated_text'].split('Evaluation with Score and Reasoning:')[1].split(last_word)
            police_officer_verdict=police_officer_verdict_split_list[0] + last_word+police_officer_verdict_split_list[1].split('Reasoning:')[0] + 'Reasoning:'+police_officer_verdict_split_list[1].split('Reasoning:')[1].split('\n')[0]
            
        elif llm=='Phi 3 mini': 
            police_officer_verdict_prompt_output = pipeline2(police_officer_verdict_prompt,max_new_tokens=800)
            first_word=get_first_word(attribute)
            police_officer_verdict=first_word+police_officer_verdict_prompt_output[0]['generated_text'].split(first_word)[3].split('<end>')[0]
            
        elif llm=='GPT-4o Mini': 
            police_officer_verdict_prompt_output= openai_generate(police_officer_verdict_prompt)
            police_officer_verdict=police_officer_verdict_prompt_output.split('<end>')[0].strip('\n')
            
        else: 
            print('Invalid LLM')
        return police_officer_verdict
             
    if task=='improvement': police_officer_verdict=run_police_officer_llm(llm=improvement_llm)
    elif task=='scoring': police_officer_verdict=run_police_officer_llm(llm=scoring_llm)
    else: print('Invalid task')
    return police_officer_verdict

def get_inspector_verdict(summary,police_officer_verdict,testimony_code='E1T1',attribute='Event Details',task='improvement'):
    inspector_verdict_prompt=create_role_specific_verdict_prompt(summary,police_officer_verdict,testimony_code=testimony_code,attribute=attribute,role='Inspector')
    def run_inspector_llm(llm):
        if llm in ['Phi 4','Deepseek']: 
            inspector_verdict_prompt_output = llm_generate(inspector_verdict_prompt)
            if llm=='Deepseek': inspector_verdict=inspector_verdict_prompt_output[0]['generated_text'][2]['content'].split('</think>')[1].split('<end>')[0].strip('\n')
            else: inspector_verdict=inspector_verdict_prompt_output[0]['generated_text'][2]['content'].split('<end>')[0].strip('\n')
            
        elif llm=='Gemma 2':
            inspector_verdict_prompt_output = pipeline2(inspector_verdict_prompt,max_new_tokens=800)
            last_word=get_last_word(attribute)
            inspector_verdict_split_list=inspector_verdict_prompt_output[0]['generated_text'].split('Evaluation with Score and Reasoning:')[1].split(last_word)
            inspector_verdict=inspector_verdict_split_list[0] + last_word+inspector_verdict_split_list[1].split('Reasoning:')[0] + 'Reasoning:'+inspector_verdict_split_list[1].split('Reasoning:')[1].split('\n')[0]
            
        elif llm=='Phi 3 mini':
            inspector_verdict_prompt_output = pipeline2(inspector_verdict_prompt,max_new_tokens=800)
            first_word=get_first_word(attribute)
            inspector_verdict=first_word+inspector_verdict_prompt_output[0]['generated_text'].split(first_word)[4].split('<end>')[0]
            
        elif llm=='GPT-4o Mini': 
            inspector_verdict_prompt_output= openai_generate(inspector_verdict_prompt)
            inspector_verdict=inspector_verdict_prompt_output.split('<end>')[0].strip('\n')
            
        else: 
            print('Invalid LLM')
        return inspector_verdict
             
    if task=='improvement': inspector_verdict=run_inspector_llm(llm=improvement_llm)
    elif task=='scoring': inspector_verdict=run_inspector_llm(llm=scoring_llm)
    else: print('Invalid task')
    return inspector_verdict

def get_captain_verdict(summary,inspector_verdict,testimony_code='E1T1',attribute='Event Details',task='improvement'): 
    captain_verdict_prompt=create_role_specific_verdict_prompt(summary,inspector_verdict,testimony_code=testimony_code,attribute=attribute,role='Captain')
    def run_captain_llm(llm):
        if llm in ['Phi 4','Deepseek']: 
            captain_verdict_prompt_output = llm_generate(captain_verdict_prompt)
            if llm=='Deepseek': captain_verdict=captain_verdict_prompt_output[0]['generated_text'][2]['content'].split('</think>')[1].split('<end>')[0].strip('\n')
            else: captain_verdict=captain_verdict_prompt_output[0]['generated_text'][2]['content'].split('<end>')[0].strip('\n')
            
        elif llm=='Gemma 2':
            captain_verdict_prompt_output = pipeline2(captain_verdict_prompt,max_new_tokens=800)
            last_word=get_last_word(attribute)
            captain_verdict_split_list=captain_verdict_prompt_output[0]['generated_text'].split('Evaluation with Score and Reasoning:')[1].split(last_word)
            captain_verdict=captain_verdict_split_list[0] + last_word+captain_verdict_split_list[1].split('Reasoning:')[0] + 'Reasoning:'+captain_verdict_split_list[1].split('Reasoning:')[1].split('\n')[0]
            
        elif llm=='Phi 3 mini':
            captain_verdict_prompt_output = pipeline2(captain_verdict_prompt,max_new_tokens=800)
            first_word=get_first_word(attribute)
            captain_verdict=first_word+captain_verdict_prompt_output[0]['generated_text'].split(first_word)[4].split('<end>')[0]
            
        elif llm=='GPT-4o Mini': 
            captain_verdict_prompt_output= openai_generate(captain_verdict_prompt)
            captain_verdict=captain_verdict_prompt_output.split('<end>')[0].strip('\n')
            
        else: 
            print('Invalid LLM')
        return captain_verdict
             
    if task=='improvement': captain_verdict=run_captain_llm(llm=improvement_llm)
    elif task=='scoring': captain_verdict=run_captain_llm(llm=scoring_llm)
    else: print('Invalid task')
    return captain_verdict

def get_scores_from_verdict(text): 
    scores = []
    for line in text.splitlines():
        if 'Score:' in line:
            if '**' in line: 
                scores.append(float(line.split(":")[1].split('**')[-1].strip()))
            else: scores.append(float(line.split(":")[1].strip()))
    return scores
            
def get_role_evaluation_score_for_summary(summary,testimony_code='E1T1',attribute='Event Details'):
    denom=25 
    if attribute=='Character Description': denom=20
    police_officer_verdict=get_police_officer_verdict(summary,'',testimony_code,attribute,task='scoring')
    police_officer_score=(sum(get_scores_from_verdict(police_officer_verdict))/denom)*100
    
    inspector_verdict=get_inspector_verdict(summary,police_officer_verdict,testimony_code,attribute,task='scoring')
    inspector_score=(sum(get_scores_from_verdict(inspector_verdict))/denom)*100
    
    captain_verdict=get_captain_verdict(summary,inspector_verdict,testimony_code,attribute,task='scoring')
    captain_score=(sum(get_scores_from_verdict(captain_verdict))/denom)*100
    
    return (police_officer_score+inspector_score+captain_score)//3
    

def get_attribute_specific_string_for_improving_summary(attribute='Event Details'): 
    if attribute=='Event Details': 
        return "Ensure the summary is about the Event Details, is concise, no more than 150 words, and starts with 'According to the witness'. The definition of the Event Details is given as the chronological sequence of events that took place.\n\n"
         
    elif attribute=='Facts': 
        return "Ensure the summary is about the Facts, is concise, no more than 150 words, and starts with 'According to the witness'.The definition of the Facts is given as all the information that the witness is claiming or states.\n\n"
    
    elif attribute=='Character Description': 
        return "Ensure the summary is about the Character Description, is concise and no more than 50 words. Character Description is defined to be the names of the characters involved in the events, as well as their appearance described by the interviewee. It will also describe the relationship between the character if it is stated by the interviewee. In case you don't find such descriptions from the eyewitness, then you have to simply state the following: 'No description was provided in the witness's testimony'. \n\n"
      

def improve_summary(captain_verdict,summary,testimony_code='E1T1',attribute='Event Details'): 
    global improvement_llm
    improve_summary_string="Modify the provided summary using the evaluation from the Police Captain and the context. " # Ensure the summary is concise, no more than 150 words, and starts with 'According to the witness'.\n\n'''
    attribute_specific_string=get_attribute_specific_string_for_improving_summary(attribute)
    evaluation_criteria_string=f'Evaluation from Police Captain:\n'+captain_verdict+'\n\n'
    summary_string=f'''Original Summary:\n{summary}\n\n'''
    context_string=f'Context:\n'+get_dialogue(testimony_code)+'\n\n'
    closing_string='Strictly follow the template given below and put <eos> only at the end of the summary :\n\nRevised Summary:\n{Your answer}\n\n'''
    if improvement_llm=='Gemma 2': closing_string='Modified Summary:\n\n'''
    elif improvement_llm=='Phi 3 mini': closing_string='Strictly follow the template given below and put <eos> at the end of your modified summary :\n\nModified Summary:\nYour answer \n<eos>\n\n'''
    elif improvement_llm=='GPT-4o Mini':  closing_string='Always put an <eos> at the end of your modified summary'
    
    improve_summary_prompt=improve_summary_string+attribute_specific_string+evaluation_criteria_string+summary_string+context_string+closing_string
    if improvement_llm in ['Phi 4','Deepseek']:
        improve_summary_prompt_output = llm_generate(improve_summary_prompt,role='modifier')
        if improvement_llm!='Deepseek': improved_summary=(improve_summary_prompt_output[0]['generated_text'][2]['content'].split('Revised Summary:')[1].split('<eos>')[0]).strip('\n')
        else: improved_summary=improve_summary_prompt_output[0]['generated_text'][2]['content'].split('</think>')[1].split('<eos>')[0].strip('\n')
        
    elif improvement_llm=='GPT-4o Mini': 
        improve_summary_prompt_output=openai_generate(improve_summary_prompt,role='modifier') 
        improved_summary=improve_summary_prompt_output.split('<eos>')[0].strip('\n')
        
    elif improvement_llm=='Gemma 2': 
        improve_summary_prompt_output = pipeline2(improve_summary_prompt,max_new_tokens=800)
        improved_summary=improve_summary_prompt_output[0]['generated_text'].split('Modified Summary:')[1].strip()
        
    elif improvement_llm=='Phi 3 mini': 
        improve_summary_prompt_output = pipeline2(improve_summary_prompt,max_new_tokens=300)
        improved_summary=improve_summary_prompt_output[0]['generated_text'].split('<eos>')[2].strip()
        
    return improved_summary

def generate_summaries(events_list=[],attribute_list=['Event Details','Facts','Character Description']): 
    def run_summarization_llm(llm, attribute,testimony_code): 
        if llm not in ['Llama 3.1','GPT-4o Mini']: 
            print('Invalid LLM')
            return 
        
        dialogue=get_dialogue(testimony_code)
        context='' #context will contain the answers of the guiding prompt (empty for char description)
        if attribute!='Character Description': #guiding prompt not required for char description
            guiding_prompt=create_guiding_prompt(dialogue)
            if llm=='Llama 3.1':
    
                guiding_prompt_output = pipeline(
                    guiding_prompt,
                    max_new_tokens=700,# 400 words
                )

                #take only the output of the prompt and remove the prompt
                context=guiding_prompt_output[0]['generated_text'].split('answer:\n\n')[1].split('<end>')[0]
            
            elif llm=='GPT-4o Mini': 
                guiding_prompt_output= openai_generate(guiding_prompt,role='summarizer')
                context=guiding_prompt_output.split('<end>')[0]
            
        attribute_specific_summarization_prompt=create_attribute_specific_summarization_prompt(context,dialogue,attribute)
        attribute_specific_max_new_tokens=800
        if attribute=='Character Description':
            attribute_specific_max_new_tokens=150 #update max new tokens for char descrription to 100
            
        if llm=='Llama 3.1':
            attribute_specific_summarization_prompt_output = pipeline(
                attribute_specific_summarization_prompt,
                max_new_tokens=attribute_specific_max_new_tokens, #400 words (200 +150 +50 extra) for ED and Facts, 100 for Char Desc
            )
            if attribute!='Character Description': 
                #just extract the summary part and use eos to discard extra content
                summary=attribute_specific_summarization_prompt_output[0]['generated_text'].split('Summary')[2].split('<eos>')[0]
            else: #char description has no guiding prompt and a different attribute specific prompt structure
                summary=attribute_specific_summarization_prompt_output[0]['generated_text'].split('Character Description:')[2].split('<eos>')[0]
            
        elif llm=='GPT-4o Mini':
            attribute_specific_summarization_prompt_output= openai_generate(attribute_specific_summarization_prompt,role='summarizer')
            if attribute!='Character Description': summary=attribute_specific_summarization_prompt_output.split('Summary')[1].split('<eos>')[0].strip('\n')
            else: summary=attribute_specific_summarization_prompt_output.split('Character Description:')[1].split('<eos>')[0].strip('\n')
        return summary
    
    
    load_models(task='summarization')
    global summarization_llm
    sum_prefix=summarization_llm.replace(' ','_')
    testimony_nums=[]
    for event_num in events_list:
        for testimony_num in [1,2,3]:
            testimony_nums.append(f'E{event_num}T{testimony_num}')
        
    for testimony_code in testimony_nums:
        for attribute in attribute_list:
            try:
                summary=run_summarization_llm(summarization_llm,attribute, testimony_code)
                attribute_for_saving_file_name=attribute.replace(' ','_')
                # put the path where the 0 shot summaries will be saved
                if not os.path.exists(f'./Generated_summaries_{sum_prefix}'):
                    os.makedirs(f'./Generated_summaries_{sum_prefix}')
                new_file=open(f'./Generated_summaries_{sum_prefix}/{testimony_code}_{attribute_for_saving_file_name}.txt','w')
                new_file.write(summary.strip('\n'))
                new_file.close()
                print(f'Summary Generated for {testimony_code} and attribute {attribute}.')
            except: 
                continue
    print('0-shot Summaries saved in file '+ f'./Generated_summaries_{sum_prefix}')

#change evaluation model
def improve_summary_using_planning_engine(summary,testimony_code='E1T1',attribute='Event Details',n=1,summary_path_prefix=None,save=False): 
    global summarization_llm,improvement_llm
    sum_prefix=summarization_llm.replace(' ','_')
    improve_prefix=improvement_llm.replace(' ','_')
    
    for i in range(n): 
        if save:
            # put the path of the folder where the improved summaries will be saved
            if not os.path.exists(f'./Generated_summaries_{sum_prefix}_sum_{improve_prefix}_eval_and_imp'):
                os.makedirs(f'./Generated_summaries_{sum_prefix}_sum_{improve_prefix}_eval_and_imp')
            file=open(f'./Generated_summaries_{sum_prefix}_sum_{improve_prefix}_eval_and_imp/{summary_path_prefix}_p{i+1}.txt','w')
            
        police_officer_verdict=get_police_officer_verdict(summary,'',testimony_code,attribute)
        inspector_verdict=get_inspector_verdict(summary,police_officer_verdict,testimony_code,attribute)
        captain_verdict=get_captain_verdict(summary,inspector_verdict,testimony_code,attribute)

        summary=improve_summary(captain_verdict,summary,testimony_code,attribute)
        
        if save:
            # put the path of the folder where the improved summaries will be saved
            file=open(f'./Generated_summaries_{sum_prefix}_sum_{improve_prefix}_eval_and_imp/{summary_path_prefix}_p{i+1}.txt','w')
            file.write(summary)
            file.close()
            print(f'Summary improved for {testimony_code} and attribute {attribute} and n={i+1}.')
    print('Improved Summaries saved in file '+ f'./Generated_summaries_{sum_prefix}_sum_{improve_prefix}_eval_and_imp')
        
def improve_summaries(testimony_code,attribute_list,max_n): 
    global summarization_llm
    load_models(task='improvement')
    sum_prefix=summarization_llm.replace(' ','_')
    for attribute in attribute_list: 
        try:
            path_prefix=f"{testimony_code}_{attribute.replace(' ','_')}"
            # put the path where 0-shot summaries are saved
            summary_file=open(f'./Generated_summaries_{sum_prefix}/{path_prefix}.txt','r')
            summary=summary_file.read()
            summary_file.close()
            improve_summary_using_planning_engine(summary,testimony_code,attribute,max_n,path_prefix,save=True)
        except:
            continue
       
def evaluate_summaries(events_list,attribute_list,max_n):
    #  len(attribute_list) lists of size n=1 each will be returned . Calculate avg across all 45 testimonies (num of events *3)
    global summarization_llm, improvement_llm
    load_models(task='scoring')
    final_scores={}
    for a in attribute_list: 
        final_scores[a]=[]
    
    sum_prefix=summarization_llm.replace(' ','_')
    improve_prefix=improvement_llm.replace(' ','_')
    # put the path of the log file 
    log_file=open(f'./evaluation_log_{sum_prefix}_sum_{improve_prefix}_eval_and_imp.txt','w')
        
    for n in range(0,max_n+1): 
        for attribute in attribute_list: 
            attribute_score=[]
            for event_num in events_list: 
                for testimony_num in [1,2,3]: 
                    testimony_code=f'E{event_num}T{testimony_num}'
                    path_prefix=f"{testimony_code}_{attribute.replace(' ','_')}"
                    n_value_suffix=f'_p{n}'
                    if n==0: n_value_suffix=''
                    path_prefix+=n_value_suffix
                    try:
                        # put the path of the folder where the improved summaries are saved
                        if n!=0: summary_file=open(f'./Generated_summaries_{sum_prefix}_sum_{improve_prefix}_eval_and_imp/{path_prefix}.txt','r')
                        else: summary_file=open(f'./Generated_summaries_{sum_prefix}/{path_prefix}.txt','r')
                        summary=summary_file.read()
                        summary_file.close()
                        score=get_role_evaluation_score_for_summary(summary,testimony_code,attribute)
                        attribute_score.append(score)
                    except: 
                        score=-1
                    if score!=-1: log_file.write(f'Score for testimony {testimony_code}, attribute {attribute} and n={n} is: {score}\n')
                    else: log_file.write(f'Error in evaluating the testimony testimony {testimony_code}, attribute {attribute} and n={n}\n')
                    log_file.flush()
                    print(f'Score for testimony {testimony_code}, attribute {attribute} and n={n} is:',score)
            final_scores[attribute].append(sum(attribute_score)/len(attribute_score))
            log_file.write(f'**\nAvg Score for attribute {attribute} and n={n} is: {final_scores[attribute][n-1]}\n**\n')
            log_file.flush()
            print('**')
            print(f'Avg Score for attribute {attribute} and n={n} is:',final_scores[attribute][n])
            print('**')
    print()
    print('FINAL SCORES:')
    log_file.write('FINAL SCORES:')
    log_file.flush()
    for attribute in final_scores.keys(): 
        print(f'Final Scores for attribute {attribute} and n=0-3 are:',final_scores[attribute])
        log_file.write(f'Final Scores for attribute {attribute} and n=0-3 are: {final_scores[attribute]}\n')
        log_file.flush()
    log_file.close()
    print('Evaluation scores saved in '+ f'./evaluation_{sum_prefix}_sum_{improve_prefix}_eval_and_imp.txt')
    return final_scores
                
def main(): 
    global cuda_id, remove_labels, summarization_llm, improvement_llm
    parser = argparse.ArgumentParser(description="A script that takes command-line arguments")
    parser.add_argument("--testimony_code", type=str, default="E1T1", help="The particular testimony to be processed")
    parser.add_argument("--events_list", type=str, default="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]", help="List of all events to be processed")
    parser.add_argument("--attribute_list", type=str, default="['Event Details','Facts','Character Description']", help="List of all attributes to be processed")
    parser.add_argument("--max_n", type=int, default=1, help="No. of planning engine iterations with a default value 1")
    parser.add_argument("--cuda_id", type=int, default=1, help="id (0 ,1 ..etc) for GPU number")
    parser.add_argument("--mode", type=str, default="improvement", help="evaluation mode or improvement mode")
    parser.add_argument("--remove_labels", type=bool, default=False, help="remove labels from dialogues or not")
    parser.add_argument("--summarization_model", type=str, default="Llama 3.1", help="Summarization model to be used")
    parser.add_argument("--improvement_model", type=str, default="Phi 4", help="Improvement model to be used")
    args = parser.parse_args() 
    mode=args.mode
    cuda_id=args.cuda_id #set the gpu no. which has become empty for execution
    remove_labels=args.remove_labels
    summarization_llm=args.summarization_model
    improvement_llm=args.improvement_model
    
    args.attribute_list = ast.literal_eval(args.attribute_list)
    args.events_list = ast.literal_eval(args.events_list)
    events_list=args.events_list
    
    
    if mode=='summarization': 
        generate_summaries(events_list,['Event Details', 'Facts', 'Character Description'])
    elif mode=='improvement': 
        improve_summaries(testimony_code=args.testimony_code,attribute_list=args.attribute_list,max_n=args.max_n)
    elif mode=='evaluation': 
        evaluate_summaries(events_list,['Event Details','Facts','Character Description'],args.max_n) 

if __name__=='__main__':  
    main()