#################################################################################
# Chat2VIS supporting functions
# https://chat2vis.streamlit.app/
# Paula Maddigan
#################################################################################

import openai
from langchain import HuggingFaceHub, LLMChain,PromptTemplate
from transformers import pipeline
import requests

def run_request(question_to_ask, model_type, api_keys):
    if model_type == "gpt-4" or model_type == "gpt-3.5-turbo" :
        # Run OpenAI ChatCompletion API
        task = "Generate Python Code Script."
        if model_type == "gpt-4":
            # Ensure GPT-4 does not include additional comments
            task = task + " The script should only include code, no comments."
        openai.api_key = api_keys.get('openai_key')
        response = openai.ChatCompletion.create(model=model_type,
            messages=[{"role":"system","content":task},{"role":"user","content":question_to_ask}])
        llm_response = response["choices"][0]["message"]["content"]
    elif model_type == "text-davinci-003" or model_type == "gpt-3.5-turbo-instruct":
        # Run OpenAI Completion API
        openai.api_key = api_keys.get('openai_key')
        response = openai.Completion.create(engine=model_type,prompt=question_to_ask,temperature=0,max_tokens=500,
                    top_p=1.0,frequency_penalty=0.0,presence_penalty=0.0,stop=["plt.show()"])
        llm_response = response["choices"][0]["text"]
    elif model_type == "gemini":
        # Google Gemini model
        gemini_key = api_keys.get('gemini_key')
        headers = {"Authorization": f"Bearer {gemini_key}"}
        payload = {"prompt": question_to_ask, "max_tokens": 500}
        response = requests.post("https://api.google.com/gemini/generate", headers=headers, json=payload)
        llm_response = response.json()['choices'][0]['text'].strip()
    else:
        # Hugging Face model
        llm = HuggingFaceHub(huggingfacehub_api_token = api_keys.get('hf_key'), repo_id="codellama/" + model_type, model_kwargs={"temperature":0.1, "max_new_tokens":500})
        llm_prompt = PromptTemplate.from_template(question_to_ask)
        llm_chain = LLMChain(llm=llm,prompt=llm_prompt)
        llm_response = llm_chain.predict()
    # rejig the response
    llm_response = format_response(llm_response)
    return llm_response

def format_response(res):
    # Remove the load_csv from the answer if it exists
    csv_line = res.find("read_csv")
    if csv_line > 0:
        return_before_csv_line = res[0:csv_line].rfind("\n")
        if return_before_csv_line == -1:
            # The read_csv line is the first line so there is nothing to need before it
            res_before = ""
        else:
            res_before = res[0:return_before_csv_line]
        res_after = res[csv_line:]
        return_after_csv_line = res_after.find("\n")
        if return_after_csv_line == -1:
            # The read_csv is the last line
            res_after = ""
        else:
            res_after = res_after[return_after_csv_line:]
        res = res_before + res_after
    return res

def format_question(primer_desc,primer_code , question, model_type):
    # Fill in the model_specific_instructions variable
    instructions = ""
    if model_type == "Code Llama":
        # Code llama tends to misuse the "c" argument when creating scatter plots
        instructions = "\nDo not use the 'c' argument in the plot function, use 'color' instead and only pass color names like 'green', 'red', 'blue'."
    primer_desc = primer_desc.format(instructions)  
    # Put the question at the end of the description primer within quotes, then add on the code primer.
    return  '"""\n' + primer_desc + question + '\n"""\n' + primer_code

def get_primer(df_dataset,df_name):
    # Primer function to take a dataframe and its name
    # and the name of the columns
    # and any columns with less than 20 unique values it adds the values to the primer
    # and horizontal grid lines and labeling
    primer_desc = "Use a dataframe called df from data_file.csv with columns '" \
        + "','".join(str(x) for x in df_dataset.columns) + "'. "
    for i in df_dataset.columns:
        if len(df_dataset[i].drop_duplicates()) < 20 and df_dataset.dtypes[i]=="O":
            primer_desc = primer_desc + "\nThe column '" + i + "' has categorical values '" + \
                "','".join(str(x) for x in df_dataset[i].drop_duplicates()) + "'. "
        elif df_dataset.dtypes[i]=="int64" or df_dataset.dtypes[i]=="float64":
            primer_desc = primer_desc + "\nThe column '" + i + "' is type " + str(df_dataset.dtypes[i]) + " and contains numeric values. "   
    primer_desc = primer_desc + "\nLabel the x and y axes appropriately."
    primer_desc = primer_desc + "\nAdd a title. Set the fig suptitle as empty."
    primer_desc = primer_desc + "{}" # Space for additional instructions if needed
    primer_desc = primer_desc + "\nUsing Python version 3.9.12, create a script using the dataframe df to graph the following: "
    pimer_code = "import pandas as pd\nimport matplotlib.pyplot as plt\n"
    pimer_code = pimer_code + "fig,ax = plt.subplots(1,1,figsize=(10,4))\n"
    pimer_code = pimer_code + "ax.spines['top'].set_visible(False)\nax.spines['right'].set_visible(False) \n"
    pimer_code = pimer_code + "df=" + df_name + ".copy()\n"
    return primer_desc,pimer_code

def summarize_graph(graph_code, model, api_keys):
    summary_prompt = f"Analyze the dataset visualized by the following Python code. Provide a summary of the data, including key insights and significant patterns. Do not describe the code:\n\n{graph_code}\n\nSummary:"
    try:

        if model in ["gpt-4", "gpt-3.5-turbo"]:
            openai.api_key = api_keys.get('openai_key')
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": "You are a professional data analyst."}, {"role": "user", "content": summary_prompt}],
                max_tokens=500,
                n=1,
                stop=None,
                temperature=0.5
            )
            summary = response['choices'][0]['message']['content'].strip()
        elif model in ["text-davinci-003", "gpt-3.5-turbo-instruct"]:
            openai.api_key = api_keys.get('openai_key')
            response = openai.Completion.create(
                engine=model,
                prompt=summary_prompt,
                max_tokens=500,
                n=1,
                stop=None,
                temperature=0.5
            )
            summary = response.choices[0].text.strip()
        elif model == "CodeLlama-34b-Instruct-hf":
            summarizer = pipeline("text-generation", model=model, api_key=api_keys.get('hf_key'))
            response = summarizer(summary_prompt, max_length=150, num_return_sequences=1)
            summary = response[0]['generated_text']
        elif model == "gemini":
            # Google Gemini model
            headers = {"Authorization": f"Bearer {api_keys.get('gemini_key')}"}
            payload = {"prompt": summary_prompt, "max_tokens": 150}
            response = requests.post("https://api.google.com/gemini/generate", headers=headers, json=payload)
            if response.status_code == 200:
                summary = response.json()['choices'][0]['text'].strip()
            else:
                raise Exception(f"Google Gemini API error: {response.status_code}, {response.text}")
        else:
            summary = "Summarizer model not supported."
        return summary
    except Exception as e:
        error_message = f"Error in summarize_graph for model {model}: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        raise