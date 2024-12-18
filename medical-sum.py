import boto3
from botocore.config import Config
import os
import fitz
from textractor import Textractor
from textractor.data.constants import TextractFeatures
from PIL import Image
import io
import pandas as pd
from botocore.exceptions import ClientError
import time
import json
import streamlit as st
from boto3.dynamodb.conditions import Key 
config = Config(
    read_timeout=600,
    retries = dict(
        max_attempts = 5 ## Handle retries
    )
)
import re
import base64
from streamlit.runtime.uploaded_file_manager import UploadedFile
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import concurrent.futures
from functools import partial
import random
from urllib3.exceptions import IncompleteRead



# Read credentials
with open('config.json') as f:
    config_file = json.load(f)
# pricing info
with open('pricing.json') as f:
    pricing_file = json.load(f)
    
DYNAMODB_TABLE=config_file["DynamodbTable"]
BUCKET=config_file["Bucket_Name"]
OUTPUT_TOKEN=config_file["max-output-token"]
S3 = boto3.client('s3')
DYNAMODB  = boto3.resource('dynamodb')
TEXTRACT_RESULT_CACHE_PATH=config_file["textract_result_path"]
CHAT_HISTORY_LENGTH=config_file["chat-history-loaded-length"]
DYNAMODB_USER=config_file["UserId"]
REGION=config_file["bedrock-region"]
bedrock_runtime = boto3.client(service_name='bedrock-runtime',region_name=REGION,config=config)
PREFIX=config_file["s3_path_prefix"]
INPUT_EXT=tuple(f".{x}" for x in config_file["input_file_ext"].split(','))
INPUT_BUCKET=config_file["Bucket_Name"]
INPUT_S3_PATH=config_file["input_s3_path"]

st.set_page_config(layout="wide")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'token' not in st.session_state:
    st.session_state['token'] = 0
if 'chat_hist' not in st.session_state:
    st.session_state['chat_hist'] = []
if 'page_summ' not in st.session_state:
    st.session_state['page_summ'] = ""
if 'extraction' not in st.session_state:
    st.session_state['extraction'] = ""
if 'user_sess' not in st.session_state:
    st.session_state['user_sess'] = f"{str(time.time()).split('.')[0]}"
if 'userid' not in st.session_state:
    st.session_state['userid']= config_file["UserId"]
if 'cost' not in st.session_state:
    st.session_state['cost'] = 0
if 'textract_cost' not in st.session_state:
    st.session_state['textract_cost'] = 0
if 'mode' not in st.session_state:
    st.session_state['mode'] = ""
if 'conf' not in st.session_state:
    st.session_state['conf'] = ""
if 'data_pages' not in st.session_state:
    st.session_state['data_pages'] = 0
if 'gt' not in st.session_state:
    st.session_state['gt'] = None
if 'editor_box' not in st.session_state:
    st.session_state['editor_box'] = None

SESSIONID = st.session_state['user_sess']


def get_object_with_retry(bucket, key):
    max_retries = 5
    retries = 0   
    backoff_base = 2
    max_backoff = 3  # Maximum backoff time in seconds
    s3 = boto3.client('s3')

    while retries < max_retries:
        try:
            response = s3.get_object(Bucket=bucket, Key=key)
            # Attempt to read the entire content to catch potential IncompleteRead errors
            # response['Body'].read()
            # response['Body'].seek(0)  # Reset the stream position
            return response
        except ClientError as e:
            error_code = e.response['Error']['Code']            
            if error_code == 'DecryptionFailureException':
                sleep_time = min(max_backoff, backoff_base ** retries + random.uniform(0, 1))
                print(f"Decryption failed, retrying in {sleep_time} seconds...")                
                time.sleep(sleep_time)               
                retries += 1
            elif error_code == 'ModelStreamErrorException':
                if retries < max_retries:
                    sleep_time = min(max_backoff, backoff_base ** retries + random.uniform(0, 1))
                    time.sleep(sleep_time)
                    retries += 1
            else:
                raise e
        except IncompleteRead as e:
            if retries < max_retries:
                sleep_time = min(max_backoff, backoff_base ** retries + random.uniform(0, 1))
                print(f"IncompleteRead error, retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                retries += 1
            else:
                raise Exception(f"Failed to read complete response after {max_retries} attempts: {str(e)}")

    # If we've exhausted all retries
    raise Exception(f"Failed to retrieve object after {max_retries} attempts")
    
def put_db(params,messages):
    """Store long term chat history in DynamoDB"""    
    chat_item = {
        "UserId": st.session_state['userid'], # user id
        "SessionId": params["session_id"], # User session id
        "messages": [messages],  # 'messages' is a list of dictionaries
        "time":messages['time']
    }

    existing_item = DYNAMODB.Table(DYNAMODB_TABLE).get_item(Key={"UserId": st.session_state['userid'], "SessionId":params["session_id"]})
    if "Item" in existing_item:
        existing_messages = existing_item["Item"]["messages"]
        chat_item["messages"] = existing_messages + [messages]
    response = DYNAMODB.Table(DYNAMODB_TABLE).put_item(
        Item=chat_item
    )


def get_chat_history_db(params,cutoff):
    current_chat, chat_hist=[],[]
    if DYNAMODB_TABLE:
        if "Item" in params['chat_histories']:  
            chat_hist=params['chat_histories']['Item']['messages'][-cutoff:]            
            for d in chat_hist:            
                current_chat.append({'role': 'user', 'content': [{"text":d['user']}]})
                current_chat.append({'role': 'assistant', 'content':[{"text":d['assistant']}]})  
        else:
            chat_hist=[]
    else:
        if params['chat_histories']:
            chat_hist = params['chat_histories'][-cutoff:]    
            for d in chat_hist:            
                    current_chat.append({'role': 'user', 'content': [{"text":d['user']}]})
                    current_chat.append({'role': 'assistant', 'content':[{"text":d['assistant']}]})     
        else:
            chat_hist = []
    return current_chat, chat_hist
    
def get_session_ids_by_user(table_name, user_id):
    """
    Get Session Ids and corresponding top message for a user to populate the chat history drop down on the front end
    """
    message_list={}
    session_ids = []
    if DYNAMODB_TABLE:
        table = DYNAMODB.Table(table_name)
        
        args = {
            'KeyConditionExpression': Key('UserId').eq(user_id)
        }
        while True:
            response = table.query(**args)
            session_ids.extend([item['SessionId'] for item in response['Items']])
            if 'LastEvaluatedKey' not in response:
                break
            args['ExclusiveStartKey'] = response['LastEvaluatedKey']
            
        for session_id in session_ids:
            try:
                message_list[session_id]=DYNAMODB.Table(table_name).get_item(Key={"UserId": user_id, "SessionId":session_id})['Item']['messages'][0]['user']
            except Exception as e:
                print(e)
                pass

    else:
        try:
            message_list={}
            # Read the existing JSON data from the file
            with open("chat_history.json", "r", encoding='utf-8') as file:
                existing_data = json.load(file)
            for session_id in existing_data:
                message_list[session_id]=existing_data[session_id][0]['user']
            
        except FileNotFoundError:
            # If the file doesn't exist, initialize an empty list
            message_list = {}
    return message_list


def bedrock_claude_(params,chat_history,system_message, prompt,model_id,image_path=None, handler=None):
    chat_history_copy = chat_history[:]
    content=[]
    if image_path:       
        if not isinstance(image_path, list):
            image_path=[image_path]      
        for img in image_path:
            s3 = boto3.client('s3',region_name="us-east-1")
            match = re.match("s3://(.+?)/(.+)", img)            
            image_name=os.path.basename(img)
            _,ext=os.path.splitext(image_name)
            if "jpg" in ext: ext=".jpeg"           
            bucket_name = match.group(1)
            key = match.group(2)    
            obj = s3.get_object(Bucket=bucket_name, Key=key)
            bytes_image=obj['Body'].read()            
            content.extend([{"text":image_name},{
              "image": {
                "format": f"{ext.lower().replace('.','')}",
                "source": {"bytes":bytes_image}
              }
            }])

    content.append({       
        "text": prompt
            })
    chat_history_copy.append({"role": "user",
            "content": content})
    system_message=[{"text":system_message}]
    response = bedrock_runtime.converse_stream(messages=chat_history_copy, modelId=model_id,inferenceConfig={"maxTokens": 2000, "temperature": 0.5,},system=system_message)
    answer=bedrock_streemer(params,response, handler) 
    return answer

def textract_pricing(pages):
    detect_text_pp_page = 1.5/1000
    tables_pp_page = 15/1000
    forms_pp_page = 50/1000
    cost= (detect_text_pp_page + tables_pp_page + forms_pp_page) * pages
    return cost
    

def bedrock_streemer(params,response, handler):
    text=''
    for chunk in response['stream']:       

        if 'contentBlockDelta' in chunk:
            delta = chunk['contentBlockDelta']['delta']       
            if 'text' in delta:
                text += delta['text']               
                handler.markdown(text.replace("$","USD ").replace("%", " percent"))

        elif "metadata" in chunk:
            st.session_state['input_token']=chunk['metadata']['usage']["inputTokens"]
            st.session_state['output_token']=chunk['metadata']['usage']["outputTokens"]
            latency=chunk['metadata']['metrics']["latencyMs"]
            pricing=st.session_state['input_token']*pricing_file[f"us.meta.{params['model']}-instruct-v1:0"]["input"]+st.session_state['output_token'] *pricing_file[f"us.meta.{params['model']}-instruct-v1:0"]["output"]
            st.session_state['cost']+=pricing             
    return text

def _invoke_bedrock_with_retries(params,current_chat, chat_template, question, model_id, image_path, handler):
    max_retries = 5
    backoff_base = 2
    max_backoff = 3  # Maximum backoff time in seconds
    retries = 0
    while True:
        try:
            response = bedrock_claude_(params,current_chat, chat_template, question, model_id, image_path, handler)
            return response
        except ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException':
                if retries < max_retries:
                    # Throttling, exponential backoff
                    sleep_time = min(max_backoff, backoff_base ** retries + random.uniform(0, 1))
                    time.sleep(sleep_time)
                    retries += 1
                else:
                    raise e
            else:
                # Some other API error, rethrow
                raise

def query_llm(params, handler):
    image_path=[]
    claude3=False
    model="us.meta."+params['model']+"-instruct-v1:0"

    current_chat,chat_hist=get_chat_history_db(params, CHAT_HISTORY_LENGTH)
                
    with open("prompt/chat.txt","r") as f:
        system_template=f.read()
    if "doc" in params:
        prompt=f'Here are the document:\n<document>\n{params["doc"]}\n</document>\n{params["prompt"]}'
    elif "image_path" in params:
        prompt=params["prompt"]
        image_path=params["image_path"]
    response=_invoke_bedrock_with_retries(params,current_chat, system_template, prompt, model, image_path, handler)
       
    
    chat_history={"user":params['prompt'],
    "assistant":response,
    # "document":params['file_item'],
    "modelID":model,
    "time":str(time.time()),
    "input_token":round(st.session_state['input_token']) ,
    "output_token":round(st.session_state['output_token'])} 

    #store convsation memory in DynamoDB table
    if DYNAMODB_TABLE:
        put_db(params,chat_history)
    else: 
        save_chat_local("chat_history.json",[chat_history])
    return response
    

# @st.cache_data
def process_and_upload_files_to_s3(file, s3_bucket, s3_prefix):
    """
    Uploads PDF and image files to an Amazon S3 bucket.
    Args:
        file (list): A list of file-like objects representing the uploaded files.
        s3_bucket (str): The name of the Amazon S3 bucket to upload the files to.
        s3_prefix (str): The prefix to use for the uploaded file names in the S3 bucket.

    Returns:
        list: A list of dictionaries, where each dictionary contains information about the uploaded files.
              The dictionary has the following keys:
                - 'file_name': The name of the uploaded file.
                - 'file_paths': A list of S3 paths for the uploaded files.
    """
    s3_client = boto3.client('s3')
    upload_info = []
    image_paths = []
    # for file in uploaded_files:
    file_bytes=s3_client.get_object(Bucket=BUCKET, Key=f"{PREFIX}/{file}")['Body'].read()
    file_name=file
    if file_name.lower().endswith('.pdf'):            
        pdf_file = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")            
        for page_index in range(len(pdf_file)):
            # Select the page
            page = pdf_file[page_index]                
            # Get the page dimensions in PDF units (1/72 inch)
            page_rect = page.rect
            # Convert the dimensions from PDF units to inches
            page_width_inches = page_rect.width / 72
            page_height_inches = page_rect.height / 72
            # Calculate the maximum DPI to keep pixel dimensions below 1500
            max_dpi = min(1500 / page_width_inches, 1500 / page_height_inches)
            # Render the page as a PyMuPDF Image object
            pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), dpi=round(max_dpi))
            # Convert the PyMuPDF Image object to bytes
            image_bytes = pix.tobytes()
            # Construct the image file name
            image_filename = f"{s3_prefix}/{file_name.replace('.pdf', '')}-page-{page_index+1}.jpeg"
            # Upload the image to S3
            s3_client.put_object(Bucket=s3_bucket, Key=image_filename, Body=image_bytes)
            image_paths.append(f"s3://{s3_bucket}/{image_filename}")           
    else:
        # Assume it's an image file
        image_filename = f"{s3_prefix}/{file_name}"
        # Upload the image to S3
        s3_client.upload_fileobj(io.BytesIO(file_bytes), s3_bucket, image_filename)
        image_path = f"s3://{s3_bucket}/{image_filename}"
        image_paths.append(               
            image_path
           )

    
        
    return image_paths


def list_s3_filess(bucket, prefix):
    keys=[]
    s3 = boto3.client('s3')    
    # List files in folder
    result = s3.list_objects(Bucket=bucket, Prefix=prefix)

    for object in result['Contents']:
        key=object['Key']
        keys.append(os.path.relpath(key, prefix))        
    return keys


def get_chat_historie_for_streamlit(params):
    """
    This function retrieves chat history stored in a dynamoDB table partitioned by a userID and sorted by a SessionID
    """
    if DYNAMODB_TABLE:
        chat_histories = DYNAMODB.Table(DYNAMODB_TABLE).get_item(Key={"UserId": st.session_state['userid'], "SessionId":params["session_id"]})
    # Constructing the desired list of dictionaries
        formatted_data = []
        if 'Item' in chat_histories:
            for entry in chat_histories['Item']['messages']:
                formatted_data.append({
                    "role": "user",
                    "content": entry["user"],
                })
                formatted_data.append({
                    "role": "assistant",
                    "content": entry["assistant"], 
                })
        else:
            chat_histories=[]           
    else:
        chat_histories=load_chat_local("chat_history.json")   
        # Constructing the desired list of dictionaries
        formatted_data = []
        if chat_histories:
            for entry in chat_histories:
                formatted_data.append({
                    "role": "user",
                    "content": entry["user"],
                })
                formatted_data.append({
                    "role": "assistant",
                    "content": entry["assistant"], 
                })
        else:
            chat_histories=[]       
    return formatted_data,chat_histories

def get_key_from_value(dictionary, value):
    return next((key for key, val in dictionary.items() if val == value), None)

def get_s3_keys(prefix):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    keys=""
    if "Contents" in response:
        keys = []
        for obj in response['Contents']:
            key = obj['Key']
            name = key[len(prefix):]
            keys.append(name)
    return keys


def handle_doc_upload_or_s3(file):
    if "claude" == st.session_state['mode']:
        result = process_and_upload_files_to_s3(file, BUCKET, PREFIX)
    elif "textract" == st.session_state['mode']:
        result = get_text_ocr_(file)
    return result


def process_files(files):
    results = []
    result_string={}
    table_string = {}
    errors = []
    future_proxy_mapping = {} 
    futures = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Partial function to pass the handle_doc_upload_or_s3 function
        func = partial(handle_doc_upload_or_s3)   
        for file in files:
            future = executor.submit(func, file)
            future_proxy_mapping[future] = file
            futures.append(future)

        # Collect the results and handle exceptions
        for future in concurrent.futures.as_completed(futures):        
            file_url= future_proxy_mapping[future]
            try:
                result = future.result()
                results.append(result)
                doc_name=os.path.basename(file_url)
                
                # result_string+=f"<{doc_name}>\n{result}\n</{doc_name}>\n"
                result_string[doc_name] = result[0]
                table_string[doc_name] = result[1]
                st.session_state['data_pages']+=result[2]

            except Exception as e:
                # Get the original function arguments from the Future object
                error = {'file': file_url, 'error': str(e)}
                errors.append(error)
                

    return table_string, errors, result_string

def read_json_from_s3(bucket, key):
    import io
    s3_client = boto3.client('s3')
    obj = get_object_with_retry(bucket,key)
    return pd.read_json(io.BytesIO(obj['Body'].read()))

def check_file_exists_in_s3(file, s3_path):
    file_base_name = os.path.basename(file)
    txt_file_name = file_base_name + ".txt"
    json_file_name = file_base_name + ".json"

    s3_keys = get_s3_keys(s3_path)

    txt_exists = any(txt_file_name == x for x in s3_keys)
    json_exists = any(json_file_name == x for x in s3_keys)

    return txt_exists and json_exists


# @st.cache_data
def get_text_ocr_(file):    
    """
    Extract text from an image file (JPG, PNG, JPEG) or PDF file using Amazon Textract.
    Args:
        file (str): The path or key of the file to be processed.
    Returns:
        str: The extracted text from the file.
    This function first checks if the text extraction result for the given file already exists in an S3 bucket at the specified `TEXTRACT_RESULT_CACHE_PATH`. If the result exists, it retrieves the text from the S3 object and returns it, skipping the text extraction process.
    If the text extraction result doesn't exist, the function proceeds to extract the text from the file using Amazon Textract. It determines the file type (image or PDF) based on the file extension and calls the appropriate Textract method (`analyze_document` for images, `start_document_analysis` for PDFs)."""
    file_base_name=os.path.basename(file)+".txt"
    if check_file_exists_in_s3(file, f"{TEXTRACT_RESULT_CACHE_PATH}/"):   
        response = get_object_with_retry(BUCKET, f"{TEXTRACT_RESULT_CACHE_PATH}/{file_base_name}")
        text = response['Body'].read().decode()
        # response = S3.get_object(Bucket=BUCKET, Key=f"{TEXTRACT_RESULT_CACHE_PATH}/{os.path.basename(file)}.json")
        word_dict=read_json_from_s3(BUCKET, f"{TEXTRACT_RESULT_CACHE_PATH}/{os.path.basename(file)}.json")      
        return text, word_dict, 0
    else:
        _, file_ext = os.path.splitext(file)    
        doc_id=os.path.basename(file)
        extractor = Textractor(region_name="us-east-1")
        image_extensions = ('.jpg', '.png', '.jpeg','.tif')
        if file_ext.lower().endswith(image_extensions):
            document = extractor.analyze_document(
                file_source=file,
                features=[TextractFeatures.LAYOUT,TextractFeatures.TABLES,TextractFeatures.FORMS],
                save_image=False,
            )
            
        elif file_ext.lower().endswith('.pdf'):            
            document = extractor.start_document_analysis(
                file_source=file,
                features=[TextractFeatures.LAYOUT,TextractFeatures.TABLES,TextractFeatures.FORMS],
                save_image=False,
            )
        from textractor.data.text_linearization_config import TextLinearizationConfig
        configs = TextLinearizationConfig(
            hide_figure_layout=False,
            hide_header_layout=False,
            # table_prefix="<table>",
            # table_suffix="</table>",
            hide_footer_layout=False,
            hide_page_num_layout=False,    
            table_linearization_format='markdown'
        )
        word_dict={}
        word_dict["word"]=[]
        word_dict["confidence_score"]=[]
        for word in document.words:
            word_dict["word"].append(str(word))
            word_dict["confidence_score"].append(round(word.confidence*100))     
            word_dic_save=json.dumps(word_dict)
            
        S3.put_object(Body=word_dic_save, Bucket=BUCKET, Key=f"{TEXTRACT_RESULT_CACHE_PATH}/{doc_id}.json") 
        time.sleep(1)
        S3.put_object(Body=document.get_text(config=configs), Bucket=BUCKET, Key=f"{TEXTRACT_RESULT_CACHE_PATH}/{doc_id}.txt")   
        # st.session_state['data_pages']+=len(document.pages)
        return document.get_text(config=configs), word_dict, len(document.pages)


def save_chat_local(file_path, new_data):
    """Store long term chat history Local Disk"""   
    try:
        # Read the existing JSON data from the file
        with open(file_path, "r",encoding='utf-8') as file:
            existing_data = json.load(file)
        if SESSIONID not in existing_data:
            existing_data[SESSIONID]=[]
    except FileNotFoundError:
        # If the file doesn't exist, initialize an empty list
        existing_data = {SESSIONID:[]}
    # Append the new data to the existing list
    from decimal import Decimal
    data = [{k: float(v) if isinstance(v, Decimal) else v for k, v in item.items()} for item in new_data]
    existing_data[SESSIONID].extend(data)
    # Write the updated list back to the JSON file
    with open(file_path, "w") as file:
        json.dump(existing_data, file)
        
def load_chat_local(file_path):
    """Load long term chat history from Local"""   
    try:
        # Read the existing JSON data from the file
        with open(file_path, "r",encoding='utf-8') as file:
            existing_data = json.load(file)
            if SESSIONID in existing_data:
                existing_data=existing_data[SESSIONID]
            else:
                existing_data=[]
    except FileNotFoundError:
        # If the file doesn't exist, initialize an empty list
        existing_data = []
    return existing_data


def is_pdf(file_bytes):
    """
    Checks if the given bytes represent a PDF file.
    Args:
        file_bytes (bytes): The bytes to check.
    Returns:
        bool: True if the bytes represent a PDF file, False otherwise.
    """
    # PDF file signature is "%PDF"
    pdf_signature = b"%PDF"
    return file_bytes.startswith(pdf_signature)

def is_image(file_bytes):
    """
    Checks if the given bytes represent an image file.
    Args:
        file_bytes (bytes): The bytes to check.
    Returns:
        bool: True if the bytes represent an image file, False otherwise.
    """
    # Common image file signatures
    image_signatures = {
        b"\xff\xd8": "jpeg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38\x37\x61": "gif",
        b"\x42\x4d": "bmp",
        b"\x49\x49\x2A\x00": "tiff",  # Little-endian TIFF
        b"\x4D\x4D\x00\x2A": "tiff",  # Big-endian TIFF
    }

    for signature, _ in image_signatures.items():
        if file_bytes.startswith(signature):
            return True

    return False

def save_string_as_pdf(text, file_path):
    # Create a PDF document
    doc = SimpleDocTemplate(file_path, pagesize=letter)

    # Get the available styles
    styles = getSampleStyleSheet()

    # Split the text into lines
    lines = text.splitlines()

    # Create a list of Paragraph objects
    elements = []
    for line in lines:
        paragraph = Paragraph(line, styles["BodyText"])
        elements.append(paragraph)

    # Build the PDF document
    doc.build(elements)
    st.write(f"PDF file saved")


def page_summary(params):
    """
    This action takes the entire rendered document page as context for the following LLM actions below.
    """
    import time
    pdf_file=""
    item_cache={}
    json_cache={}
    s3 = boto3.client('s3')
    if params['file_item'] and params["processor"] == "Textract":
        if isinstance(params['file_item'], list):
            st.session_state['mode']='textract'
            doc_list=[]
            json_list=[]
            if all(isinstance(file, UploadedFile) for file in params['file_item']):
                for file in params['file_item']:
                    if not (file.name.endswith('.json') or file.name.endswith('.txt')):
                        pdf_name=file.name
                        pdf_bytes=file.read()
                        item_cache[pdf_name]=pdf_bytes
                        s3.put_object(Bucket=BUCKET, Key=f"{PREFIX}/{pdf_name}", Body=pdf_bytes)
                        doc_list.append(pdf_name)
                    else:
                        json_name=file.name
                        json_bytes=file.read().decode()
                 
                        json_cache[json_name]=json_bytes                   
                        json_list.append(json_name)
                if doc_list:
                    doc_file_names=[f"s3://{BUCKET}/{PREFIX}/{x}" for x in doc_list]           
                    params['file_processed'] = doc_file_names
                if json_list:                    
                    st.session_state['gt']= json_cache

    if params['s3_objects'] and params["processor"] == "Textract":
        if isinstance(params['s3_objects'], list):
            st.session_state['mode']='textract'
            doc_list=[]
            json_list=[]
            if all(isinstance(file, str) for file in params['s3_objects']):
                for file in params['s3_objects']:
                    if not (file.endswith('.json') or file.endswith('.txt')):
                        pdf_name=file
                        # st.write(f"{INPUT_S3_PATH}/{pdf_name}")
                        pdf_bytes=get_object_with_retry(BUCKET, f"{INPUT_S3_PATH}/{pdf_name}")["Body"].read()
                        item_cache[pdf_name]=pdf_bytes                   
                        doc_list.append(pdf_name)
                    else:
                        json_name=file
                        json_bytes=get_object_with_retry(BUCKET, f"{INPUT_S3_PATH}/{json_name}")["Body"].read().decode()
                        json_cache[json_name]=json_bytes                   
                        json_list.append(json_name)
                if doc_list:
                    doc_file_names=[f"s3://{BUCKET}/{INPUT_S3_PATH}/{x}" for x in doc_list]   
                    if 'file_processed' in params and params['file_processed']:
                        params['file_processed']= params['file_processed']+doc_file_names
                    else:
                        params['file_processed']=doc_file_names
                if json_list:
                    if st.session_state['gt']:
                        st.session_state['gt'].update(json_cache)
                    else:
                        st.session_state['gt']= json_cache
                        

    if  item_cache:
        colm1,colm2=st.columns([1,1])
        page_count=len(item_cache.keys())
        with colm1:
            if page_count==1:
                filee=item_cache[list(item_cache.keys())[0]]
                if is_pdf(filee):                
                    base64_pdf = base64.b64encode(filee).decode('utf-8')
                    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="550" height="800" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
                elif  is_image(filee):
                    img = Image.open(io.BytesIO(filee)) 
                    st.image(img)
            else:
                col1, col2, col3 = st.columns(3)
                # Buttons
                if col1.button("Previous", key="prev_page"):
                    st.session_state.page_slider=max(st.session_state.page_slider-1,0)
                if col3.button("Next", key="next_page"):
                    st.session_state.page_slider=min(st.session_state.page_slider+1,page_count-1)
                # Page slider
                col2.slider("Page Slider", min_value=0, max_value=page_count-1, key="page_slider")
                # Rendering pdf page 
                filee=item_cache[list(item_cache.keys())[st.session_state.page_slider]]   
                st.write(f"**{list(item_cache.keys())[st.session_state.page_slider]}**")
                if is_pdf(filee):                
                    base64_pdf = base64.b64encode(filee).decode('utf-8')
                    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="550" height="800" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
                elif  is_image(filee):
                    img = Image.open(io.BytesIO(filee)) 
                    st.image(img)
        with colm2:            
            tab2, tab3, tab4, tab5 = st.tabs(["**Extracted Text**", "**QnA**","**Confidence Score**", "**Ground Truth**"])
            with tab2.container(height=800,border=False):
                if st.button('Extract',type="primary",key='summ'): 
                    results, errors, result_string = process_files(params['file_processed'])  
                    if errors:
                        st.write(errors)
                    st.session_state['page_summ']=result_string 
                    st.session_state['conf'] = results
              
                if st.session_state['page_summ']:
                    # st.write(st.session_state['page_summ'].keys())
                    file_list = list(st.session_state['page_summ'].keys())
                    tab_objects = st.tabs([f"**{x}**" for x in st.session_state['page_summ']])
                    
                    for ids,tabs in enumerate(tab_objects):  
                        # st.write(tabs)
                        with tabs:                            
                            container_tab=st.empty()                            
                            container_tab.markdown(st.session_state['page_summ'][file_list[ids]],unsafe_allow_html=True) 
                            col2a, col2b,col2c,col2d, col2e= st.columns(5, gap='small')
                   
                            if col2a.button('Edit',type="primary",key=f'editorrr{ids}'): 
                               container_tab.text_area(label = "", value = st.session_state['page_summ'][file_list[ids]], 
                                                                                         height = 500, key = f"text_area{ids}")             
                           
                            if col2d.button('Save Edit',type="primary",key=f'saver{ids}'):    
                                if f"text_area{ids}" in st.session_state:
                                    st.session_state['page_summ'][file_list[ids]] = st.session_state[f"text_area{ids}"]
                                    save_string_to_s3(st.session_state['page_summ'][file_list[ids]], BUCKET, f"textract_gt/{file_list[ids]}.txt")
                                    st.rerun()
                                else:
                                    st.error("Nothing to save. Make edits before saving")
                                
            with tab3.container(height=500,border=False):   
                if params["session_id"].strip():
                    st.session_state.messages, params['chat_histories']=get_chat_historie_for_streamlit(params)

                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"].replace("$", "\$"),unsafe_allow_html=True )
                   
                if prompt := st.chat_input("Whats up?"):
                    st.session_state.messages.append({"role": "user", "content": prompt})        
                    with st.chat_message("user"): 
                        st.markdown(prompt,unsafe_allow_html=True )

                    with st.chat_message("assistant"): 
                        message_placeholder = st.empty()
                        params["prompt"]=prompt
                        params['doc']=st.session_state['page_summ']
                        answer=query_llm(params, message_placeholder)
                        message_placeholder.markdown(answer.replace("$", "\$"),unsafe_allow_html=True )
                        st.session_state.messages.append({"role": "assistant", "content": answer}) 
                    st.rerun()
            with tab4.container(height=1000,border=False):        
                if st.session_state['conf']: 
                    tab_object_1 = st.tabs([f"**{x}**" for x in st.session_state['conf']])
                    for ids,tabss in enumerate(tab_object_1):                        
                        with tabss:                            
                            container_tab_2=st.empty()                            
                            container_tab_2.dataframe(st.session_state['conf'][file_list[ids]])
            with tab5.container(height=1000,border=False):        
                if st.session_state['gt']: 
                    json_list = list(st.session_state['gt'].keys())
                    tab_object_2 = st.tabs([f"**{x}**" for x in st.session_state['gt']])
                    for idss,tabsss in enumerate(tab_object_2):                        
                        with tabsss:                            
                            container_tab_3=st.empty()            
                            try:
                                file_cont=json.loads(st.session_state['gt'][json_list[idss]])
                            except:
                                file_cont=st.session_state['gt'][json_list[idss]]
                            container_tab_3.write(file_cont)
    else:
        st.session_state['conf'] = ""
        st.session_state['page_summ'] = ""      


def save_string_to_s3(string_data, bucket_name, object_key):
    """
    Save a string to an S3 bucket.

    Args:
    string_data (str): The string to save
    bucket_name (str): The name of the S3 bucket
    object_key (str): The object key (file path) in the S3 bucket

    Returns:
    bool: True if successful, False otherwise
    """
    # Create an S3 client
    s3_client = boto3.client('s3')

    try:
        # Convert string to bytes
        binary_data = string_data.encode('utf-8')

        # Create a file-like object from the bytes
        file_obj = io.BytesIO(binary_data)

        # Upload the file to S3
        s3_client.upload_fileobj(file_obj, bucket_name, object_key)

        st.write(f"Successfully saved string to s3://{bucket_name}/{object_key}")
        return True
    except Exception as e:
        st.error(f"Error saving string to S3: {str(e)}")
        return False


def list_csv_xlsx_in_s3_folder(bucket_name, folder_path):
    """
    List all CSV and XLSX files in a specified S3 folder.

    :param bucket_name: Name of the S3 bucket
    :param folder_path: Path to the folder in the S3 bucket
    :return: List of CSV and XLSX file names in the folder
    """
    s3 = boto3.client('s3')
    documents = []

    try:
        # Ensure the folder path ends with a '/'
        if not folder_path.endswith('/'):
            folder_path += '/'

        # List objects in the specified folder
        paginator = s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=folder_path)

        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Get the file name
                    file_name = obj['Key']

                    # Check if the file is a PNG, PDF, TIF, or JPG
                    if file_name.lower().endswith(INPUT_EXT):
                        documents.append(os.path.basename(file_name))
           
                    

        return documents

    except ClientError as e:
        print(f"An error occurred: {e}")
        return []

def app_sidebar():
    with st.sidebar:
        st.metric(label="Bedrock Session Cost", value=f"${round(st.session_state['cost'],2)}") 
        st.metric(label="Textract Session Cost", value=f"${round(textract_pricing(st.session_state['data_pages']),2)}")         
        st.write("-----")       
        models=[ "llama3-2-90b", "llama3-2-11b", "llama3-2-3b","llama3-1-70b", "llama3-1-8b"]        
        model=st.selectbox('**Model**', models,)
        params={"model":model}       
        user_chat_id=get_session_ids_by_user(DYNAMODB_TABLE, st.session_state['userid'])
        dict_items = list(user_chat_id.items())
        dict_items.insert(0, (st.session_state['user_sess'],""))      
        user_chat_id = dict(dict_items)
        chat_items=st.selectbox("**Chat sessions**",user_chat_id.values(),key="chat_sessions")
        session_id=get_key_from_value(user_chat_id, chat_items)
        bucket_items=list_csv_xlsx_in_s3_folder(INPUT_BUCKET, INPUT_S3_PATH)
        bucket_objects=st.multiselect("**Files**",bucket_items,key="objector",default=None)

        if "claude-3" in model:
            process_images=st.selectbox("**Document Processor**",["Textract"],key="processor")
        else:
            process_images="Textract"
        uploaded_files = st.file_uploader("Upload PDF or image files", type=["pdf", "png", "jpg", "jpeg","tif","json","txt"], accept_multiple_files=True)
        params={"model":model, "session_id":session_id, "chat_item":chat_items,"file_item":uploaded_files, "processor":process_images,'s3_objects':bucket_objects }
        return params

    
    
def main():
    params=app_sidebar()
    page_summary(params)
   
    
if __name__ == '__main__':
    main()   
