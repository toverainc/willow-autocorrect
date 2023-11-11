from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from jsonget import json_get, json_get_default
from typing import Optional
import json
import logging
import requests

# For typesense-server
import subprocess
import time
import threading

from datetime import datetime
from zoneinfo import ZoneInfo
from decouple import config
import typesense

HA_URL = config('HA_URL', default="http://homeassistant.local:8123", cast=str)
HA_TOKEN = config('HA_TOKEN', default=None, cast=str)
LOG_LEVEL = config('LOG_LEVEL', default="debug", cast=str)
TGI_URL = config(f'TGI_URL', default=None, cast=str)

# TGI model stuff
MAX_NEW_TOKENS = config('MAX_NEW_TOKENS', default=40, cast=int)
TEMPERATURE = config('TEMPERATURE', default=0.7, cast=float)
TOP_K = config('TOP_K', default=40, cast=int)
TOP_P = config('TOP_P', default=0.1, cast=float)
REPETITION_PENALTY = config('REPETITION_PENALTY', default=1.176, cast=float)
SYS_PROMPT = config('SYS_PROMPT', default="", cast=str)

# This doesn't seem to be getting from docker to here - FIX
TYPESENSE_API_KEY = config('TYPESENSE_API_KEY', default='testing', cast=str)
TYPESENSE_HOST = config('TYPESENSE_HOST', default='127.0.0.1', cast=str)
TYPESENSE_PORT = config('TYPESENSE_PORT', default=8108, cast=int)

HA_URL = f'{HA_URL}/api/conversation/process'
HA_TOKEN = f'Bearer {HA_TOKEN}'

print(f'Config vars: {HA_URL} {HA_TOKEN} {TGI_URL}')

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Typesense
def start_typesense():
    def run(job):
        proc = subprocess.Popen(job)
        proc.wait()
        return proc

    # Fix this in prod to use some kind of unique/user provided/etc key. Not that big of a deal but...
    job = ['/usr/local/sbin/typesense-server', '--data-dir=/app/data/typesense', '--api-key=testing', '--log-dir=/dev/shm']

    # server thread will remain active as long as FastAPI thread is running
    thread = threading.Thread(name='typesense-server', target=run, args=(job,), daemon=True)
    thread.start()
    #log.info(f'Waiting {TYPESENSE_SLEEP} seconds for typesense...')
    #time.sleep(TYPESENSE_SLEEP)

ts_client = typesense.Client({
  'nodes': [{
    'host': TYPESENSE_HOST,
    'port': TYPESENSE_PORT,
    'protocol': 'http'
  }],
  'api_key': TYPESENSE_API_KEY,
  'connection_timeout_seconds': 1
})

app = FastAPI(title="WAC Proxy",
              description="Make voice better",
              version="0.1",
              openapi_url="/openapi.json",
              docs_url="/docs",
              redoc_url="/redoc")

log = logging.getLogger("WAC")
try:
    log.setLevel(LOG_LEVEL).upper()
except:
    pass

# Effectively disable for now in dev with uvicorn --reload
@app.on_event("startup")
async def startup_event():
    #start_typesense()
    pass

ha_headers = {
    "Authorization": HA_TOKEN,
}

llm_headers = {
    "Content-Type": "application/json",
}

def do_tgi(text, speech):
    if TGI_URL == 'None':
        return speech

    text = f'{SYS_PROMPT} {text}'
    log.info(f'Doing LLM request with text {text}')
    data = {
    'inputs': text,
    'parameters': {
        'max_new_tokens': MAX_NEW_TOKENS,
        'details': False,
        'temperature': TEMPERATURE,
        'top_k': TOP_K,
        'top_p': TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        'watermark': False,
    },
}
    time_start = datetime.now()
    response = requests.post(f'{TGI_URL}/generate', headers=llm_headers, json=data)
    time_end = datetime.now()
    infer_time = time_end - time_start
    infer_time_milliseconds = infer_time.total_seconds() * 1000
    log.info('LLM inference took ' + str(infer_time_milliseconds) + ' ms')

    response = response.json()
    response = json_get(response, "/generated_text", str)
    # LLama likes to throw all kinds of weird stuff in responses
    response = response.strip()
    response = response.replace('Answer: ', '')
    response = response.replace("I'm happy to help! ", '')
    log.info(f'LLM responded with {response}')
    return response

# KK HA Stuff and what-not
def get_time():
    now = datetime.now(tz=ZoneInfo("America/Chicago"))
    dt_string = now.strftime("%I %M")
    return dt_string

def get_date():
    now = datetime.now(tz=ZoneInfo("America/Chicago"))
    dt_string = now.strftime("%A %B %d")
    return dt_string

def learn_command(command):
    results = do_wac_search(command, raw = False, json = False, distance = 5, num_results = 5)
    if results is None:
        log.info(f"Adding command '{command}' to WAC autolearn")
        command_json = {
            'command': command,
            'rank': 1.0,
            'source': 'autolearn',
            }
        # Use create to update in real time
        ts_client.collections['commands'].documents.create(command_json)
    else:
        log.info(f"WAC already knows command '{command}'")

def parse_ha(response):
    speech = "No Willow auto correct match"
    success = False
    if response is not None:
        log.debug(str(response))
        speech = json_get(response, "/response/speech/plain/speech", str)
        status = json_get(response, "/response/response_type", str)
        code = json_get_default(response, "/response/data/code", "intent_match")

        if code == "no_intent_match":
            speech = "No Willow auto correct match"
            success = False
        else:
            success = True
    
    return success, speech

def do_ha(text, language):
    response = None
    data = {"text": text, "language": language}
    ha_response = requests.post(HA_URL, headers=ha_headers, json=data)
    log.debug(str(ha_response))
    ha_response_json = ha_response.json()
    success, speech = parse_ha(ha_response_json)
    if success:
        log.info(f"HA Returned intent match on command '{text}' - learning")
        learn_command(text)
        response = speech
    #else:
    #    response = do_tgi(text, speech)

    return response

def wac_search(command, distance = 2, num_results = 5):
    log.info(f'WAC Search distance is {distance}')
    search_parameters = {
  'q'         : command,
  'query_by'  : 'command',
  'sort_by'   : '_text_match:desc,rank:desc',
  'text_match_type': 'max_score',
  'prioritize_token_position': False,
  'drop_tokens_threshold': 1,
  'typo_tokens_threshold': 2,
  'split_join_tokens': 'fallback',
  'num_typos' : distance,
  'min_len_1typo': 1,
  'min_len_2typo': 1,
  'per_page' : num_results
}
    return ts_client.collections['commands'].documents.search(search_parameters)

def get_first_command(command, results, distance, search_time, json = False):
    first_result = None
    first_result_source = None
    tokens_matched = 0
    text_score = 0
    try:
        text_score = json_get(results, "/hits[0]/text_match")
        tokens_matched = json_get(results, "/hits[0]/text_match_info/tokens_matched")
        if tokens_matched <= 2:
            return
        first_result = json_get(results, "/hits[0]/document/command")
        first_result_source = json_get(results, "/hits[0]/document/source")
    except:
        pass

    if json:
        return {'input': command, 'command': first_result, 'source': first_result_source, 'distance': distance, 'tokens_matched': tokens_matched, 'score': text_score, 'search_time': search_time}
    else:
        return first_result

def do_wac_search(command, raw = False, json = False, distance = 5, num_results = 5):
    # Search
    time_start = datetime.now()
    results = wac_search(command, distance, num_results)
    time_end = datetime.now()
    search_time = time_end - time_start
    search_time_milliseconds = search_time.total_seconds() * 1000
    #pretty = json.dumps(results, indent=2)
    #log.debug(pretty)
    log.info('WAC search took ' + str(search_time_milliseconds) + ' ms')
    if raw:
        return results
    else:
        results = get_first_command(command, results, distance, search_time_milliseconds, json)
        log.info(f'WAC matched: {results}')
        return results

def match_route(text, language):
    ha_result = do_ha(text, language)
    if ha_result:
        return ha_result
    else:
        text = do_wac_search(text, raw = False, json = False, distance = 5, num_results = 5)
        ha_result = do_ha(text, language)
    return ha_result

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/api/search", summary="WAC Search", response_description="WAC Search Results")
async def api_search(request: Request, command: str, raw: Optional[bool] = False, json: Optional[bool] = True, distance: Optional[str] = 2, num_results: Optional[str] = 5):
    log.info(f"Doing WAC search for command '{command}'")
    results = do_wac_search(command, raw, json, distance, num_results)
    if json:
        return JSONResponse(content=results)
    else:
        results = (str(results))
        return PlainTextResponse(content=results)

@app.post("/proxy")
async def do_proxy(request: Request):
    request_json = await request.json()
    language = request_json['language']
    text = request_json['text']
    response = match_route(text, language)

    return PlainTextResponse(content=response)