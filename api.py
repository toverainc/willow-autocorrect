from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from jsonget import json_get, json_get_default
from typing import Optional
import json
import logging
import requests

from datetime import datetime
from zoneinfo import ZoneInfo
from decouple import config
import typesense

HA_URL = config('HA_URL', default="http://homeassistant.local:8123", cast=str)
HA_TOKEN = config('HA_TOKEN', default=None, cast=str)
LOG_LEVEL = config('LOG_LEVEL', default="debug", cast=str)
TGI_URL = config('TGI_URL', default=None, cast=str)

# TGI model stuff
MAX_NEW_TOKENS = config('MAX_NEW_TOKENS', default=40, cast=int)
TEMPERATURE = config('TEMPERATURE', default=0.7, cast=float)
TOP_K = config('TOP_K', default=40, cast=int)
TOP_P = config('TOP_P', default=0.1, cast=float)
REPETITION_PENALTY = config('REPETITION_PENALTY', default=1.176, cast=float)
SYS_PROMPT = config('SYS_PROMPT', default="", cast=str)

# This doesn't seem to be getting from docker to here - FIX
TYPESENSE_API_KEY = config('TYPESENSE_API_KEY', default='testing', cast=str)
TYPESENSE_HOST = config('TYPESENSE_HOST', default='localhost', cast=str)
TYPESENSE_PORT = config('TYPESENSE_PORT', default=9001, cast=int)

HA_URL = f'{HA_URL}/api/conversation/process'
HA_TOKEN = f'Bearer {HA_TOKEN}'
TGI_URL = f'{TGI_URL}/generate'

print(f'Config vars: {HA_URL} {HA_TOKEN} {TGI_URL}')

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Typesense
ts_client = typesense.Client({
  'nodes': [{
    'host': TYPESENSE_HOST,
    'port': TYPESENSE_PORT,
    'protocol': 'http'
  }],
  'api_key': TYPESENSE_API_KEY,
  'connection_timeout_seconds': 2
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

ha_headers = {
    "Authorization": HA_TOKEN,
}

llm_headers = {
    "Content-Type": "application/json",
}

def do_llm(text):
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
    response = requests.post(TGI_URL, headers=llm_headers, json=data)
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

def parse_ha(response):
    log.debug(str(response))
    speech = json_get(response, "/response/speech/plain/speech", str)
    status = json_get(response, "/response/response_type", str)
    code = json_get_default(response, "/response/data/code", "intent_match")

    if code == "no_intent_match":
        success = False
    else:
        success = True
    
    return success, speech

def do_ha(text, language):
    data = {"text": text, "language": language}
    response = requests.post(HA_URL, headers=ha_headers, json=data)
    log.debug(str(response))
    response = response.json()
    success, speech = parse_ha(response)
    if success:
        response = speech
    else:
        response = do_llm(text)

    return response

def wac_search(command):
    search_parameters = {
  'q'         : command,
  'query_by'  : 'command',
  'sort_by'   : '_text_match:desc',
  'num_typos' : 5,
  'per_page' : 5
}
    return ts_client.collections['commands'].documents.search(search_parameters)

def get_first_command(results, json = False):
    try:
        first_result = json_get(results, "/hits[0]/document/command")
        first_result_source = json_get(results, "/hits[0]/document/source")
    except:
        first_result = None
    
    if json:
        return {'command': first_result, 'source': first_result_source}
    else:
        return first_result

def do_wac_search(command, raw = False, json = False):
    # Search
    time_start = datetime.now()
    results = wac_search(command)
    time_end = datetime.now()
    search_time = time_end - time_start
    search_time_milliseconds = search_time.total_seconds() * 1000
    #pretty = json.dumps(results, indent=2)
    #log.debug(pretty)
    log.info('WAC search took ' + str(search_time_milliseconds) + ' ms')
    if raw:
        return results
    else:
        results = get_first_command(results, json)
        log.info(f'WAC matched: {results}')
        return results

def match_route(text, language):
    if "what time" in text:
        return get_time()
    elif "date" in text:
        return get_date()
    else:
        return do_ha(text, language)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/api/search", summary="WAC Search", response_description="WAC Search Results")
async def api_search(request: Request, command: str, raw: Optional[bool] = False, json: Optional[bool] = True):
    log.info(f"Doing WAC search for command '{command}'")
    results = do_wac_search(command, raw, json)
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