from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from jsonget import json_get, json_get_default
from typing import Optional
import json
import logging
import requests

from datetime import datetime
from decouple import config
import typesense

HA_URL = config('HA_URL', default="http://homeassistant.local:8123", cast=str)
HA_TOKEN = config('HA_TOKEN', default=None, cast=str)
LOG_LEVEL = config('LOG_LEVEL', default="debug", cast=str)
TGI_URL = config(f'TGI_URL', default=None, cast=str)

# Typesense config vars
TYPESENSE_API_KEY = config('TYPESENSE_API_KEY', default='testing', cast=str)
TYPESENSE_HOST = config('TYPESENSE_HOST', default='127.0.0.1', cast=str)
TYPESENSE_PORT = config('TYPESENSE_PORT', default=8108, cast=int)
TYPESENSE_PROTOCOL = config('TYPESENSE_PROTOCOL', default='http', cast=str)
TYPESENSE_TIMEOUT = config('TYPESENSE_TIMEOUT', default=1, cast=int)

HA_URL = f'{HA_URL}/api/conversation/process'
HA_TOKEN = f'Bearer {HA_TOKEN}'

# The number of matching tokens to consider a successful WAC search
# More tokens = closer match
TOKEN_MATCH_THRESHOLD = config(
    'TOKEN_MATCH_THRESHOLD', default=3, cast=int)

# The typesense collection to use
COLLECTION = config(
    'COLLECTION', default='commands', cast=str)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

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

# Basic stuff we need
ha_headers = {
    "Authorization": HA_TOKEN,
}

# The real WAC MVP
typesense_client = typesense.Client({
    'nodes': [{
        'host': TYPESENSE_HOST,
        'port': TYPESENSE_PORT,
        'protocol': TYPESENSE_PROTOCOL,
    }],
    'api_key': TYPESENSE_API_KEY,
    'connection_timeout_seconds': TYPESENSE_TIMEOUT
})

# The schema for WAC commands - you really do not want to mess with this
wac_commands_schema = {
    'name': 'commands',
    'fields': [
        {'name': 'command', 'type': 'string', "sort": True},
        {'name': 'rank', 'type': 'float'},
        {'name': 'is_alias', 'type': 'bool', 'optional': True},
        {'name': 'alias', 'type': 'string', 'optional': True, "sort": True},
        {'name': 'accuracy', 'type': 'float', 'optional': True},
        {'name': 'source', 'type': 'string', 'optional': True, "sort": True},

    ],
    'default_sorting_field': 'rank',
    "token_separators": [".", "-"]
}


def init_typesense():
    try:
        typesense_client.collections[COLLECTION].retrieve()
    except:
        log.info(f'WAC collection {COLLECTION} not found - initializing')
        typesense_client.collections.create(wac_commands_schema)


@app.on_event("startup")
async def startup_event():
    init_typesense()

# WAC Search


def wac_search(command, exact_match=False, distance=2, num_results=5, raw=False, token_match_threshold=TOKEN_MATCH_THRESHOLD):
    # Set fail by default
    success = False
    wac_command = command
    tokens_matched = 0

    # Do not change these unless you know what you are doing
    wac_search_parameters = {
        'q': command,
        'query_by': 'command',
        'sort_by': '_text_match:desc,rank:desc',
        'text_match_type': 'max_score',
        'prioritize_token_position': False,
        'drop_tokens_threshold': 1,
        'typo_tokens_threshold': 2,
        'split_join_tokens': 'fallback',
        'num_typos': distance,
        'min_len_1typo': 1,
        'min_len_2typo': 1,
        'per_page': num_results
    }
    if exact_match:
        log.info(f"Doing exact match WAC Search")
        wac_search_parameters.update({'filter_by': f'command:={command}'})

    # Try WAC search
    try:
        log.info(
            f"Doing WAC Search for command '{command}' with distance {distance}")
        wac_search_result = typesense_client.collections[COLLECTION].documents.search(
            wac_search_parameters)
        # For management API
        if raw:
            return wac_search_result
        text_score = json_get(wac_search_result, "/hits[0]/text_match")
        tokens_matched = json_get(
            wac_search_result, "/hits[0]/text_match_info/tokens_matched")
        wac_command = json_get(wac_search_result, "/hits[0]/document/command")
        source = json_get(wac_search_result, "/hits[0]/document/source")

        if tokens_matched >= token_match_threshold:
            log.info(
                f"WAC Search passed token threshold {token_match_threshold} with result {tokens_matched} from source {source}")
            success = True
        else:
            log.info(
                f"WAC Search didn't meet threshold {token_match_threshold} with result {tokens_matched} from source {source}")
    except:
        log.info(f"WAC Search for command '{command}' failed")

    return success, wac_command

# WAC Add


def wac_add(command):
    log.info(f"Doing WAC Add for command '{command}'")
    try:
        log.info(f"Search WAC before adding command '{command}'")
        wac_exact_search_status, wac_command = wac_search(
            command, exact_match=True)
        if wac_exact_search_status is True:
            log.info('Not adding duplicate command')
            return

        command_json = {
            'command': command,
            'rank': 1.0,
            'source': 'autolearn',
        }
        # Use create to update in real time
        typesense_client.collections[COLLECTION].documents.create(command_json)
        log.info(f"Added WAC command '{command}'")
    except:
        log.error(f"WAC Add for command '{command}' failed!")

    return


# Request coming from proxy


def api_post_proxy_handler(command, language, token_match_threshold=TOKEN_MATCH_THRESHOLD):

    # Init speech for when all else goes wrong
    speech = "Sorry, I don't know that command."

    try:
        ha_data = {"text": command, "language": language}
        ha_response = requests.post(HA_URL, headers=ha_headers, json=ha_data)
        ha_response = ha_response.json()
        code = json_get_default(
            ha_response, "/response/data/code", "intent_match")

        if code == "no_intent_match":
            log.info(f"No Initial HA Intent Match for command '{command}'")
        else:
            log.info(f"Initial HA Intent Match for command '{command}'")
            wac_add(command)
            # Set speech to HA response and return
            log.info('Setting speech to HA response')
            speech = json_get(
                ha_response, "/response/speech/plain/speech", str)
            return speech
    except:
        pass

    # Do WAC Search
    wac_success, wac_command = wac_search(
        command, exact_match=False, distance=2, num_results=5, token_match_threshold=token_match_threshold)

    if wac_success:

        # Re-run HA with WAC Command
        try:
            log.info(
                f"Attempting WAC HA Intent Match with command '{wac_command}' from provided command '{command}'")
            ha_data = {"text": wac_command, "language": language}
            ha_response = requests.post(
                HA_URL, headers=ha_headers, json=ha_data)
            ha_response = ha_response.json()
            code = json_get_default(
                ha_response, "/response/data/code", "intent_match")

            if code == "no_intent_match":
                log.info(f"No WAC Command HA Intent Match: '{wac_command}'")
            else:
                log.info(f"WAC Command HA Intent Match: '{wac_command}'")

            # Set speech to HA response - whatever it is at this point
            speech = json_get(
                ha_response, "/response/speech/plain/speech", str)
            log.info(f"Setting speech to HA response '{speech}'")

        except:
            pass

    log.info(f"Final speech response '{speech}'")
    return speech


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/api/search", summary="WAC Search", response_description="WAC Search")
async def api_get_wac(request: Request, command, distance: Optional[str] = 2, num_results: Optional[str] = 5, exact_match: Optional[bool] = False):
    time_start = datetime.now()

    results = wac_search(command, exact_match=exact_match,
                         distance=distance, num_results=num_results, raw=True)

    time_end = datetime.now()
    search_time = time_end - time_start
    search_time_milliseconds = search_time.total_seconds() * 1000
    log.info('WAC search took ' + str(search_time_milliseconds) + ' ms')
    return JSONResponse(content=results)


@app.post("/api/proxy")
async def api_post_proxy(request: Request):
    try:
        time_start = datetime.now()
        request_json = await request.json()
        language = json_get_default(request_json, "/language", "en")
        text = json_get(request_json, "/text")
        token_max_threshold = json_get_default(
            request_json, "/token_match_threshold", TOKEN_MATCH_THRESHOLD)
        response = api_post_proxy_handler(text, language, token_max_threshold)
        time_end = datetime.now()
        search_time = time_end - time_start
        search_time_milliseconds = search_time.total_seconds() * 1000
        log.info('WAC proxy took ' + str(search_time_milliseconds) + ' ms')
        return PlainTextResponse(content=response)
    except:
        raise HTTPException(status_code=500, detail="WAC Failed")
