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
TYPESENSE_SLOW_TIMEOUT = config(
    'TYPESENSE_SLOW_TIMEOUT', default=120, cast=int)
TYPESENSE_TIMEOUT = config('TYPESENSE_TIMEOUT', default=1, cast=int)

# HA
HA_URL = f'{HA_URL}/api/conversation/process'
HA_TOKEN = f'Bearer {HA_TOKEN}'

# Search distance for text string distance
SEARCH_DISTANCE = config(
    'SEARCH_DISTANCE', default=2, cast=int)

# The number of matching tokens to consider a successful WAC search
# More tokens = closer match
TOKEN_MATCH_THRESHOLD = config(
    'TOKEN_MATCH_THRESHOLD', default=3, cast=int)

# The number of matching tokens to consider a successful WAC search
# larger float = further away (less close in meaning)
VECTOR_DISTANCE_THRESHOLD = config(
    'VECTOR_DISTANCE_THRESHOLD', default=0.5, cast=float)

# Hybrid/fusion search threshold.
# larger float = closer (reverse of vector distance)
FUSION_SCORE_THRESHOLD = config(
    'FUSION_SCORE_THRESHOLD', default=0.5, cast=float)

# Typesense embedding model to use
TYPESENSE_SEMANTIC_MODEL = config(
    'TYPESENSE_SEMANTIC_MODEL', default='all-MiniLM-L12-v2', cast=str)

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

# For operations that take a while like initial vector schema and model download
slow_typesense_client = typesense.Client({
    'nodes': [{
        'host': TYPESENSE_HOST,
        'port': TYPESENSE_PORT,
        'protocol': TYPESENSE_PROTOCOL,
    }],
    'api_key': TYPESENSE_API_KEY,
    'connection_timeout_seconds': TYPESENSE_SLOW_TIMEOUT
})

# The schema for WAC commands - you really do not want to mess with this
wac_commands_schema = {
    'name': COLLECTION,
    'fields': [
        {'name': 'command', 'type': 'string', "sort": True},
        {'name': 'rank', 'type': 'float'},
        {'name': 'is_alias', 'type': 'bool', 'optional': True},
        {'name': 'alias', 'type': 'string', 'optional': True, "sort": True},
        {'name': 'accuracy', 'type': 'float', 'optional': True},
        {'name': 'source', 'type': 'string', 'optional': True, "sort": True},
        {
            "name": "all-MiniLM-L12-v2",
            "type": "float[]",
            "embed": {
                "from": [
                    "command"
                ],
                "model_config": {
                    "model_name": "ts/all-MiniLM-L12-v2"
                }
            }
        },
        {
            "name": "multilingual-e5-small",
            "type": "float[]",
            "embed": {
                "from": [
                    "command"
                ],
                "model_config": {
                    "model_name": "ts/multilingual-e5-small"
                }
            }
        },
        {
            "name": "gte-small",
            "type": "float[]",
            "embed": {
                "from": [
                    "command"
                ],
                "model_config": {
                    "model_name": "ts/gte-small"
                }
            }
        },
    ],
    'default_sorting_field': 'rank',
    "token_separators": [".", "-"]
}


def init_typesense():
    try:
        typesense_client.collections[COLLECTION].retrieve()
    except:
        log.info(
            f"WAC collection '{COLLECTION}' not found - initializing with timeout {TYPESENSE_SLOW_TIMEOUT} - please wait.")
        # Hack around slow initial schema generation because of model download
        slow_typesense_client.collections.create(wac_commands_schema)
        log.info(f"WAC collection '{COLLECTION}' initialized")

    log.info(f"Connected to WAC Typesense host '{TYPESENSE_HOST}'")


@app.on_event("startup")
async def startup_event():
    init_typesense()

# WAC Search


def wac_search(command, exact_match=False, distance=SEARCH_DISTANCE, num_results=5, raw=False, token_match_threshold=TOKEN_MATCH_THRESHOLD, semantic="off", vector_distance_threshold=VECTOR_DISTANCE_THRESHOLD, fusion_score_threshold=FUSION_SCORE_THRESHOLD):
    # Set fail by default
    success = False
    wac_command = command
    tokens_matched = 0
    vector_distance = 1.0

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
    if exact_match is True:
        log.info(f"Doing exact match WAC Search")
        wac_search_parameters.update({'filter_by': f'command:={command}'})

    # Support per request semantic or hybrid semantic search
    if semantic == "hybrid":
        log.info(
            f"Doing hybrid semantic WAC Search with model {TYPESENSE_SEMANTIC_MODEL}")
        wac_search_parameters.update(
            {'query_by': f'{TYPESENSE_SEMANTIC_MODEL},command'})
    elif semantic == "on":
        log.info(
            f"Doing semantic WAC Search with model {TYPESENSE_SEMANTIC_MODEL}")
        wac_search_parameters.update(
            {'query_by': f'{TYPESENSE_SEMANTIC_MODEL}'})

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

        # Semantic handling
        log.info(f"Trying scoring evaluation with top match '{wac_command}'")
        if semantic == "on":
            vector_distance = json_get(
                wac_search_result, "/hits[0]/vector_distance")

            if vector_distance <= vector_distance_threshold:
                log.info(
                    f"WAC Semantic Search passed vector distance threshold {vector_distance_threshold} with result {vector_distance} from source {source}")
                success = True
            else:
                log.info(
                    f"WAC Semantic Search didn't meet vector distance threshold {vector_distance_threshold} with result {vector_distance} from source {source}")
        elif semantic == "hybrid":
            fusion_score = json_get(
                wac_search_result, "/hits[0]/hybrid_search_info/rank_fusion_score")
            if fusion_score >= fusion_score_threshold:
                log.info(
                    f"WAC Semantic Hybrid Search passed fusion score threshold {fusion_score_threshold} with result {fusion_score} from source {source}")
                success = True
            else:
                log.info(
                    f"WAC Semantic Hybrid Search didn't meet fusion score threshold {fusion_score_threshold} with result {fusion_score} from source {source}")
        # Regular old token match
        else:
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
    learned = False
    try:
        log.info(f"Searching WAC before adding command '{command}'")
        wac_exact_search_status, wac_command = wac_search(
            command, exact_match=True)
        if wac_exact_search_status is True:
            log.info('Refusing to add duplicate command')
            return learned

        command_json = {
            'command': command,
            'rank': 1.0,
            'source': 'autolearn',
        }
        # Use create to update in real time
        typesense_client.collections[COLLECTION].documents.create(command_json)
        log.info(f"Added WAC command '{command}'")
        learned = True
    except:
        log.error(f"WAC Add for command '{command}' failed!")

    return learned


# Request coming from proxy


def api_post_proxy_handler(command, language, distance=SEARCH_DISTANCE, token_match_threshold=TOKEN_MATCH_THRESHOLD, exact_match=False, semantic="off", vector_distance_threshold=VECTOR_DISTANCE_THRESHOLD, fusion_score_threshold=FUSION_SCORE_THRESHOLD):

    log.info(
        f"Processing proxy request for command '{command}' with distance {distance} token match threshold {token_match_threshold} exact match {exact_match} semantic {semantic} with vector distance threshold {vector_distance_threshold} and hybrid threshold {fusion_score_threshold}")
    # Init speech for when all else goes wrong
    speech = "Sorry, I can't find that command."
    # Default to command isn't learned
    learned = False

    try:
        ha_data = {"text": command, "language": language}
        log.info(f"Trying initial HA intent match '{command}'")
        ha_response = requests.post(HA_URL, headers=ha_headers, json=ha_data)
        ha_response = ha_response.json()
        code = json_get_default(
            ha_response, "/response/data/code", "intent_match")

        if code == "no_intent_match":
            log.info(f"No Initial HA Intent Match for command '{command}'")
        else:
            log.info(f"Initial HA Intent Match for command '{command}'")
            learned = wac_add(command)
            # Set speech to HA response and return
            log.info('Setting speech to HA response')
            speech = json_get(
                ha_response, "/response/speech/plain/speech", str)
            if learned is True:
                speech = f"{speech} and learned command"
            return speech
    except:
        pass

    # Do WAC Search
    wac_success, wac_command = wac_search(command, exact_match=exact_match, distance=distance, num_results=5, raw=False,
                                          token_match_threshold=token_match_threshold, semantic=semantic, vector_distance_threshold=vector_distance_threshold, fusion_score_threshold=fusion_score_threshold)

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
            log.info(f"HA speech: '{speech}'")
            speech = f"{speech} with corrected command {wac_command}"
            log.info(f"Setting final speech to '{speech}'")
        except:
            pass

    log.info(f"Final speech response '{speech}'")
    return speech


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/api/search", summary="WAC Search", response_description="WAC Search")
async def api_get_wac(request: Request, command, distance: Optional[str] = SEARCH_DISTANCE, num_results: Optional[str] = 5, exact_match: Optional[bool] = False, semantic: Optional[str] = "off"):
    time_start = datetime.now()

    # Little fix for compatibility
    if semantic == "true":
        semantic = "on"
    elif semantic == "false":
        semantic = "off"

    results = wac_search(command, exact_match=exact_match,
                         distance=distance, num_results=num_results, raw=True, semantic=semantic)

    time_end = datetime.now()
    search_time = time_end - time_start
    search_time_milliseconds = search_time.total_seconds() * 1000
    log.info('WAC search took ' + str(search_time_milliseconds) + ' ms')
    return JSONResponse(content=results)


@app.post("/api/proxy")
async def api_post_proxy(request: Request, distance: Optional[int] = SEARCH_DISTANCE, token_match_threshold: Optional[int] = TOKEN_MATCH_THRESHOLD, exact_match: Optional[bool] = False, semantic: Optional[str] = "off", vector_distance_threshold: Optional[float] = VECTOR_DISTANCE_THRESHOLD, fusion_score_threshold: Optional[float] = FUSION_SCORE_THRESHOLD):
    time_start = datetime.now()
    request_json = await request.json()
    language = json_get_default(request_json, "/language", "en")
    text = json_get(request_json, "/text")

    # Little fix for compatibility
    if semantic == "true":
        semantic = "on"
    elif semantic == "false":
        semantic = "off"

    response = api_post_proxy_handler(text, language, distance=distance, token_match_threshold=token_match_threshold,
                                      exact_match=exact_match, semantic=semantic, vector_distance_threshold=vector_distance_threshold, fusion_score_threshold=fusion_score_threshold)
    time_end = datetime.now()
    search_time = time_end - time_start
    search_time_milliseconds = search_time.total_seconds() * 1000
    log.info('WAC proxy took ' + str(search_time_milliseconds) + ' ms')
    return PlainTextResponse(content=response)
