import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from jsonget import json_get, json_get_default
from pydantic import BaseModel
from typing import Optional
import json
import logging
import requests
import websockets

from datetime import datetime
from decouple import config
import typesense

# For typesense-server when not in dev mode
import subprocess
import threading
import time

HA_HOST = config(
    'HA_HOST', default="homeassistant.local", cast=str)
HA_PORT = config(
    'HA_PORT', default=8123, cast=int)
HA_TOKEN = config('HA_TOKEN', default=None, cast=str)
LOG_LEVEL = config('LOG_LEVEL', default="info", cast=str).upper()
TGI_URL = config(f'TGI_URL', default=None, cast=str)

# Typesense config vars
TYPESENSE_API_KEY = config('TYPESENSE_API_KEY', default='testing', cast=str)
TYPESENSE_HOST = config('TYPESENSE_HOST', default='127.0.0.1', cast=str)
TYPESENSE_PORT = config('TYPESENSE_PORT', default=8108, cast=int)
TYPESENSE_PROTOCOL = config('TYPESENSE_PROTOCOL', default='http', cast=str)
TYPESENSE_SLOW_TIMEOUT = config(
    'TYPESENSE_SLOW_TIMEOUT', default=120, cast=int)
TYPESENSE_THREADS = config('TYPESENSE_THREADS', default=8, cast=int)
TYPESENSE_TIMEOUT = config('TYPESENSE_TIMEOUT', default=1, cast=int)

# "Prod" vs "dev"
RUN_MODE = config(f'RUN_MODE', default="prod", cast=str)
if RUN_MODE == "prod":
    TYPESENSE_HOST = "127.0.0.1"
    TYPESENSE_PORT = 8108
    TYPESENSE_PROTOCOL = "http"


# HA
HA_TOKEN_HTTP = f'Bearer {HA_TOKEN}'

HA_URL_HTTP = f"http://{HA_HOST}:{HA_PORT}"
HA_URL_WS = f"ws://{HA_HOST}:{HA_PORT}"

# Default number of search results and attempts
CORRECT_ATTEMPTS = config(
    'CORRECT_ATTEMPTS', default=1, cast=int)

# Search distance for text string distance
SEARCH_DISTANCE = config(
    'SEARCH_DISTANCE', default=2, cast=int)

# The number of matching tokens to consider a successful WAC search
# More tokens = closer match
TOKEN_MATCH_THRESHOLD = config(
    'TOKEN_MATCH_THRESHOLD', default=3, cast=int)

# The number of matching tokens to consider a successful WAC search
# larger float = further away (less close in meaning)
# NOTE: Different models have different score mechanisms
# This will likely need to get adjusted if you use models other than all-MiniLM-L12-v2
VECTOR_DISTANCE_THRESHOLD = config(
    'VECTOR_DISTANCE_THRESHOLD', default=0.29, cast=float)

# Hybrid/fusion search threshold.
# larger float = closer (reverse of vector distance)
HYBRID_SCORE_THRESHOLD = config(
    'HYBRID_SCORE_THRESHOLD', default=0.85, cast=float)

# Typesense embedding model to use
TYPESENSE_SEMANTIC_MODEL = config(
    'TYPESENSE_SEMANTIC_MODEL', default='all-MiniLM-L12-v2', cast=str)

# Default semantic mode
TYPESENSE_SEMANTIC_MODE = config(
    'TYPESENSE_SEMANTIC_MODE', default='hybrid', cast=str)

# The typesense collection to use
COLLECTION = config(
    'COLLECTION', default='commands', cast=str)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

log = logging.getLogger("WAC")
try:
    log.setLevel(LOG_LEVEL)
    log.info(f"Set log level {LOG_LEVEL}")
except Exception as e:
    log.exception(f"Set log level {LOG_LEVEL} failed with {e}")
    pass

# Typesense


def start_typesense():
    def run(job):
        proc = subprocess.Popen(job)
        proc.wait()
        return proc

    # Fix this in prod to use some kind of unique/user provided/etc key. Not that big of a deal but...
    job = ['/usr/local/sbin/typesense-server', '--data-dir=/app/data/ts',
           f'--api-key={TYPESENSE_API_KEY}', '--log-dir=/dev/shm', f'--thread-pool-size={TYPESENSE_THREADS}']

    # server thread will remain active as long as FastAPI thread is running
    thread = threading.Thread(name='typesense-server',
                              target=run, args=(job,), daemon=True)
    thread.start()


app = FastAPI(title="WAC Proxy",
              description="Willow Auto Correct REST Proxy",
              version="0.1",
              openapi_url="/openapi.json",
              docs_url="/",
              redoc_url="/redoc")

# Basic stuff we need
ha_headers = {
    "Authorization": HA_TOKEN_HTTP,
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
        {'name': 'timestamp', 'type': 'int64', 'optional': True},
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
    "token_separators": [",", ".", "-"]
}


def init_typesense():
    try:
        typesense_client.collections[COLLECTION].retrieve()
    except:
        log.info(
            f"WAC collection '{COLLECTION}' not found. Initializing with timeout {TYPESENSE_SLOW_TIMEOUT} - please wait.")
        # Hack around slow initial schema generation because of model download
        slow_typesense_client.collections.create(wac_commands_schema)
        log.info(f"WAC collection '{COLLECTION}' initialized")

    log.info(f"Connected to WAC Typesense host '{TYPESENSE_HOST}'")


@app.on_event("startup")
async def startup_event():
    app.command_endpoint = HomeAssistantWebSocketEndpoint(
        app, HA_URL_WS, HA_TOKEN)
    app.session_tracker = {}

    if RUN_MODE == "prod":
        log.info('Starting Typesense')
        start_typesense()
        log.info('Typesense started. Waiting for ready...')
        time.sleep(10)
    init_typesense()

# Add HA entities


def add_ha_entities():
    log.info('Adding entities from HA')
    entity_types = ['cover.', 'fan.', 'light.', 'switch.']

    url = f"{HA_URL_HTTP}/api/states"

    response = requests.get(url, headers=ha_headers)
    entities = response.json()

    devices = []

    for type in entity_types:
        for entity in entities:
            entity_id = entity['entity_id']

            if entity_id.startswith(type):
                attr = entity.get('attributes')
                friendly_name = attr.get('friendly_name')
                if friendly_name is None:
                    # in case of blank or misconfigured HA entities
                    continue
                # Add device
                if friendly_name not in devices:
                    devices.append(friendly_name.lower())

    # Make the devices unique
    devices = [*set(devices)]

    for device in devices:
        on = (f'turn on {device}')
        off = (f'turn off {device}')

        wac_add(on, rank=0.5, source='ha_entities')
        wac_add(off, rank=0.5, source='ha_entities')


# WAC Search


def wac_search(command, exact_match=False, distance=SEARCH_DISTANCE, num_results=CORRECT_ATTEMPTS, raw=False, token_match_threshold=TOKEN_MATCH_THRESHOLD, semantic="off", semantic_model=TYPESENSE_SEMANTIC_MODEL, vector_distance_threshold=VECTOR_DISTANCE_THRESHOLD, hybrid_score_threshold=HYBRID_SCORE_THRESHOLD):
    # Set fail by default
    success = False
    wac_command = command

    # Absurd values to always lose if something goes wrong
    tokens_matched = 0
    vector_distance = 10.0
    hybrid_score = 0.0

    # Do not change these unless you know what you are doing
    wac_search_parameters = {
        'q': command,
        'query_by': 'command',
        'sort_by': '_text_match:desc,rank:desc,accuracy:desc',
        'text_match_type': 'max_score',
        'prioritize_token_position': False,
        'drop_tokens_threshold': 1,
        'typo_tokens_threshold': 1,
        'split_join_tokens': 'fallback',
        'num_typos': distance,
        'min_len_1typo': 3,
        'min_len_2typo': 6,
        'per_page': num_results,
        'limit_hits': num_results,
        'prefix': False,
        'use_cache': False,
        'exclude_fields': 'all-MiniLM-L12-v2,gte-small,multilingual-e5-small',
        'search_cutoff_ms': 100,
        'max_candidates': 4,
    }
    if exact_match is True:
        log.info(f"Doing exact match WAC Search")
        wac_search_parameters.update({'filter_by': f'command:={command}'})

    # Support per request semantic or hybrid semantic search
    if semantic == "hybrid":
        log.info(
            f"Doing hybrid semantic WAC Search with model {semantic_model}")
        wac_search_parameters.update(
            {'query_by': f'command,{semantic_model}'})
    elif semantic == "on":
        log.info(
            f"Doing semantic WAC Search with model {semantic_model}")
        wac_search_parameters.update(
            {'query_by': f'{semantic_model}'})

    # Try WAC search
    try:
        log.info(
            f"Doing WAC Search for command '{command}' with distance {distance}")
        wac_search_result = typesense_client.collections[COLLECTION].documents.search(
            wac_search_parameters)
        # For management API
        if raw:
            log.info(f"Returning raw results")
            return wac_search_result

        try:
            id = json_get(wac_search_result, "/hits[0]/document/id")
            text_score = json_get(wac_search_result, "/hits[0]/text_match")
            tokens_matched = json_get(
                wac_search_result, "/hits[0]/text_match_info/tokens_matched")
            wac_command = json_get(
                wac_search_result, "/hits[0]/document/command")
            source = json_get(wac_search_result, "/hits[0]/document/source")
        except:
            log.info(f"Command '{command}' not found")
            return success, command

        if exact_match and wac_command:
            log.info(
                f"Returning exact command '{wac_command}' match with id {id}")
            success = True
            return success, wac_command

        log.info(
            f"Trying scoring evaluation with top match '{wac_command}' with id {id} from source {source}")
        # Semantic handling
        if semantic == "on":
            vector_distance = json_get(
                wac_search_result, "/hits[0]/vector_distance")

            if vector_distance <= vector_distance_threshold:
                log.info(
                    f"WAC Semantic Search passed vector distance threshold {vector_distance_threshold} with result {vector_distance}")
                success = True
            else:
                log.info(
                    f"WAC Semantic Search didn't meet vector distance threshold {vector_distance_threshold} with result {vector_distance}")
        elif semantic == "hybrid":
            hybrid_score = json_get(
                wac_search_result, "/hits[0]/hybrid_search_info/rank_fusion_score")
            if hybrid_score >= hybrid_score_threshold:
                log.info(
                    f"WAC Semantic Hybrid Search passed hybrid score threshold {hybrid_score_threshold} with result {hybrid_score}")
                success = True
            else:
                log.info(
                    f"WAC Semantic Hybrid Search didn't meet hybrid score threshold {hybrid_score_threshold} with result {hybrid_score}")
        # Regular old token match
        else:
            if tokens_matched >= token_match_threshold:
                log.info(
                    f"WAC Search passed token threshold {token_match_threshold} with result {tokens_matched}")
                success = True
            else:
                log.info(
                    f"WAC Search didn't meet threshold {token_match_threshold} with result {tokens_matched}")

    except Exception as e:
        log.exception(f"WAC search for command '{command}' failed with {e}")

    return success, wac_command

# WAC Add


def wac_add(command, rank=0.9, source='autolearn'):
    log.info(f"Doing WAC add for command '{command}'")
    learned = False
    try:
        log.info(f"Searching WAC before adding command '{command}'")
        wac_exact_search_status, wac_command = wac_search(
            command, exact_match=True)
        if wac_exact_search_status is True:
            log.info('Refusing to add duplicate command')
            return learned

        # Get current time as int
        curr_dt = datetime.now()
        timestamp = int(round(curr_dt.timestamp()))
        log.debug(f"Current timestamp: {timestamp}")
        command_json = {
            'command': command,
            'rank': rank,
            'accuracy': 1.0,
            'source': source,
            'timestamp': timestamp,
        }
        # Use create to update in real time
        typesense_client.collections[COLLECTION].documents.create(command_json)
        log.info(f"Added WAC command '{command}'")
        learned = True
    except Exception as e:
        log.exception(f"WAC add for command '{command}' failed with {e}")

    return learned


class HomeAssistantCommand(BaseModel):
    id: int = 0
    initial_id: int = 0
    command: str = None
    distance: Optional[int] = SEARCH_DISTANCE,
    token_match_threshold: Optional[int] = TOKEN_MATCH_THRESHOLD,
    exact_match: Optional[bool] = False,
    semantic: Optional[str] = TYPESENSE_SEMANTIC_MODE,
    vector_distance_threshold: Optional[float] = VECTOR_DISTANCE_THRESHOLD,
    hybrid_score_threshold: Optional[float] = HYBRID_SCORE_THRESHOLD,
    semantic_model: Optional[str] = TYPESENSE_SEMANTIC_MODEL
    learned: bool = False
    speech: str = "Sorry, I can't find that command."
    time_start: datetime = datetime.now()
    done: bool = False


class CommandEndpointConfigException(Exception):
    """Raised when an the command endpoint configuration is invalid

    Attributes:
        msg -- error message
    """

    def __init__(self, msg="Command Endpoint configuration is invalid"):
        self.msg = msg
        super().__init__(self.msg)


class CommandEndpointRuntimeException(Exception):
    """"Raised when an exception occurs while contacting the command endpoint

    Attributes:
        msg -- error message
    """

    def __init__(self, msg="Runtime exception occured in Command Endpoint"):
        self.msg = msg
        super().__init__(self.msg)


class CommandEndpointResult():
    ok: bool = False
    speech: str = ""

    def __init__(self):
        self.ok = False
        self.speech = "Error!"


class CommandEndpoint():
    name = "WAC CommandEndpoint"
    log = logging.getLogger("WAC")


class HomeAssistantWebSocketEndpoint(CommandEndpoint):
    name = "WAS Home Assistant WebSocket Endpoint"

    def __init__(self, app, url, token):
        self.app = app
        self.url = url
        self.token = token

        self.haws = None

        loop = asyncio.get_event_loop()
        self.task = loop.create_task(self.connect())

    async def connect(self):
        while True:
            try:
                # deflate compression is enabled by default, making tcpdump difficult
                async with websockets.connect(f"{self.url}/api/websocket", compression=None) as self.haws:
                    while True:
                        msg = await self.haws.recv()
                        await self.cb_msg(msg)
            except Exception as e:
                self.log.info(f"{self.name}: exception occurred: {e}")
                await asyncio.sleep(1)

    async def cb_msg(self, msg):
        # self.log.info(f"haws_cb: {self.app} {msg}")
        msg = json.loads(msg)
        self.log.debug(f"Msg from HA: {msg}")

        msg_type = json_get(msg, "/type", str)

        try:
            if msg_type == "event":
                id = json_get(msg, "/id")
                ev_type = json_get(msg, "/event/type")
                if ev_type == "intent-end":
                    initial_id = app.session_tracker[id].initial_id
                    # TODO: Update this to use response_type and response_code no_intent_match
                    response_type = json_get(
                        msg, "/event/data/intent_output/response/response_type")
                    if response_type != "error":
                        speech = json_get_default(
                            msg, "/event/data/intent_output/response/speech/plain/speech", "Success")

                        if initial_id != 0:
                            log.debug(f"initial_id: {initial_id}")
                            app.session_tracker[initial_id].speech = speech
                            app.session_tracker[initial_id].done = True
                            return

                        learned = wac_add(
                            app.session_tracker[id].command, rank=0.9, source='autolearn')

                        if learned:
                            speech = f"{speech} and learned command"

                        app.session_tracker[id].speech = speech
                        app.session_tracker[id].done = True

                    else:
                        if initial_id != 0:
                            return
                        # Do WAC Search
                        log.info(
                            "Initial command failed HA intent match - doing WAC search")
                        command = app.session_tracker[id].command
                        distance = app.session_tracker[id].distance
                        exact_match = app.session_tracker[id].exact_match
                        hybrid_score_threshold = app.session_tracker[id].hybrid_score_threshold
                        semantic = app.session_tracker[id].semantic
                        semantic_model = app.session_tracker[id].semantic_model
                        token_match_threshold = app.session_tracker[id].token_match_threshold
                        vector_distance_threshold = app.session_tracker[id].vector_distance_threshold

                        wac_success, wac_command = wac_search(command, exact_match=exact_match, distance=distance, num_results=CORRECT_ATTEMPTS, raw=False,
                                                              token_match_threshold=token_match_threshold, semantic=semantic, semantic_model=semantic_model, vector_distance_threshold=vector_distance_threshold, hybrid_score_threshold=hybrid_score_threshold)

                        if wac_success:
                            log.info(
                                f"Attempting WAC HA Intent Match with command '{wac_command}' from provided command '{command}'")

                            ha_cmd = HomeAssistantCommand(
                                command=wac_command, initial_id=id)

                            newid = self.send({"text": ha_cmd.command})
                            app.session_tracker[newid] = ha_cmd

                        else:
                            app.session_tracker[id].done = True

            elif msg_type == "auth_required":
                auth_msg = {
                    "type": "auth",
                    "access_token": self.token,
                }
                self.log.debug(
                    f"Authenticating HA WebSocket connection: {auth_msg}")
                await self.haws.send(json.dumps(auth_msg))

        except Exception as e:
            log.error(f"Session_tracker: {self.app.session_tracker}")
            log.error(f"Error occurred while parsing HA msg: {e}")
            return

    def parse_response(self, response):
        return None

    def send(self, jsondata):
        id = int(time.time() * 1000)

        if "language" in jsondata:
            jsondata.pop("language")

        out = {
            'end_stage': 'intent',
            'id': id,
            'input': jsondata,
            'start_stage': 'intent',
            'type': 'assist_pipeline/run',
        }

        self.log.debug(f"Sending to HA WS: {out}")
        asyncio.ensure_future(self.haws.send(json.dumps(out)))
        return id

    def stop(self):
        self.log.debug(f"Stopping {self.name}")
        self.task.cancel()


# Request coming from proxy


def api_post_proxy_handler(command, language, distance=SEARCH_DISTANCE, token_match_threshold=TOKEN_MATCH_THRESHOLD, exact_match=False, semantic="off", semantic_model=TYPESENSE_SEMANTIC_MODEL, vector_distance_threshold=VECTOR_DISTANCE_THRESHOLD, hybrid_score_threshold=HYBRID_SCORE_THRESHOLD):

    log.info(
        f"Processing proxy request for command '{command}' with distance {distance} token match threshold {token_match_threshold} exact match {exact_match} semantic {semantic} with vector distance threshold {vector_distance_threshold} and hybrid threshold {hybrid_score_threshold}")
    # Init speech for when all else goes wrong

    try:
        log.info(f"Trying initial HA intent match '{command}'")

        ha_data = {"text": command, "language": language}

        ha_cmd = HomeAssistantCommand(
            command=command,
            distance=distance,
            exact_match=exact_match,
            hybrid_score_threshold=hybrid_score_threshold,
            semantic=semantic,
            semantic_model=semantic_model,
            token_match_threshold=token_match_threshold,
            vector_distance_threshold=vector_distance_threshold,
        )

        id = app.command_endpoint.send(ha_data)
        app.session_tracker[id] = ha_cmd
        return id

    except Exception as e:
        log.exception(f"WAC FAILED with {e}")
        return "Willow auto correct encountered an error!"

    # Do WAC Search
    wac_success, wac_command = wac_search(command, exact_match=exact_match, distance=distance, num_results=CORRECT_ATTEMPTS, raw=False,
                                          token_match_threshold=token_match_threshold, semantic=semantic, semantic_model=semantic_model, vector_distance_threshold=vector_distance_threshold, hybrid_score_threshold=hybrid_score_threshold)

    if wac_success:

        # Re-run HA with WAC Command
        try:
            log.info(
                f"Attempting WAC HA Intent Match with command '{wac_command}' from provided command '{command}'")
            ha_data = {"text": wac_command, "language": language}
            time_start = datetime.now()
            ha_response = requests.post(
                url, headers=ha_headers, json=ha_data, timeout=(1, 10))
            time_end = datetime.now()
            ha_time = time_end - time_start
            second_ha_time_milliseconds = ha_time.total_seconds() * 1000
            log.info('HA took ' + str(second_ha_time_milliseconds) + ' ms')
            ha_response = ha_response.json()
            code = json_get_default(
                ha_response, "/response/data/code", "intent_match")

            if code == "no_intent_match":
                log.info(f"No WAC Command HA Intent Match: '{wac_command}'")
            else:
                log.info(f"WAC Command HA Intent Match: '{wac_command}'")

            # Set speech to HA response - whatever it is at this point
            speech = json_get_default(
                ha_response, "/response/speech/plain/speech", "Success")
            log.info(f"HA speech: '{speech}'")
            speech = f"{speech} with corrected command {wac_command}"
            log.info(f"Setting final speech to '{speech}'")
        except Exception as e:
            log.exception(f"WAC FAILED with {e}")
            return "Willow auto correct encountered an error!"

    if second_ha_time_milliseconds is not None:
        total_ha_time = first_ha_time_milliseconds + second_ha_time_milliseconds
    else:
        total_ha_time = first_ha_time_milliseconds

    log.info(f"Final speech response '{speech}'")
    log.info(f"Total HA time is {total_ha_time} ms")
    return speech


@app.get("/api/add_ha_entities", summary="Add Entities from HA", response_description="Status")
async def api_add_ha_entities():
    try:
        add_ha_entities()
        return JSONResponse(content={'success': True})
    except Exception as e:
        log.exception(f"Add HA Entities failed with {e}")
        raise HTTPException(
            status_code=500, detail="WAC Add HA Entities Failed")


@app.get("/api/re_init", summary="Wipe DB and Start Over", response_description="Status")
async def api_reinitialize():
    try:
        log.info('Re-initializing...')
        typesense_client.collections[COLLECTION].delete()
        init_typesense()
        return JSONResponse(content={'success': True})
    except Exception as e:
        log.exception(f"Re-init failed with {e}")
        raise HTTPException(status_code=500, detail="WAC Re-init Failed")


@app.get("/api/delete", summary="Delete command")
async def api_delete(id: int):
    try:
        log.info(f"Attempting to delete command ID {id}")
        delete = typesense_client.collections[COLLECTION].documents[id].delete(
        )
        command = json_get(delete, "/command")
        log.info(f"Successfully deleted command '{command}' with id {id}")
        response = {'success': True, 'deleted': command}
    except:
        log.info(f"Failed to deleted command with id {id}")
        response = {'success': False}

    return JSONResponse(content=response)


@app.get("/api/search", summary="WAC Search", response_description="WAC Search")
async def api_get_wac(command, distance: Optional[str] = SEARCH_DISTANCE, num_results: Optional[str] = CORRECT_ATTEMPTS, exact_match: Optional[bool] = False, semantic: Optional[str] = TYPESENSE_SEMANTIC_MODE, semantic_model: Optional[str] = TYPESENSE_SEMANTIC_MODEL):
    try:
        time_start = datetime.now()

        # Little fix for compatibility
        if semantic == "true":
            semantic = "on"
        elif semantic == "false":
            semantic = "off"

        results = wac_search(command, exact_match=exact_match,
                             distance=distance, num_results=num_results, raw=True, semantic=semantic, semantic_model=semantic_model)

        time_end = datetime.now()
        search_time = time_end - time_start
        search_time_milliseconds = search_time.total_seconds() * 1000
        log.info('WAC search took ' + str(search_time_milliseconds) + ' ms')
        return JSONResponse(content=results)
    except Exception as e:
        log.exception(f"Search failed with {e}")
        raise HTTPException(status_code=500, detail="WAC Search Failed")


class PostProxyBody(BaseModel):
    text: Optional[str] = "How many lights are on?"
    language: Optional[str] = "en"


@app.post("/api/proxy", summary="Proxy Willow Requests", response_description="WAC Response")
async def api_post_proxy(
    body: PostProxyBody,
    distance: Optional[int] = SEARCH_DISTANCE,
    token_match_threshold: Optional[int] = TOKEN_MATCH_THRESHOLD,
    exact_match: Optional[bool] = False,
    semantic: Optional[str] = TYPESENSE_SEMANTIC_MODE,
    vector_distance_threshold: Optional[float] = VECTOR_DISTANCE_THRESHOLD,
    hybrid_score_threshold: Optional[float] = HYBRID_SCORE_THRESHOLD,
    semantic_model: Optional[str] = TYPESENSE_SEMANTIC_MODEL
):
    try:
        time_start = datetime.now()

        # Little fix for compatibility
        if semantic == "true":
            semantic = "on"
        elif semantic == "false":
            semantic = "off"

        id = api_post_proxy_handler(body.text, body.language, distance=distance, token_match_threshold=token_match_threshold,
                                    exact_match=exact_match, semantic=semantic, semantic_model=semantic_model, vector_distance_threshold=vector_distance_threshold, hybrid_score_threshold=hybrid_score_threshold)

        log.debug(f"Waiting for session with ID {id}")
        # we need to keep the HTTP request open here until WebSocket messages mark the session done
        # TODO: give up after XX seconds
        while app.session_tracker[id].done != True:
            await asyncio.sleep(0.01)

        time_end = datetime.now()
        search_time = time_end - time_start
        search_time_milliseconds = search_time.total_seconds() * 1000
        log.info('WAC proxy total time ' +
                 str(search_time_milliseconds) + ' ms')
        return PlainTextResponse(content=app.session_tracker[id].speech)
    except Exception as e:
        log.exception(f"Proxy failed with {e}")
        raise HTTPException(status_code=500, detail="WAC Proxy Failed")
