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

# This doesn't seem to be getting from docker to here - FIX
TYPESENSE_API_KEY = config('TYPESENSE_API_KEY', default='testing', cast=str)
TYPESENSE_HOST = config('TYPESENSE_HOST', default='127.0.0.1', cast=str)
TYPESENSE_PORT = config('TYPESENSE_PORT', default=8108, cast=int)

HA_URL = f'{HA_URL}/api/conversation/process'
HA_TOKEN = f'Bearer {HA_TOKEN}'

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

ha_headers = {
    "Authorization": HA_TOKEN,
}

# Request coming from proxy
def api_post_proxy_handler(text, language):

    # Init speech for when all else goes wrong
    speech = "Sorry, I don't know that command."

    data = {"text": text, "language": language}
    try:
        ha_response = requests.post(HA_URL, headers=ha_headers, json=data)
        ha_response = ha_response.json()
        code = json_get_default(ha_response, "/response/data/code", "intent_match")

        if code == "no_intent_match":
            log.info('No HA Intent Match')
        else:
            log.info('HA Intent Match')

        # Set speech to HA response
        log.info('Setting speech to HA response')
        speech = json_get(ha_response, "/response/speech/plain/speech", str)

    except:
        speech = "HA Failed"

    return speech

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/proxy")
async def api_post_proxy(request: Request):
    request_json = await request.json()
    language = request_json['language']
    text = request_json['text']
    response = api_post_proxy_handler(text, language)

    return PlainTextResponse(content=response)