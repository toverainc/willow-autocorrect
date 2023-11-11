from decouple import config
from jsonget import json_get
from requests import get
import typesense
import json

HA_URL = config('HA_URL', default="http://homeassistant.local:8123", cast=str)
HA_TOKEN = config('HA_TOKEN', default=None, cast=str)

HA_TOKEN = f'Bearer {HA_TOKEN}'

TYPESENSE_API_KEY = config('TYPESENSE_API_KEY', default='testing', cast=str)
TYPESENSE_HOST = config('TYPESENSE_HOST', default='127.0.0.1', cast=str)
TYPESENSE_PORT = config('TYPESENSE_PORT', default=8108, cast=int)

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

commands_schema = {
  'name': 'commands',
  'fields': [
    {'name': 'command', 'type': 'string*', 'facet': False},
    {'name': 'rank', 'type': 'float' },
    {'name': 'is_alias', 'type': 'bool', 'optional': True },
    {'name': 'alias', 'type': 'string*', 'optional': True, 'facet': False},
    {'name': 'accuracy', 'type': 'float', 'optional': True },
    {'name': 'source', 'type': 'string*', 'optional': True },

  ],
  'default_sorting_field': 'rank',
  "token_separators": [".", "-"]
}

def add_entities():
    print('Adding entities from HA')
    entity_types = ['cover.', 'fan.', 'light.', 'switch.']

    url = f"{HA_URL}/api/states"
    headers = {
        "Authorization": HA_TOKEN,
        "content-type": "application/json",
    }

    response = get(url, headers=headers)
    entities = response.json()

    # For dev and debugging
    json_object = json.dumps(entities, indent=2)
    with open("work/entities.json", "w") as outfile:
        outfile.write(json_object)

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
        print(f"Adding command: '{on}'")
        print(f"Adding command: '{off}'")
        command_on = {
            'command': on,
            'rank': 2.0,
            'source': 'ha_entity',
            }
        command_off = {
            'command': off,
            'rank': 2.0,
            'source': 'ha_entity',
            } 

        ts_client.collections['commands'].documents.create(command_on)
        ts_client.collections['commands'].documents.create(command_off)

# For testing
try:
    ts_client.collections['commands'].delete()
except:
    print('Commands does not exist')

try:
    collections = ts_client.collections.retrieve()
    print(str(collections))
    first_collection = json_get(collections, "[0]/name", str)
except:
    print('No collections found - creating commands schema')
    ts_client.collections.create(commands_schema)
    add_entities()
else:
    print('Already have entities')
