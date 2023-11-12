# Willow Auto Correct (WAC) - EXTREME PREVIEW
One step closer to "better than Alexa".

## Introduction
Voice assistants make use of speech to text (STT/ASR) implementations like Whisper.
While they work well most command endpoint platforms (like Home Assistant) have pretty tight matching 
of STT output to "intent", where "intent" is matching an action and device - whatever the transcript/command is supposed to do.

This can lead to usability issues. A simple transcription error like "turn of" instead of "turn off" does nothing.

Voice assistants are supposed to be convenient, fast, and easy. If you have to repeat yourself why bother?

## Introducing Willow Auto Correct
Willow Auto Correct smoothes out these STT errors by leveraging [Typesense](https://typesense.org/) to learn and fix them.

Typesense as used by Willow Auto Correct combined with Willow, Willow Application Server, and Willow Inference Server
is a gigantic leap forward for voice assistant usability in the real world.

That said this is a very, very early technology preview. Caveat emptor!

## Getting Started

Clone this repo.

Add your Home Assistant base URL and long-lived access token to `.env`:
```
cat .env
HA_TOKEN="shhh_your_token"
HA_URL="http://homeassistant.local:8123"
```

Start things up:

```
./utils.sh build-docker
./utils.sh ts # Starts typesense as daemon
./utils.sh run # Starts WAC in foreground
```

Then configure WAS via the web interface with the REST command endpoint (no auth):

`http://your_machine_ip:9000/proxy`

Save and Apply changes.

This will insert WAC in between your Willow devices, WAS, and Home Assistant.

DOUBLE CHECK: Make sure you have "WAS Command Endpoint (EXPERIMENTAL)" enabled under "Advanced Settings"!!!

While you're being brave why don't you try WOW (Willow One Wake) and play around with notifications?

### Learning Flow (Autolearn)

Initially all WAC does is replace "Sorry, I didn't understand that" with "Sorry, I don't know that command".

This lets you know you're using it.

At first, commands are passed-through to HA. When HA responds that the intent was matched the following happens:

1) WAC searches typesense to make sure we don't already know about that successful command. This uses exact string search.
2) If the matching intent command is new, add to typesense.
3) The command does what it does.

If the intent isn't matched and WAC doesn't have a prior successful intent match we don't do anything other than return "Sorry, I don't know that command".
This is what you have today.

### Operational Flow

Once WAC starts learning successfully matched commands things get interesting.

### Fixing basic stuff

Learned commands will make full use of typesense distance (fuzzy) matching.
Distance matching corrects things like variations in the transcript - characters and words being moved around, etc. Examples:

- "Turn-on" matches "turn on"
- "Turn-of" matches "turn off"

Our typesense schema specifically includes the default of spaces plus '.' and '-'. We can alter this if need be.

Overall this functionality can be configured with the Levenshtein distance matching API param `distance` which we support providing dynamically.

There are also a variety of additional knobs to tune: look around line 110 in `wac.py` if you are interested - and that's just a start!

We intend to incorporate early feedback to expose configuration parameters and improve defaults for when WAC is integrated with WAS.

### Figuring out what you're actually trying to do

Typesense uses "semantic search".

Symantic search can recognize variations in language - it understands what you "mean". So for example:

- "Turn on the lights in eating room" matches "turn on dining room".
- "turn on upstairs desk lamps" matches "turn on upstairs desk lights"

Between distance matching and symantic search WAC can match some truly wild variations in commands/STT errors:

- "turn-of lights and eating room." becomes 'turn on dining room lights.'
- "turn-on lights in primary toilet" becomes "turn on lights in master bathroom"

It's also very good about completely ignoring random speech inserted from the transcript.
It does not care at all - it only matches on tokens from each of the provided words and ignores the rest.

All of this is case-insensitive.

### Fun Configuration
You can define `TOKEN_MATCH_THRESHOLD` in `.env` with an integer.
The default is 3 which is pretty much middle of the road.

4 is pretty tight but still useful.
2 is aggressive but can get sloppy.
1 will almost always match "something" depending on how many Autolearn commands you have. Probably a bad idea - just try to feed it good commands at first.
Any larger numbers are meant for longer text strings typically not seen in voice commands.

One idea for the Autolearn/training phase is to teach WAC the commands you intend to use while speaking clearly and close to the device.
This will populate the typsense index with the commands you actually use - enabling the full power of WAC while cutting down on mistakes by not including things you don't intend to do.

### This thing is all over the place...

Sometimes "smart" is too smart and then dumb. WAC has an interface at `http://your_machine_ip:9000/docs` where you can run a search with various parameters.
The output provided is the raw result from typesense and very verbose.

Search for "Turn-on eating room":

```
{
  "facet_counts": [],
  "found": 6,
  "hits": [
    {
      "document": {
        "command": "turn on dining room.",
        "id": "5",
        "rank": 1,
        "source": "autolearn"
      },
      "highlight": {
        "command": {
          "matched_tokens": [
            "turn",
            "on",
            "room"
          ],
          "snippet": "<mark>turn</mark> <mark>on</mark> dining <mark>room</mark>."
        }
      },
      "highlights": [
        {
          "field": "command",
          "matched_tokens": [
            "turn",
            "on",
            "room"
          ],
          "snippet": "<mark>turn</mark> <mark>on</mark> dining <mark>room</mark>."
        }
      ],
      "text_match": 1736172750663319600,
      "text_match_info": {
        "best_field_score": "3315670777856",
        "best_field_weight": 15,
        "fields_matched": 1,
        "score": "1736172750663319673",
        "tokens_matched": 3
      }
    }
...etc
```

The important thing to look for is the `text_match_info/tokens_matched` field, which is what we use for the `TOKEN_MATCH_THRESHOLD` above.
This can give you an idea of how to tune this thing for whatver your actual experience is.

### Resource Utilization and Performance
Resource utilization is very minimal. It's a complete non-issue unless you have tons of commands and even then probably not a big deal.

In my testing the entire docker container uses ~60mb of RAM and a few percent CPU (will vary on system, but fine even for Raspberry Pi).

Latency of typesense itself is typically in single digit milliseconds. It's all of the other stuff (WAC logic, HA, etc) that can result in ~100ms latency.
See Performance below.

## The Future

### Full integration with WAS
Included in WAS, "just works".

### Performance
I'm too lazy to deal with HA websockets so we open a new REST connection every time (at least twice).
This is "slow". When WAC is intergrated with WAS we will use websockets if available - just like WAS does today.

Typesense tuning. One example: for instant responsiveness of learned commands we don't use the aggressive memory cache. We might want to.

### Rank
Our configured matching criteria includes the stuff above plus a user defined rank.
This is a float value that can be attached to a command to heavily weight matching priority in addition to the fuzzy distance matching and symantic search.

This will be integrated in the WAS Web UI.

### Aliases
Our typesense schema includes the concept of "aliases".
This lets you basically say "do all of your fancy stuff with whatever I add to the admin interface AND do your fancy stuff again to match a command you learned".

### Accuracy
Our schema also has the concept of "accuracy".
For learned commands users "thumbs up/thumbs down/re-arrange" matches and we can use this to influence the match weighting as well.

### Getting Aggresive
We currently only grab the first result from Typesense and retry HA once with it. We might want to tweak this.

### More Match Configuration
See that typesense output above? We can use those large scores, etc to do additional ranking.

### LLM Integration
We have internal testing with various LLMs. Typesense and Langchain [can be integrated](https://python.langchain.com/docs/integrations/vectorstores/typesense?ref=typesense) so this will get really interesting.

### Vector search accelerated with WIS
This is actually all pretty simple in the grand scheme of things.
We can include WIS accelerated text embedding models and vector search in typesense to expand this functionality.

## Why is this a big deal?
1) Repeating yourself is the worst.
2) Mumble from further away, in worse conditions.
3) Likely get away with using a lower resource-utilization Whisper model (even though WIS is really fast). Even on CPU!
