# Willow Auto Correct (WAC) - PREVIEW
One step closer to "better than Alexa".

## Introduction
Voice assistants make use of speech to text (STT/ASR) implementations like Whisper.
While they work well, most command endpoint platforms (like Home Assistant) have pretty tight matching 
of STT output to "intent". Intent is matching a voice command to an action.

A simple speech recognition error like "turn of" instead of "turn off" doesn't work. Frustrating.

Voice assistants are supposed to be convenient, fast, and easy. If you have to repeat yourself why bother?

We can use this "tight" matching to our advantage by learning your commands when the command platform "tight" match is successful.

That's when Willow Auto Correct goes to work!

Seem to good to be true? Read on.

## Introducing Willow Auto Correct
Willow Auto Correct smooths out these STT errors and more by leveraging [Typesense](https://typesense.org/) to learn and fix them.

Typesense as used by Willow Auto Correct combined with Willow, Willow Application Server (WAS), and Willow Inference Server (WIS)
is a leap forward for open-source, local, and private voice assistant usability in the real world.

That said this is a very, very early technology preview. Caveat emptor!

## Why is this a big deal?
1) Repeating yourself and remembering how to talk to your voice "assistant" is the worst.
2) You can likely get away with using lower resource utilization Whisper model. WIS is already really fast and accurate - WIS + WAC is even faster while being much more accurate. Especially on CPU!
3) Speak the way you do. We hesitate, mumble, and say what we mean with variety. Other people understand us. Voice assistants should too.

## Getting Started

Clone this repo.

Add your Home Assistant base URL and long-lived access token to `.env`:
```
HA_TOKEN="shhh_your_token"
HA_URL="http://your_home_assistant_host:8123"
```

You can get these from WAS or HA.

Start things up:

```
./utils.sh build-docker # Builds WAC
./utils.sh ts # Starts typesense as daemon
./utils.sh run # Starts WAC in foreground
```

Then configure WAS via the web interface with the REST command endpoint (no auth):

`http://your_wac_machine_ip:9000/api/proxy`

Save and Apply changes.

This will insert WAC in between your Willow devices, WAS, and Home Assistant.

DOUBLE CHECK: Make sure you have "WAS Command Endpoint (EXPERIMENTAL)" enabled under "Advanced Settings"!!!

While you're being brave why don't you try Willow One Wake (WOW) and play around with notifications?

### Learning Flow (Autolearn)

Initially all WAC does is replace "Sorry, I didn't understand that" with "Sorry, I can't find that command".

This lets you know you're using it.

Commands are pass-through to HA. When HA responds that the intent was matched the following happens:

1) WAC searches Typesense to make sure we don't already know about that successful command. This uses exact string search.
2) If the matching intent is new add it to Typesense.
3) The command does what it does in Home Assistant.
4) We give you the Home Assistant output and let you know we learned a new command.

If the intent isn't matched and WAC doesn't have a prior successful intent match we don't do anything other than return "Sorry, I can't find that command".
This is what you have today.

Autolearn is a lifelong learner - even as you add entities, change their names, speak differently, etc.

### Operational Flow

Once WAC starts learning successfully matched commands things get interesting.

### Fixing basic stuff

Learned commands make full use of Typesense distance (fuzzy) matching.
Fuzzy matching corrects things like variations in the transcript - minor variations in strings. Examples:

- "Turn-on" matches "turn on"
- "Turn of" matches "turn off"

Our Typesense schema explicitly includes the default of spaces plus '.' and '-'. We can alter this if need be.

Overall this functionality can be configured with the Levenshtein distance matching API param `distance`, which we support providing dynamically. What is Levenshtein distance? It's a 70 year old way to figure out how many times you need to move letters around to make two sentences match. If it ain't broke don't fix it!

There are also a variety of additional knobs to tune: look around line 110 in `wac.py` if you are interested - and that's just a start! Typesense is really on another level.

We intend to incorporate early feedback to expose configuration parameters and improve defaults for when WAC is integrated with WAS. All configuration options in your browser with WAS and WAS Web UI.

### Figuring out what you're actually trying to do

Typesense includes "semantic search".

Semantic search can recognize variations in language - it understands what you're trying to say. For example:

- "Turn on the lights in eating room" matches "turn on dining room".
- "turn on upstairs desk lamps" matches "turn on upstairs desk lights"

Between distance matching and semantic search Typesense can match some truly wild variations in commands:

- "turn-of lights and eating room." becomes 'turn off dining room lights.'
- "turn-on lights in primary toilet" becomes "turn on lights in master bathroom"

As you can see both of these examples have multiple speech recognition errors and vocabulary/grammar variance.

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

One idea to seed Autolearn/training phase is to teach WAC the commands you intend to use while speaking clearly and close to the device.
This will populate the Typesense index with the commands you actually use - enabling the full power of WAC while cutting down on mistakes by not including things you don't intend to do.

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
This can give you an idea of how to tune this thing for whatever your actual experience is.

### Resource Utilization and Performance
Resource utilization is very minimal. It's a complete non-issue unless you have tons of commands and even then probably not a big deal.

In my testing the entire docker container uses ~60mb of RAM and a few percent CPU (will vary on system, but fine even for Raspberry Pi).

Semantic search uses more memory with the Auto Correct container using ~300mb of RAM. This is due to the language model used by semantic search.

Latency of typesense itself is typically in single digit milliseconds. It's all of the other stuff (WAC logic, HA, etc) can result in ~200ms latency. See Performance below.

## The Future

### Full integration with WAS
Included in WAS, "just works". Command addition, deletion, ranking, ordering, searching, thresholds, alias managment, etc all in the WAS Web UI.

### Performance
I'm too lazy to deal with HA websockets so we open a new REST connection every time (at least twice).
This is "slow". When WAC is integrated with WAS we will use websockets if available - just like WAS does today.

Typesense tuning. One example: for instant responsiveness of learned commands we don't use the aggressive memory cache. We might want to.

### Rank
Our configured matching criteria includes the stuff above plus a user defined rank.
This is a float value that can be attached to a command to heavily weight matching priority in addition to the fuzzy distance matching and semantic search.

This will be integrated in the WAS Web UI.

### Aliases
Our typesense schema includes the concept of "aliases".
This lets you basically say "do all of your fancy stuff with whatever I add to the admin interface AND do your fancy stuff again to match a command you learned".

### Accuracy
Our schema also has the concept of "accuracy".
For learned commands users thumbs up/thumbs down/re-arrange matches and we can use this to influence the match weighting as well.

### Getting Aggressive
We currently only grab the first result from Typesense and retry HA once with it. We might want to tweak this.

### More Match Configuration
See that Typesense output above? We can use those large scores, etc to do additional ranking. Again - Typesense is on another level. I feel like I'm selling it at this point.

### LLM Integration
We have internal testing with various LLMs. Typesense and Langchain [can be integrated](https://python.langchain.com/docs/integrations/vectorstores/typesense?ref=typesense) so this will get really interesting.

### Vector search accelerated with WIS
Up to this point this all has been pretty simple in the grand scheme of things.
We can leverage WIS accelerated text embedding models and vector search in Typesense and WAC.

### Multiple languages
Not a problem, just need to get around to it.
