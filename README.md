# flightplot
A tool for generating executive-level plots for drone-related engagement and scenario flight profiles using airdate flight logs.

Usage:
python flightplot.py [flight.csv] [extra_events.json] [margin%]

Context:
 - flight.csv *should* be a CSV generated from AIRDATA UAV (https://airdata.com)

 - extra_events.json is any additional events you'd like to include in the notable events and along the flight tracks. Here is an example:

[
  { "time_ms": 52500,   "event": "Operator engaged payload"       },
  { "time_ms": 102300,  "event": "RSSI dropped (possible jamming)"},
  { "time_ms": 140000,  "event": "Camera refocus test"            },
  { "time_ms": 7000,    "event": "GPS glitch detected"            }
]

- margin% is just a number for scaling the Flight Track plot. 15 is default.

Other smart things happen in the script logic. Please note that if you are using airdata free tier, altitudes will print with MSL. AGL is only for paid tiers and will automatically take priority over MSL.