# Public training event logs

The following real-world logs supplement the generated foundation-training
corpus. Each contains fewer than 100,000 events and represents a different
organization, process, and time period from both the other additions and the
logs reserved in `../logs_eval/`.

| Local file | Domain | Cases | Events | Activities | Source and citation | Dataset terms |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `activities_of_daily_living.xes.gz` | Sensor-derived household routines | 148 | 11,138 | 32 | Timo Sztyler and Josep Carmona (2015), [Activities of daily living of several individuals](https://doi.org/10.4121/uuid:01eaba9f-d3ed-4e04-9945-b8b302764176) | [4TU General Terms of Use](https://doi.org/10.4121/resource:terms_of_use) |
| `bpic2013_incidents.xes.gz` | Volvo IT incident management | 7,554 | 65,533 | 4 `concept:name` values (with lifecycle detail) | Ward Steeman (2013), [BPI Challenge 2013, incidents](https://doi.org/10.4121/uuid:500573e6-accc-4b0c-9576-aa5468b10cee) | [4TU General Terms of Use](https://doi.org/10.4121/resource:terms_of_use) |
| `bpic2015_municipality_1.xes.gz` | Dutch municipal building permits | 1,199 | 52,217 | 398 | Boudewijn van Dongen (2015), [BPI Challenge 2015, Municipality 1](https://doi.org/10.4121/uuid:a0addfda-2044-4541-a450-fdcc9fe16d17) | [4TU General Terms of Use](https://doi.org/10.4121/resource:terms_of_use) |
| `bpic2020_request_for_payment.xes.gz` | TU/e non-travel payment requests | 6,886 | 36,796 | 19 | Boudewijn van Dongen (2020), [BPI Challenge 2020: Request For Payment](https://doi.org/10.4121/uuid:895b26fb-6f25-46eb-9e48-0dca26fcd030) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| `production.xes.gz` | Industrial production | 225 | 4,543 | 55 | Dafna Levy (2014), [Production Analysis with Process Mining Technology](https://doi.org/10.4121/uuid:68726926-5ac5-4fab-b873-ee76ea412399) | [4TU General Terms of Use](https://doi.org/10.4121/resource:terms_of_use) |

Dataset-specific terms govern these data files; the repository's software
license does not relicense them.

## Integrity and preparation

The BPI 2013 and BPI 2020 files are byte-for-byte copies of their repository
artifacts. BPI 2015 is the same upstream XML payload compressed with a
deterministic gzip header.

| Local file | Upstream artifact | Upstream MD5 | Local SHA-256 |
| --- | --- | --- | --- |
| `activities_of_daily_living.xes.gz` | derived from the eight XES members of `data.zip` | archive: `0c362d657d0a2dd829bfe0d3bedac223` | `50d3cd76ad2cccb6bf8f0a7760a04264aea7c02d870ab850899a10dfc77c9ac8` |
| `bpic2013_incidents.xes.gz` | `BPI_Challenge_2013_incidents.xes.gz` | `d4809bd55e3e1c15b017ab4e58228297` | `6ec2136c607648608655802486df97a866f351668751b7ecbca51e1f8715d522` |
| `bpic2015_municipality_1.xes.gz` | `BPIC15_1.xes` | `6c8a04d566eef571cf0aaf7b6f70310b` | `941fad17a9adca05808b3d98264fa6f183e91c24107c64a8c12dd959c012b954` |
| `bpic2020_request_for_payment.xes.gz` | `RequestForPayment.xes.gz` | `2eb4dd20e70b8de4e32cc3c239bde7f2` | `028e2d645153190200b2ea998747efa02cf1beb6f83db62598963cf391f114b1` |
| `production.xes.gz` | derived from `Production_Data.csv` in `data.zip` | archive: `46174bf470258f9895dd153c78bc8b61` | `be5956a57c8106db5e1151033e63ee4836e6d3b4f794b39e7674d0f3ac057d5c` |

The daily-living dataset publishes eight small XES logs covering weekday and
weekend behavior for several individuals. The local artifact combines all 148
traces into one training pool, preserves the original events and attributes,
and prefixes each case identifier with its source member name. This prevents
case-ID collisions while avoiding eight tiny pools receiving disproportionate
weight from per-log sampling. The source archive SHA-256 is
`9d25fe4f90ae4d9949928f4e9420b823ea3c401c3b95746bdd56e4e4ed44b6c4`.

The production repository publishes tabular process data rather than XES. The
local XES conversion preserves all 4,543 source rows, including repeated rows,
and performs the following deterministic mapping:

- `Case ID` becomes the trace `concept:name`.
- Each row becomes one event at `Complete Timestamp`; `Activity` and `Resource`
  become `concept:name` and `org:resource`.
- `Start Timestamp`, part, worker, report type, work-order quantity, completion,
  rejection, MRB, and rework fields are retained as event attributes.
- Events are stably ordered by completion time and original source-row order.
- The source has timezone-naive timestamps. They are serialized with UTC as a
  neutral XES timezone without changing wall-clock values or within-case time
  differences.
- `duration:minutes` is recomputed from the start and completion timestamps
  because one source `Span` value uses an inconsistent spreadsheet date format.

The source archive SHA-256 is
`b3a43f457393bc9dd0c616db189d01aa55e823f202cc8ce0cd537572840644cf`;
the extracted `Production_Data.csv` SHA-256 is
`ef2477450926aa650e83a038cc81578a048a31ee22dfb858b5eab2a7883a21e6`.

## Selection and separation rationale

These logs add real escalation/rework, approval, manufacturing, resource,
calendar, long-running public-administration, and sensor-derived human-routine
patterns that are absent from a corpus dominated by generated business flows.
They are source-disjoint: household sensors, Dutch municipal permits, Volvo IT
incidents, TU/e payment requests, and industrial production come from
independent datasets. None is a derivative, subset, or alternate serialization
of Billing, Helpdesk, Receipt, Road Traffic Fines, or Sepsis in `../logs_eval/`.

For the two additions, an ingestion audit compared all 63,355 events and 1,347
traces with the 993 XES files already in `logs/`, `logs/out/`, and
`logs_eval/`. It found zero matching exact event signatures (case ID, activity,
timestamp, lifecycle, and resource) and zero matching case-independent trace
signatures (ordered activity, timestamp, and lifecycle). The additions also
have zero such matches with one another. Both stay below the 100,000-event
limit, contain `concept:name` and `time:timestamp` on every event, and are
explicit members of the content-addressed source corpus in `config.py`.
