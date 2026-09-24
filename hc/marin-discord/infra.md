# #infra

Archive coverage: 2025-04-30–2026-08-25 (UTC); 7,369 messages, including 2,445 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1364827114670657616).

## Discussion and lessons

- Infrastructure discussion spanned Ray and Iris scheduling, TPU preemption, cluster/controller incidents, storage costs, observability, and migration of jobs from Ray commands to Iris. The high message volume includes many transient operations reports. [Source](https://marin-discord.pages.dev/#1364827114670657616/1476412061339619469).
- Checkpoint storage was estimated at roughly $60k per month. Retention proposals prompted concern that one-month deletion could remove baselines needed for three- to six-month research projects; warning and recovery policy were still under discussion. [Cost estimate](https://marin-discord.pages.dev/#1364827114670657616/1490767485547319388); [research concern](https://marin-discord.pages.dev/#1364827114670657616/1490771304188481657).
- The recurring reliability agenda included image retention, independent job-crash alerting, preemption cleanup, worker reattachment, and consistent Parquet storage. These were operational prerequisites for interpreting long-run experiments, not just maintenance chores. [Discussion](https://marin-discord.pages.dev/#1364827114670657616/1409642552960880780).
- The January 2026 Ray token-auth migration required a cluster reboot and checkpoint/resubmission planning. By February, Iris had region-aware scheduling and an `iris job run` path, setting up the move away from Ray-specific launch commands. [Discussion](https://marin-discord.pages.dev/#1364827114670657616/1456402718481711337).
- An Evalchemy integration was reported about 10x faster after moving evaluation in-process and fixing bugs; orphaned subprocesses had been holding TPU devices after a parent process died. This is an example of performance and correctness sharing one root cause. [Discussion](https://marin-discord.pages.dev/#1364827114670657616/1471664293479972886).
