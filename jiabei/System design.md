# [Design Youtube](https://www.youtube.com/watch?v=IUrQ5_g3XKs&t=331s)

- Small chunks to meet API gateway restriction
- Storage aync update metadata
- Preprocess different resolutions
- Store url to chunks for resolutions in metadata
- Adaptively adjust resolution
- Cache in CDN
- Streaming protocol: 


# [Design Uber](https://www.hellointerview.com/learn/system-design/problem-breakdowns/uber)

<img src=".\img\Uber.png" height="300px">

- How to achieve low latency when a rider requests a ride
    - The bottleneck is fetching all the drivers nearby
        - Regular DB indexed lat and lang won’t work because it can not afford the update location request QPS from drivers. There will be 600K TPS.
        - Trade offs between geo hash and quad tree
            - Quad tree: 
                - Pros: storing global data because there would be geospatial data: like data points are not stored evenly
                - Cons: updating data takes time as tree need to be reshuffled
            - Geohash:
                - Pro: update is cheap
                - Cons: unnecessary storage of some bits
    - Use geohash in redis

- How to achieve consistency: 1 driver should be only matched to 1 rider
    - Drawbacks of keeping a state in the DB: there’s no chance to change the state from in-request back to available
    - We can use a distributed lock in redis
- How to further reduce TPS
    - Reduce the frequency of update location.

# Design a global secondary index
"I would store separate noSQL DBs for each core entity: Customer, Restaurant, Dasher. the PK for each entity is their unique ID; SK is order_id. When an order reuqest or dasher makes update request through API Gateway, the event is added into the MQ. Each NoSQL entity is handled by a separate service. These services consumes the messages from the main MQ and insert/update the event in the DB. The alternative considered are relational DB and search index. Relational DB is good for fast view stats, as most of them supports SQL like GROUP_BY, which supports low latency read queries. However, it is not ideal to support high volumn of write requests. Search index is able support heavy writes as the writes can be modeled as WAL, and that support large load. However, it does not meet the consistency requirement., because building search index takes some time, and the lag is usually longer than 15 mins. The benefit of multiple entites is that the detail information of order can be stored only once in Order table, and when supporting queries, it could be done by 2 separate fetches: i.e. query orders by customer id, first query customer entity to fetch order_ids filtered by customer id and in time order. Then use the order ids to fetch from order entity. This way duplicated data is minimized."

- “You're using an MQ to decouple writes from processing. How would you guarantee consistency and ordering of updates — say, if two updates for the same order come close together?”
    - Use a FIFO MQ partitioned by order_id. 
    - Use state machine and idempotency for ordering
    - retry + DLQ strategy

- “Say one of the services (e.g., Customer service) crashes and loses its place in the MQ. How do you ensure its DB can be recovered and consistent with the Order DB?”
    - snapshot + replay + offset tracking

- “What’s your approach if we need to fetch all orders for a restaurant that gets tens of thousands of orders per day — do you still query via SK on the Restaurant DB?”
    - shard by restaurant ID, resharding keep hot restaurant balanced
    - When reading, use pagination by time

- "One requirement was filtering by time range and order status. Can you efficiently support this on NoSQL using your current SK design?"

- "Each event results in writes to 3 different services (Customer, Dasher, Restaurant). How do you manage consistency and durability in the face of partial failures?"


## improved design
- Order service receives the requests, and update to order DB.
- The change to order DB also emitts an event to a MQ. The subscribers use this event to async write to other tables


# Design Twitter

- How would you design the system to handle high fan-out scenarios—such as when a celebrity with 100 million followers posts a new update? Would you use fan-out on write or fan-out on read, and why?
    - Hybrid aproach

- Suppose you choose to precompute user feeds. What strategies would you use to keep the precomputed feeds fresh and consistent with new content or changes (like deleted posts or changes in relevance)? How would you balance storage cost vs latency?
    - Keep the feed for active users in cache
    - feed generation runs on a write trigger.

- Precompute prioritization: how would you prioritize whose feeds to precompute and when?
    - Define tier thresholds. Precompute differently for different tier of users

- Partial Fan-out + Merge strategy: How would you implement the hibrid aproach? Prevent duplicate posts and stale views
    - Use post ID with versioning and conflict resolution policies.
    - Define a merge policy

- Cold start strategy: Inactive user loged in the system how would the feed generated?
    - Show personalized trending limit to recent 1-2 days, generate the feed during this time.

- Feed Mutation Tracking: How would you track which users' feed caches include a specific post so that the post update can be propagated?

- Backpressure Control on fan out: during an unexpected fan-out storm, how would you system protect itself while still delivering decent UX?
    - rate limiting, dynamic shedding.

# Design Ticket master

# Design contending updates
## problem
Context: DoorDash's system tracks the real-time status of millions of food delivery orders. Each order has a status that updates frequently and from different sources:

Drivers updating pickup, en route, delivery times via mobile app.

Merchants updating food prep status (e.g., “preparing,” “ready for pickup”).

System services predicting status using GPS and ETA algorithms.

Customer support agents overriding or correcting statuses.

Problem: Design a highly available, consistent, and performant system to store and serve the current status of an order.

## high level design
Our core components are Driver client, Merchant client, Agent client: These clients send the request through API gateway to talk to backend service; API Gateway: for authentication, rate limiting, load balancing; Order service to handle order update and status check requests; WAL: Write ahead log to store the changes to the orders as events, the logs should be stored in flash for durability; An offline archiving service periodically read from WAL to find the completed orders to write to the archive storage which might be on disk; MQ between Order service and WAL to improve latency; Estimation service subscribe to MQ to generate Estimation to write to WAL. For reading order status, we can add a cache layer between order service and WAL to improve latency. When reading from WAL, Order service is responsible for resolving and determining the final status from reading events in WAL, order service then write this status in cache and record the offset in WAL.
<img src=".\img\ContendingUpdates.png" height="500px">

## deep dive questions
1. How would you design the logic for resolving contending updates from different sources (e.g., driver vs merchant vs estimation service) in Order Service? How do you prevent stale or out-of-order updates from regressing the status?

- Order service first read all the events of an order from WAL, filter out only legitimate events. The service maintains a state machine inside and based on the status transition rules to determine the final state. i.e. a in-preparing state is not allowed from delivered state. Also Order service determines whether or not to show ETA of a stage to client.

**Expected answer**:
> State Machine, logical timestamp, source precedence, deduping

2. The cache contains a resolved status and WAL offset. If the cache entry is evicted or stale, and the WAL is large, how do you efficiently reconstruct the latest order status without reading the full log for that order?
- ?

**Expected answer**:
> Store a checkpointed snapshot in a separate KV store. 
>
> On cache miss
> - read snapshot
> - resume from WAL
> TTL for snapshots

3. Let’s say a driver, merchant, and estimation service all send updates for the same order in parallel. How does your system ensure consistency in the final status while maintaining low latency under high concurrency? Would you lock, use timestamps, version vectors, or something else?
- First of all, we use a MQ to decouple order service from the storage improve low latency. Additionally, for WAL, we use a quorum algorithm to balance consistency and low latency. Then for the events, we'll use version vectors to guarantee ordering.

**Expected answer**:
> - MQ helps decoupling
> - version vectors
> - per-source logical clocks
> - WAL quorum

4. Suppose you have a regional outage and your Order Service needs to failover to another region. How do you ensure WAL replication is consistent and you don’t replay updates out of order or lose updates?
- For WAL, we have quorum algorithm to sync the updates periodically

**Expected answer**:
> - cross-region replicated WAl
> - ensure quorum replication
> - metadata about offset
> - idempotent writes using event ID or tombstones

5. A staff-level concern: Some updates (e.g., “Delivered”) are critical milestones that trigger billing, notifications, and dashboards. How would you implement an exactly-once mechanism for processing these milestone updates without breaking under retries, duplication, or failover scenarios?
- We need an idempotency solution: for example 2-phase commit after reading this event and set a status flag of this event.

**Expected answer**:
> - idempotency key for events
> - durable write of processing status
> - 2PC or saga
> - deduplication queue
> - outbox pattern

outbox pattern is a super useful tool for ensuring idempotency and exactly-once semantics. a local outbox table in the same transaction as the state change. a background worker reads from the outbox and publishes to kafka