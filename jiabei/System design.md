# Design Youtube

https://www.youtube.com/watch?v=IUrQ5_g3XKs&t=331s

- Small chunks to meet API gateway restriction
- Storage aync update metadata
- Preprocess different resolutions
- Store url to chunks for resolutions in metadata
- Adaptively adjust resolution
- Cache in CDN
- Streaming protocol: 


# Design Uber
https://www.hellointerview.com/learn/system-design/problem-breakdowns/uber

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
    - 

- "Each event results in writes to 3 different services (Customer, Dasher, Restaurant). How do you manage consistency and durability in the face of partial failures?"


## improved design
- Order service receives the requests, and update to order DB.
- The change to order DB also emitts an event to a MQ. The subscribers use this event to async write to other tables