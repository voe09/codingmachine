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
