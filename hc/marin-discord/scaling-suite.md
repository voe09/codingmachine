# #scaling-suite

Archive coverage: 2025-05-22–2026-06-10 (UTC); 187 messages, including 123 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1375005693899309126).

## Discussion and lessons

- The suite checked whether small runs reliably forecast large-run loss and exposed training instabilities. One 1e21 run ended within 0.0002 loss of prediction, while larger-run spikes prompted examination of specific data batches. [Source](https://marin-discord.pages.dev/#1375005693899309126/1479166845788356760).
- Forecast uncertainty was discussed explicitly: an error of 0.02 loss over two orders of magnitude was considered small relative to the bootstrap interval, not a reason by itself to abandon a run. [Source](https://marin-discord.pages.dev/#1375005693899309126/1481304884547551512).
- The suite pre-registered Paloma macro-loss forecasts before the larger run, preventing later tuning of the prediction to the outcome. The 1e21 run finished just 0.0002 loss from its forecast. [Discussion](https://marin-discord.pages.dev/#1375005693899309126/1478447834943586428).
- Loss spikes in the 1e22 run were traced to particular malformed/repetitive samples, including a digit wall and a repeated citation error. Inspecting exact batches is more informative than assuming optimizer instability from the aggregate curve. [Discussion](https://marin-discord.pages.dev/#1375005693899309126/1479206554124292146).
- At 1e23 scale, a fixed loss asymptote predicted pessimistically and a learned asymptote optimistically; the observed cooldown trend fell between them. For resource planning, a conservative forecast can be more useful than minimum point error. [Discussion](https://marin-discord.pages.dev/#1375005693899309126/1487556665875107870).
