# Release-note draft (not published)

Runs by status: crashed 1, measured 23.

- Observed successful output throughput of lq_final 0.6% lower than bf16_window on local_diagnostics:decode_128_512_c512 (ratio 0.994, 95% interval 0.870 to 1.136, 5 paired blocks); the interval includes no change (95% t interval on log ratios, 5 blocks; per-block ratios 1.055, 1.023, 1.048, 0.821, 1.046; experimental: quality not yet qualified (campaign quality_qualification pending); H100 port evaluated with paper-inspired methodology; not a paper reproduction; screening results are not confirmation)
