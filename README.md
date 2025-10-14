# whale-tracker

#Goal. Detect statistically unlikely clusters of large trades (“whales”) in Kalshi markets and surface them in a real-time dashboard for further review.
#Whale (single trade). A trade with size (or notional) at or above the 99th percentile of the last 7 days for that market (with a small absolute notional floor).
#Whale cluster. ≥ 3 whale trades within a 120-second rolling window.
#Null model. Whale arrivals follow a random baseline rate λ estimated from recent history.
#Alert. Raise an alert when the current window simultaneously has (i) Poisson tail p < 0.01 for ≥3 whales given λ, and (ii) z-score > 3 versus recent windows.
#Output. Live panel per market with latest whale prints, highlighted cluster windows, p-values, and z-scores; historical log for auditing.
