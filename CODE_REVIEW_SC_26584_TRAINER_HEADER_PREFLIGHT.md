# sc-26584 late trainer header-preflight repair

Reviewed source: `d327a28f4c78141c75cd02527cf2699e18d63619` (merged inference main).
This narrow repair follows the first real campaign header refusal; no campaign
cell has been accepted. It is a late engine fix requiring a new reviewed main PR
and a permanent consumer pin only after that PR merges.

#### [F-288] P1: MLX trainer omits the ordinary backbone from header admission

`MossImageTrainer::load` used `Sources::prepare`, whose `None` selection is the
generation factory's intentional shared-only preparation. It excludes all five
ordinary transformer paths and uses a default transformer config. Consequently
the trainer's strict `FootprintFacts` predicate sees zero backbone elements and
refuses valid Base/SFT parents before model loading. Candle already selects
explicit acceleration `Off`. Synthetic training fixtures construct complete
`Some(Off)` sources directly and therefore bypassed the faulty public factory.

Repair: select explicit `Off` in the MLX trainer, including the authenticated
ordinary transformer config, index and three shards. Keep generation's deferred
selection, strict header requirements and admitted eager-loading boundary intact.
Both Base/SFT and both adapter methods share this factory; Candle is unchanged.

`CI_IMPACT`: one ordinary pure regression calls the actual public Base/SFT trainer
factories with a tiny literal prepared pin and deliberately absent backbone pins.
It distinguishes the trainer's first required transformer path from generation's
first shared path, and asserts no eager loader entry. It reads no tensor headers,
creates no arrays, queries no device and executes no learned model or optimizer.
No workflow or learned oracle changes are needed. The required real-weight
campaign still must exercise the corrected production load path on its final pin;
this repair does not accept or relabel the earlier refusal as campaign evidence.

Validation: format, strict all-targets/test-support Clippy, the exact public-factory
regression and all 16 existing pure training-memory tests pass. Review-id and
clock-assertion ratchets pass. Independent focused review passed the final source
and fixture, and the coordinator independently repeated strict Clippy and the
exact public-factory test successfully. No local model/GPU execution.
