Experiment specification (schema version 1)
========================================

Purpose and units
-----------------

Compare two fixed information protocols on synthetic one-step episodes with
three agents. Each utility is a negative illustrative cost per agent per
episode. There is no temporal state, learned representation, reinforcement
learning, physical system, or operational interpretation.

Data-generating process
-----------------------

For each episode, first draw hidden disturbances:

* ``stable``: three independent Bernoulli(stable_prior) draws.
* ``localized``: Bernoulli(localized_prior) decides whether a shock occurs;
  its location is uniformly selected from the three agents.
* ``shared``: one Bernoulli(shared_prior) draw is copied to all agents.

Then independently flip each local hidden bit with probability ``noise`` to
produce observations. All probability parameters accept the endpoints 0 and 1.
Independent episodes have no persistent shock or agent state. "Stable" is
a descriptive name for the default low independent prior, not an enforced
stationarity or no-disturbance guarantee.

Only the evaluator retains ``Episode.truth``. The reporting function accepts
``Observation`` alone. The action function accepts only a policy name and
``PolicyInput``; it never accepts an ``Episode``. These are explicit software
boundaries, not a security sandbox for arbitrary hostile Python code.

Protocols and action costs
--------------------------

No sharing supplies each agent only its own observation. Sharing broadcasts
the three observations simultaneously and truthfully in agent-index order.
Each sender pays ``communication`` exactly once per sharing episode, including
senders whose policies ignore the reports. There is no recipient fee, message
loss, report falsification, suppression, or reporting strategy.

The only actions are ``continue`` and ``isolate``. A disturbed agent continuing
pays ``missed``. An undisturbed agent isolating pays ``false_isolation``.
The other two disturbance/action cases have zero action cost. Communication
cost is additive. All costs must be finite and nonnegative, with a finite
maximum combined cost. There are no cross-agent payoff externalities: changing
an opponent's action does not change an agent's utility. Shared observations
are useful only insofar as their correlation predicts that agent's disturbance.

Declared candidates and calibration
-----------------------------------

The ordered no-sharing class is ``continue, isolate, local``.
Sharing adds ``any_report, majority_report`` in that order. These last policies
isolate at report counts of at least one and at least two, respectively.
Policy class membership is fixed before generating samples. No randomized
policies, history-dependent policies, or arbitrary mappings are searched.

Pool calibration episodes equally across all declared seeds and all three
scenarios. For each mode and each agent independently, select the policy with
the highest pooled mean utility; exact ties choose the first declared policy.
Scoring a policy with a homogeneous profile gives the same own-agent score as
any opponent profile because utilities and reports are action-separable.
Freeze the two selected profiles before generating evaluation results. They
are scenario-agnostic: no scenario name or prior is a runtime policy input.

Streams and sample accounting
------------------------------

Use a private ``random.Random`` instance initialized with the string
``oes-v1:{seed}:{phase}:{scenario}``; phase is ``calibration`` or ``evaluation``.
Do not mutate the global RNG. This makes changing calibration count leave
evaluation streams unchanged. Within the same interpreter implementation,
configuration and seeds reproduce output. Record Python version externally
when comparing runs across versions; identical byte-for-byte output across
all future Python versions is not guaranteed.

With K seeds, C calibration episodes per scenario per seed, and N evaluation
episodes per scenario per seed, unique episode totals are 3*K*C and 3*K*N.
Each episode contains three agents, not three independent episode samples.
Both protocols and every tested unilateral deviation reuse the identical
evaluation episode objects. Reuse does not increase the unique sample count.
No extra evaluation randomness is drawn during policy or deviation scoring.

Held-out comparisons and restricted deviations
----------------------------------------------

For each seed/scenario, compute each agent's mean baseline utility in both
modes and their paired difference (sharing minus no sharing). A positive
difference favors sharing on that sample only. Averaging these episode-wise
paired differences is algebraically equivalent to subtracting paired means.

For each agent in each mode, enumerate its entire declared candidate class
against the fixed baseline opponents on these same episodes, including its
unchanged policy. Report all candidate utilities, the maximum tested utility,
the baseline utility, and their nonnegative difference. The maximum across
agents is named ``restricted empirical deviation gap``. The protocol itself
is fixed; a unilateral deviation cannot switch off sharing or its cost.

Calibration is held out from evaluation. However, the best *tested deviation*
is chosen using evaluation utility and is not itself validated on another
holdout. Its maximum utility is subject to finite-sample maximization bias.
No zero-gap result certifies absence of untested or population-level gains.

Average paired differences equally across scenarios within each seed, then
equally across seeds. Report per-agent standard error as sample standard
deviation of seed means / sqrt(K), or JSON null when K=1. Seeds reuse a pooled
calibration fit: this describes held-out seed variation conditional on that
fit, not uncertainty of the entire selection procedure. No hypothesis test,
population guarantee, or general sharing benefit follows.

JSON contract
--------------

``python -m oes_coordination`` writes JSON to stdout; invalid CLI configurations
produce an argparse error on stderr with exit code 2. JSON disallows NaN and
infinity. There are no timestamps, external datasets, or runtime dependencies.

* ``schema_version``: 1.
* ``configuration``: noise, priors, all costs, seeds, and both episode counts.
* ``sample_counts``: counts per seed/scenario and unique total counts.
* ``selected_profiles``: calibrated no-sharing and sharing profiles.
* ``results``: one entry per seed/scenario with ``modes`` and paired
  sharing-minus-no-sharing utility differences.
* Each mode: sample count, profile, policy class, agent baseline and tested
  deviation utilities, best tested deviation, per-agent improvement, and
  ``maximum_improvement``.
* ``paired_summary``: seed count, per-seed agent means, overall agent means,
  descriptive standard errors, and a caveat.
* ``toy_games``: independent coordination and matching-pennies payoff tables,
  all profile/best-response checks, pure equilibria, and existence flags.

The source defines the exact field names and declaration-order tie breaking.
Example commands and the utility and gap equations are in the root README.

Exhaustive toy checks versus empirical findings
-----------------------------------------------

The two-player coordination table has payoffs (4,4), (0,3), (3,0), (2,2)
for (A,A), (A,B), (B,A), (B,B). Its only pure equilibria are (A,A) and (B,B).
Matching pennies pays (1,-1) for equal actions and (-1,1) otherwise and has
no pure equilibrium. Enumerate every action profile and every unilateral
action, retaining all tied best responses. These are exhaustive arithmetic
checks for the specified finite tables only, not statistical estimates.
Neither toy is the three-agent disturbance model. Mixed equilibria are
outside scope, and no-pure-equilibrium results must not be called nonexistence
of any equilibrium.

Validation protocol
--------------------

Run ``python -m unittest discover -s tests -v`` from the repository root.
Run the JSON CLI with at least two seeds and inspect counts, configuration,
both modes, finite utilities, and both toy equilibrium flags. CI repeats
tests and a small JSON smoke check on Python 3.11, 3.12, 3.13, and 3.14.
The project uses no separate linter or build tool. CI configuration is not
evidence that a remote run passed; report executed and unexecuted checks
separately. No real-world validation or general equilibrium proof is attempted.
