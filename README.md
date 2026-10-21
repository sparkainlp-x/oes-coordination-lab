# oes-coordination-lab

Standalone **Python 3.11+**, standard-library-only research package. Three agents
observe noisy local binary disturbances and choose `continue` or `isolate`.
Compare no sharing against a mandatory observation-reporting protocol, and
enumerate a **restricted empirical deviation gap** over declared finite policies.
All data, costs, and payoffs are synthetic and illustrative.

## Run

No installation or third-party dependencies are required. From the repository root:

```sh
python -m unittest discover -s tests -v
python -m oes_coordination --help
python -m oes_coordination --seeds 0 1 2 --calibration-episodes 200 --evaluation-episodes 1000 > /tmp/oes-results.json
python -m json.tool /tmp/oes-results.json > /tmp/oes-results-pretty.json
```

For a small smoke experiment:

```sh
python -m oes_coordination --seeds 11 22 --calibration-episodes 10 --evaluation-episodes 20
```

Configure `--noise`, `--stable-prior`, `--localized-prior`, `--shared-prior`,
`--missed-cost`, `--false-isolation-cost`, and `--communication-cost`.
Probabilities must be in [0, 1]; costs must be finite and nonnegative, including
their maximum possible combined payoff magnitude. Counts must be positive
integers and seeds distinct integers. One seed is supported for diagnostics
(standard error is then `null`); use multiple seeds for comparisons.

## Model and information boundaries

For agent \(i\in\{0,1,2\}\), hidden disturbance \(D_i\in\{0,1\}\) and observation
\(O_i=D_i\mathbin{\mathrm{xor}}E_i\), where the errors are independent
\(E_i\sim\mathrm{Bernoulli}(\eta)\). Each episode is independently generated:

| Scenario | Hidden disturbance distribution | Default prior |
| --- | --- | --- |
| `stable` | Independent \(D_i\sim\mathrm{Bernoulli}(p_s)\), low background rate by default | \(p_s=0.02\) |
| `localized` | With probability \(p_l\), exactly one uniformly chosen agent is disturbed; otherwise none | \(p_l=0.3\) |
| `shared` | With probability \(p_h\), all three are disturbed; otherwise none | \(p_h=0.3\) |

Thus `stable` means low independent background disturbances, not necessarily
zero; set `--stable-prior 0` for an undisturbed baseline. In `localized`, each
agent's marginal disturbance probability is \(p_l/3\). In `shared`, it is \(p_h\).
Default noise \(\eta=0.1\). Overriding priors can remove the qualitative contrast.

`GroundTruth`, `Observation`, `PolicyInput`, and `Action` are separate types.
Policies receive only `PolicyInput(local, reports)` and return an action:
they receive no ground truth, scenario label, prior, seed, episode, or utility.
The evaluator alone combines hidden truth with actions to calculate utilities.
Calibration can use synthetic truth to score candidates, but frozen policies
cannot access it when choosing actions on held-out episodes.

**Reporting protocol:** before actions, every agent simultaneously broadcasts
exactly its own observed bit, truthfully and without extra noise or delay.
Everyone receives the ordered vector \(R=(O_0,O_1,O_2)\), including their own
bit. There is no report of hidden truth, no sequential signaling, and no option
to suppress or alter a report. Each sender pays \(c_c\) once per episode, not
per recipient, even if its policy ignores the vector. No-sharing agents receive
only \(O_i\) and pay no communication cost. Reporting is **not strategic**:
there are no reporting actions, deception incentives, or reporting deviations.

The synthetic utility is the negative cost:

\[
u_i(D_i,a_i,S)=
-c_m D_i\,\mathbf{1}[a_i=\mathrm{continue}]
-c_f(1-D_i)\,\mathbf{1}[a_i=\mathrm{isolate}]
-c_c S ,
\]

where \(S=1\) for the sharing protocol and \(0\) otherwise. Defaults:
\(c_m=5,\ c_f=1,\ c_c=0.1\). Correct isolation and undisturbed continuation
have no action cost. There is no isolation cost when actually disturbed, no
reward term, no externality, and no coupling to opponents' actions.
This deliberate separability makes the disturbance experiment an
**information comparison**, not a model of strategic cooperation.

## Finite policies, calibration, and deviations

Declared deterministic policies (in tie-breaking order):

| Policy | Isolate iff | No sharing | Sharing |
| --- | --- | --- | --- |
| `continue` | Never | yes | yes |
| `isolate` | Always | yes | yes |
| `local` | \(O_i=1\) | yes | yes |
| `any_report` | \(\sum_j R_j\geq1\) | no | yes |
| `majority_report` | \(\sum_j R_j\geq2\) | no | yes |

Calibration pools all seeds and all three scenarios, with equal sample counts,
and selects each agent's highest mean-utility candidate separately in each
mode; ties use declaration order. The resulting two profiles are frozen
across all held-out scenarios and seeds. Because utilities and reports do not
depend on opponents' actions, independent calibration is valid here; it is
not an algorithm for solving general coupled games.

For fixed baseline profile \(\pi\), mode-specific class \(\mathcal P_S\), and
the **same** held-out episodes \(e_1,\ldots,e_N\), enumerate every unilateral
replacement, holding opponents fixed:

\[
\widehat U_i(q,\pi_{-i})=\frac1N\sum_{t=1}^N
u_i(e_t;q,\pi_{-i}),\qquad
\widehat g_i=\max\left(0,\max_{q\in\mathcal P_S}
\widehat U_i(q,\pi_{-i})-\widehat U_i(\pi)\right),\qquad
\widehat G=\max_i\widehat g_i.
\]

The identity replacement is included, so its gain is exactly zero and the
best tested deviation cannot be worse than baseline. JSON reports baseline
utility, every tested utility, best tested deviation and utility, each
nonnegative improvement, and maximum improvement, separately per seed/scenario.
Deviations cannot switch protocol or avoid its fee.

This is a **restricted empirical deviation gap**, not an unrestricted Nash
certificate. A zero gap means only no profitable tested deviation on those
finite samples. It says nothing about untested policies or population utility.
Choosing the largest held-out deviation is diagnostic and has selection bias;
it does not validate the selected deviation on a further independent sample.

## Paired results and separate toy games

Calibration and evaluation use separate deterministic random streams keyed by
seed, phase, and scenario. Both modes and all deviations reuse exactly the
same evaluation episodes. JSON includes the full configuration, sample counts,
selected profiles, per-agent sharing-minus-no-sharing mean differences, and
their per-seed summaries (equal scenario weighting).

The reported standard error is the sample standard deviation of seed means
divided by the square root of seed count, conditional on the pooled calibrated
profiles. It is descriptive, not a significance test or a confidence guarantee;
three default seeds are a small sample. Communication can hurt utility, and
sharing policies can ignore reports. No outcome is assumed to favor sharing.
See [the experiment specification](docs/experiment-spec.rst) for the exact
protocol and output interpretation.

Two **separate two-player finite normal-form toy games** are also included:

| Row / column | A | B |
| --- | --- | --- |
| A | (4, 4) | (0, 3) |
| B | (3, 0) | (2, 2) |

The coordination toy has exactly pure equilibria `(A,A)` and `(B,B)`.
The matching-pennies toy pays `(1,-1)` for equal actions and `(-1,1)` for
unequal actions; it has **no pure equilibrium**. JSON lists every profile,
payoff, unilateral action payoff, best response, and equilibrium check.
These exhaustive results apply **only to those declared finite games**,
not the disturbance experiment. No pure equilibrium does not mean no mixed
equilibrium; mixed equilibria are not computed.

## Validation and scope

`unittest` covers payoff tables, protocol costs, information boundaries,
determinism, invalid inputs, identity deviations, restricted-gap arithmetic,
paired evaluation, JSON CLI, and exact toy equilibria/no-equilibrium cases.
GitHub Actions runs the suite and a CLI smoke check on Python 3.11–3.14.
There is no additional build or lint tool.

No reinforcement learning, physical validation, operational safety claim,
general cooperation advantage, or general Nash-equilibrium claim is made.
Finite-sample findings are not proofs; exhaustive toy enumeration is not
evidence of real-world applicability. No license is added: license selection
is reserved for the owner.
