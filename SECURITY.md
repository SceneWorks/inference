# Security Policy

## Supported releases

The newest immutable `runtime-*` tag is the supported product release. `main` is
the development line for the next release. Older tags remain available as exact
rollback points but may be superseded by a newer tag carrying security fixes.

## Reporting a vulnerability

Do not place exploit details, credentials, licensed model weights, private model
artifacts, or other sensitive material in an issue or pull request.

Report suspected vulnerabilities through an existing private SceneWorks
maintainer channel and identify `SceneWorks/inference`. A repository
administrator will open a draft GitHub Security Advisory when coordinated review
is needed. If no such channel is available, open a minimal repository issue that
contains only a request for private security contact; wait for a maintainer to
establish that contact before sharing details.

Include, when applicable:

- the affected `runtime-*` tag or commit;
- backend, operating system, hardware, and bundle name;
- impact and a minimal reproduction that contains no secrets or restricted
  weights;
- dependency, model repository, and immutable model revision involved; and
- any known mitigations or evidence that the issue is upstream.

SceneWorks maintainers will triage the report, coordinate fixes with affected
products or upstream projects, and agree on disclosure timing with the reporter.
No response or remediation SLA is promised by this policy.
