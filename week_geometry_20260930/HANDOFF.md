# Week geometry queue handoff
No GPU started by this branch author. Root controller must wait prior extension complete and choose GPU.
Manifest jobs.json: freeze, fresh d10/d30 dev, dev-only train gate, fresh d10/d30 test, conditional d10 4arms×3seed2000 updates, always-run partial report.
CLI paths and all completion markers are explicit in jobs.json. CUDA_VISIBLE_DEVICES comes from root. gate.json has allow boolean and numeric evidence, no test read.
Negative dev gate stops every training job; report still lists screening outcomes. Root deadline Oct7 15:29UTC and no-new-train Oct6 15:29UTC.
All artifacts use project-linked attached storage. Full checkpoint resume uses actor/critic/optimizer/scheduler/RNG/lambda; paused is not complete.
Screen selection is frozen before oracle labels. Diagnostic hard-radius and original-soft-dual scopes are separate; training never filters radius.
Not executed or tested here; engineering tests not authorized. Actual progress will be in artifacts/*/heartbeat.json and root logs. Code manually reviewed only.

Cost gate: 21jobs total. After positive dev, geometry seed101 first runs original-science expected2000 with stop_after100, fullcheckpoint; prefix max_wall21600 estimated7200. budget_gate.py uses actual recorded update times and remaining deadline, allows full12jobs only if projected whole3seed matchedblock fits min72h/45%remaining with1.25safety. Otherwise prefix retained, never called complete2000. Final primary latest2000; validation-best secondary with explicit selected update.
