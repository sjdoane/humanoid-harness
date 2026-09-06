# G1 learning evidence view

| status | evidence |
|---|---|
| progress | The local UI can show explicitly registered G1 course runs after the existing feedback/evaluator validator accepts their retained evidence. |
| bottleneck | No run is registered by source control; the ignored local registry remains a deliberate researcher action. |
| next step | Review a manifest and digest, register only the intended snapshot, then open **G1 learning** and choose **Validate runs**. |

## Local registration

The server reads only `artifacts/gmt/g1_learning_registry.json`. It does not
accept a path from an HTTP query. The file is ignored by Git and must contain at
most 12 entries:

```json
{
  "schema_version": 1,
  "runs": [
    {
      "run_id": "descriptive-local-id",
      "manifest_path": "/absolute/path/to/course_run_manifest.json",
      "manifest_sha256": "64-lowercase-hex-characters",
      "label": "final_policy"
    }
  ]
}
```

`label` is `final_policy` or `zero_residual`. Find the exact manifest digest
with `shasum -a 256 /absolute/path/to/course_run_manifest.json`, create the
registry under `artifacts/gmt/`, and then run the existing `humanoid-harness ui`
command. Do not commit the registry or local run data.

## Reading the result

- **PASS** means every fixed G1 development gate passed. It is not protected,
  held-out, or formal task-success evidence; `episode_success` remains absent.
- Metrics come from the retained objective evaluation after the existing G1
  feedback builder recomputes and cross-checks the trace.
- Trainer settings are producer-recorded manifest facts. The UI shows the
  effective contract and a derived SHA-256 rather than guessing a trainer name.
- Validation is sequential, happens only when the view is first opened or the
  button is pressed, and rejects concurrent refreshes. There is no polling.
- The older four-robot monitor remains historical native `Humanoid-v5`
  evidence. It is not mixed with G1 results.
