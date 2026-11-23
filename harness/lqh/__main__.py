"""python -m lqh <command> ...

  run PLAN --out DIR [--port N] [--blocks 1,2]   run a campaign plan (servers, points, evidence, validation, metrics)
  validate RUN_DIR...                           validate run directories (writes validation.json)
  metrics RUN_DIR...                            validate and compute metrics.json
  report RESULTS_DIR... --out DIR               build the publication bundle (HTML, CSV, JSON, sidecars, ledgers)
  check BUNDLE_DIR                              independent aggregate checker against the raw evidence
  smoke [--out DIR]                             mock-server end-to-end smoke (harness behaviour only)
  infx-command ...                              print the pinned InferenceX client command for one point
"""
import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    command, rest = argv[0], argv[1:]
    if command == "run":
        from .runner import main as run
        return run(rest)
    if command in ("validate", "metrics"):
        import json
        from .analyze import compute
        from .validate import validate_run
        status = 0
        for path in rest:
            result = validate_run(path) if command == "validate" else compute(path)
            print(json.dumps({"run": path, "status": result["status"],
                              "errors": result.get("errors") or result.get("validation_errors")}))
            status |= result["status"] not in ("measured", "smoke_only")
        return status
    if command == "report":
        from .report import main as report
        return report(rest)
    if command == "check":
        from .checker import main as check
        return check(rest)
    if command == "smoke":
        from .smoke import main as smoke
        return smoke(rest)
    if command == "infx-command":
        from .adapters.infx import main as infx_main
        return infx_main(rest)
    print(f"unknown command {command}\n{__doc__}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
