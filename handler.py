"""
RunPod Serverless Worker for ML Project Labs
Receives an ordered list of notebook code cells:
  input: { "cells": [ { "id": "cell-1", "code": "x = 1" }, ... ] }
Returns:
  output: {
    "cells": [
      { "id": "cell-1", "text": "...", "error": null, "images": [] }
    ],
    "logs": "..."
  }
"""

import sys
import io
import traceback
import base64
import runpod

# Try to configure matplotlib for headless rendering
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def harvest_matplotlib_figures():
    """Extract any open matplotlib figures as base64-encoded PNG strings."""
    images = []
    if not HAS_MPL:
        return images
    try:
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("ascii")
            images.append(f"data:image/png;base64,{b64}")
            buf.close()
        plt.close("all")
    except Exception as e:
        print(f"[warning] failed to harvest figure: {e}", file=sys.stderr)
    return images


def handler(job):
    """Executes cells in order within a single persistent global namespace."""
    job_input = job.get("input", {})
    cells = job_input.get("cells", [])

    if not isinstance(cells, list):
        return {
            "error": "Invalid input: 'cells' must be a list of {id, code} objects",
            "cells": [],
            "logs": ""
        }

    # Shared execution context across all cells in this run
    exec_globals = {
        "__name__": "__main__",
        "__doc__": None,
        "__package__": None,
    }

    cell_results = []
    overall_logs = []

    for idx, cell in enumerate(cells):
        cell_id = cell.get("id", f"cell-{idx}")
        code = cell.get("code", "")

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        cell_error = None
        images = []

        try:
            sys.stdout = stdout_buf
            sys.stderr = stderr_buf

            # Compile first to detect syntax errors cleanly
            compiled = compile(code, f"<cell-{cell_id}>", "exec")
            exec(compiled, exec_globals)

            # Harvest any plots generated in this cell
            images = harvest_matplotlib_figures()

        except Exception as e:
            cell_error = traceback.format_exc()
            overall_logs.append(f"[cell {cell_id} failed]: {e}")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        stdout_text = stdout_buf.getvalue()
        stderr_text = stderr_buf.getvalue()

        output_text = stdout_text
        if stderr_text:
            output_text = (output_text + "\n" if output_text else "") + stderr_text

        cell_results.append({
            "id": cell_id,
            "text": output_text.strip() if output_text else None,
            "error": cell_error,
            "images": images,
        })

        # If a cell crashes, halt remaining cells (standard Jupyter behavior)
        if cell_error:
            overall_logs.append(f"Execution halted after error in cell {cell_id}.")
            # Append empty results for remaining unexecuted cells
            for remaining in cells[idx + 1:]:
                cell_results.append({
                    "id": remaining.get("id", ""),
                    "text": None,
                    "error": "Skipped due to error in previous cell",
                    "images": [],
                })
            break

    return {
        "cells": cell_results,
        "logs": "\n".join(overall_logs),
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

