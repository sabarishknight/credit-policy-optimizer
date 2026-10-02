"""Minimal notebook executor (no Jupyter needed).

Reads a percent-format script (``# %%`` code cells, ``# %% [markdown]`` cells),
executes every code cell in one namespace and writes a v4 .ipynb with outputs
(stdout, rich HTML tables and inline PNG figures) embedded.
"""
import ast, base64, io, json, sys, time, traceback, contextlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse(path):
    cells, cur, kind = [], [], "code"
    for line in open(path, encoding="utf-8").read().splitlines():
        if line.startswith("# %%"):
            if cur and any(l.strip() for l in cur):
                cells.append((kind, cur))
            kind = "markdown" if "[markdown]" in line else "code"
            cur = []
            continue
        cur.append(line)
    if cur and any(l.strip() for l in cur):
        cells.append((kind, cur))
    out = []
    for kind, lines in cells:
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else (l[1:] if l.startswith("#") else l) for l in lines]
        out.append((kind, "\n".join(lines)))
    return out


def run(src, dst):
    cells = parse(src)
    ns = {"__name__": "__main__"}
    nb_cells, outputs = [], None
    count = 0

    def display(obj):
        outputs.append(rich(obj))

    def rich(obj):
        data = {"text/plain": repr(obj)}
        if hasattr(obj, "to_html") and hasattr(obj, "columns"):
            data["text/html"] = obj.to_html(border=0, classes="dataframe", float_format=lambda x: f"{x:,.4g}")
            data["text/plain"] = obj.to_string()
        elif hasattr(obj, "_repr_html_"):
            data["text/html"] = obj._repr_html_()
        return {"output_type": "display_data", "data": data, "metadata": {}}

    def flush_figs():
        for num in plt.get_fignums():
            fig = plt.figure(num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
            outputs.append({"output_type": "display_data",
                            "data": {"image/png": base64.b64encode(buf.getvalue()).decode(), "text/plain": "<Figure>"},
                            "metadata": {}})
        plt.close("all")

    ns["display"] = display
    plt.show = lambda *a, **k: flush_figs()

    for kind, text in cells:
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})
            continue
        count += 1
        outputs = []
        t0 = time.time()
        buf = io.StringIO()
        try:
            tree = ast.parse(text)
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
            with contextlib.redirect_stdout(buf):
                exec(compile(tree, f"<cell {count}>", "exec"), ns)
                val = eval(compile(last, f"<cell {count}>", "eval"), ns) if last else None
            if buf.getvalue():
                outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": buf.getvalue()})
            flush_figs()
            if val is not None:
                r = rich(val)
                r.update({"output_type": "execute_result", "execution_count": count})
                outputs.append(r)
        except Exception:
            print(buf.getvalue())
            print(f"\n*** ERROR in cell {count}:\n{text[:400]}\n")
            traceback.print_exc()
            sys.exit(1)
        print(f"cell {count:>3} ok  {time.time()-t0:6.1f}s", file=sys.stderr)
        nb_cells.append({"cell_type": "code", "execution_count": count, "metadata": {},
                         "outputs": outputs, "source": text.splitlines(True)})

    nb = {"cells": nb_cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": sys.version.split()[0]}},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(dst, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", dst, file=sys.stderr)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
