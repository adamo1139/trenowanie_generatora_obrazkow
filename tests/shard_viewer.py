"""Browse the shards of tests/fetch_preview_shards.py in a web browser (standard library only):
a grid of 384 px thumbnails per source (sortable by score, filterable by prompt text), and per image the 384 px
version next to the full-resolution original, with every caption and score.

  python tests/shard_viewer.py --root /path/data_preview --port 8765   # then open http://localhost:8765
"""
import argparse
import html
import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PER_PAGE = 120
CSS = """
:root{--bg:#f7f7f5;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e2e2dc;--acc:#2f6fdd}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecece8;--muted:#9a9a94;--card:#21211f;--line:#34342f;
--acc:#7aa7ff}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg)}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;
display:flex;flex-wrap:wrap;gap:10px;align-items:center}
a{color:var(--acc);text-decoration:none}nav a{padding:4px 10px;border-radius:6px;border:1px solid var(--line)}
nav a.on{background:var(--acc);color:#fff;border-color:var(--acc)}
input,select,button{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);
color:var(--fg)}main{padding:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px}
.cell{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.cell img{width:100%;aspect-ratio:1;object-fit:cover;display:block}
.cell div{padding:6px 8px;font-size:12px;color:var(--muted);display:-webkit-box;-webkit-line-clamp:3;
-webkit-box-orient:vertical;overflow:hidden}
.pair{display:grid;grid-template-columns:minmax(0,384px) minmax(0,1fr);gap:16px;align-items:start}
.pair figure{margin:0}.pair img{width:100%;border-radius:8px;border:1px solid var(--line)}
figcaption{color:var(--muted);font-size:12px;margin-top:4px}
table{border-collapse:collapse;width:100%;margin-top:16px}td{border-top:1px solid var(--line);padding:6px 8px;
vertical-align:top}td:first-child{color:var(--muted);white-space:nowrap;width:1%}
.pager{display:flex;gap:8px;align-items:center;margin:14px 0}
@media (max-width:700px){.pair{grid-template-columns:1fr}.grid{grid-template-columns:repeat(2,1fr)}}
"""


def load(root):
    data = {}
    for src in sorted(os.listdir(root)):
        p = os.path.join(root, src, "meta.jsonl")
        if os.path.exists(p):
            rows = [json.loads(line) for line in open(p)]
            have = set(os.listdir(os.path.join(root, src, "orig")))
            data[src] = [dict(r, has_orig=f"{r['key']}.jpg" in have) for r in rows]
    return data


def score_keys(rows):
    return sorted({k for r in rows[:50] for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                   and (k.startswith(("score", "aesthetic", "wk")))})


class App(BaseHTTPRequestHandler):
    root, data = None, None

    def log_message(self, *a):
        pass

    def send(self, body, ctype="text/html; charset=utf-8", code=200):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def page(self, title, body, src=None):
        nav = "".join(f'<a class="{"on" if s == src else ""}" href="/s/{s}">{html.escape(s)} ({len(r)})</a>'
                      for s, r in self.data.items())
        return (f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,"
                f"initial-scale=1'><title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
                f"<header><b>Shard viewer</b><nav>{nav}</nav></header><main>{body}</main></body></html>")

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = {k: v[-1] for k, v in urllib.parse.parse_qs(u.query).items()}
        parts = [urllib.parse.unquote(p) for p in u.path.strip("/").split("/") if p]
        if not parts:
            return self.send(self.page("Shard viewer", "<p>Pick a source above.</p>"))
        if parts[0] == "img" and len(parts) == 4 and parts[1] in self.data and parts[2] in ("orig", "thumb"):
            path = os.path.join(self.root, parts[1], parts[2], os.path.basename(parts[3]))
            if os.path.exists(path):
                ct = "image/webp" if path.endswith(".webp") else "image/jpeg"
                return self.send(open(path, "rb").read(), ct)
            return self.send("not found", "text/plain", 404)
        if parts[0] == "s" and len(parts) >= 2 and parts[1] in self.data:
            src, rows = parts[1], self.data[parts[1]]
            if len(parts) == 3:
                return self.detail(src, rows, parts[2], q)
            return self.grid(src, rows, q)
        self.send("not found", "text/plain", 404)

    def filtered(self, rows, q):
        text = q.get("q", "").lower()
        out = [r for r in rows if text in json.dumps(r, ensure_ascii=False).lower()] if text else list(rows)
        if q.get("sort"):
            out.sort(key=lambda r: (r.get(q["sort"]) is None, -(r.get(q["sort"]) or 0)))
        return out

    def grid(self, src, rows, q):
        rows = self.filtered(rows, q)
        pg = max(0, int(q.get("p", 0)))
        keys = score_keys(rows) if rows else []
        opts = "".join(f'<option value="{k}" {"selected" if q.get("sort") == k else ""}>{k} ↓</option>' for k in keys)
        form = (f'<form><input name=q placeholder="filter text" value="{html.escape(q.get("q", ""))}"> <select name=sort>'
                f'<option value="">shard order</option>{opts}</select> <button>Apply</button></form>')
        keep = urllib.parse.urlencode({k: v for k, v in q.items() if k != "p"})
        cells = []
        for r in rows[pg * PER_PAGE:(pg + 1) * PER_PAGE]:
            k = r["key"]
            cells.append(f'<a class=cell href="/s/{src}/{k}?{keep}"><img loading=lazy src="/img/{src}/thumb/{k}.webp">'
                         f'<div>{html.escape((r.get("prompt") or "")[:200])}</div></a>')
        n_pages = max(1, -(-len(rows) // PER_PAGE))
        pager = (f'<div class=pager>{"<a href=?%s&p=%d>← prev</a>" % (keep, pg - 1) if pg else ""}'
                 f'<span>page {pg + 1} / {n_pages} · {len(rows)} images</span>'
                 f'{"<a href=?%s&p=%d>next →</a>" % (keep, pg + 1) if pg + 1 < n_pages else ""}</div>')
        self.send(self.page(src, form + pager + f'<div class=grid>{"".join(cells)}</div>' + pager, src))

    def detail(self, src, rows, key, q):
        rows_f = self.filtered(rows, q)
        idx = next((i for i, r in enumerate(rows_f) if r["key"] == key), None)
        r = next((r for r in rows if r["key"] == key), None)
        if r is None:
            return self.send("not found", "text/plain", 404)
        keep = urllib.parse.urlencode(q)
        nav = ""
        if idx is not None:
            prev_k = rows_f[idx - 1]["key"] if idx > 0 else None
            next_k = rows_f[idx + 1]["key"] if idx + 1 < len(rows_f) else None
            nav = (f'<div class=pager><a href="/s/{src}?{keep}">↑ grid</a>'
                   f'{f"<a href=/s/{src}/{prev_k}?{keep}>← prev</a>" if prev_k else ""}'
                   f'<span>{idx + 1} / {len(rows_f)}</span>'
                   f'{f"<a href=/s/{src}/{next_k}?{keep}>next →</a>" if next_k else ""}</div>')
        orig = (f'<figure><a href="/img/{src}/orig/{key}.jpg" target=_blank><img src="/img/{src}/orig/{key}.jpg"></a>'
                f'<figcaption>original ({r.get("width") or 1024}×{r.get("height") or 1024}, click for 1:1)</figcaption>'
                f'</figure>' if r["has_orig"] else "<p>(original not downloaded yet)</p>")
        pair = (f'<div class=pair><figure><img src="/img/{src}/thumb/{key}.webp"><figcaption>384 px</figcaption>'
                f'</figure>{orig}</div>')
        rows_html = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{html.escape(json.dumps(v, ensure_ascii=False) if isinstance(v, list) else str(v))}</td></tr>"
            for k, v in r.items() if k not in ("has_orig",) and v not in (None, "", []))
        self.send(self.page(f"{src} {key}", nav + pair + f"<table>{rows_html}</table>" + nav, src))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    a = p.parse_args()
    App.root, App.data = a.root, load(a.root)
    print(f"{', '.join(f'{s}: {len(r)}' for s, r in App.data.items())} -> http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), App).serve_forever()


if __name__ == "__main__":
    main()
