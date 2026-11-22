"""
ui/templates.py
───────────────
HTML template helpers and component generators for the DocIQ Streamlit UI.
Keeps UI python logic clean and completely decoupled from raw markup strings.
"""

import base64
from pathlib import Path
import uuid


def load_stylesheet(css_filename: str = "styles.css") -> str:
    """Reads and returns the CSS file content wrapped in <style> tags."""
    css_path = Path(__file__).parent / css_filename
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            return f"<style>\n{f.read()}\n</style>"
    return ""


# ── Welcome Hero Banner ───────────────────────────────────────────────────
HERO_BANNER_HTML = """
<div class="hero-box">
    <span class="hero-tag">✨ Multimodal Hybrid RAG</span>
    <span class="hero-tag" style="background:rgba(147, 51, 234, 0.2); border-color:rgba(147, 51, 234, 0.4); color:#c084fc;">📊 Adaptive Visual Flowcharts</span>
    <div class="hero-title">Ask anything from your documents</div>
    <p class="hero-desc">
        DocIQ parses text, diagrams, images, and tables from PDFs. Generates <b>interactive visual flowcharts</b> automatically for workflows, architectures, and comparisons.
    </p>
</div>
"""


# ── Mode Badge ────────────────────────────────────────────────────────────
def get_mode_badge_html(mode: str) -> str:
    """Render the top right mode badge."""
    return f"""
    <div class="mode-badge-container">
        <span class="mode-badge">Mode: {mode}</span>
    </div>
    """


# ── System Health Status Pill ─────────────────────────────────────────────
def get_health_pill_html(is_healthy: bool, chunks_count: int = 0) -> str:
    """Render the sidebar system health pill."""
    if is_healthy:
        return f"<div class='status-pill-healthy'>● Connected · {chunks_count:,} chunks</div>"
    return "<div class='status-pill-warning'>▲ API Connecting...</div>"


# ── Confidence Badge ──────────────────────────────────────────────────────
def get_confidence_badge_html(confidence: float) -> str:
    """Render a colored confidence pill based on score."""
    bg_color = "#16a34a" if confidence > 0.8 else "#d97706" if confidence > 0.5 else "#dc2626"
    return f"<span class='confidence-badge' style='background:{bg_color};'>Confidence: {confidence:.0%}</span>"


# ── Source Citation Card ──────────────────────────────────────────────────
def get_source_card_html(chunk_type: str, filename: str, page: str, chunk_id: str) -> str:
    """Render a styled source citation metadata card."""
    ctype = str(chunk_type).lower()
    badge_class = (
        "source-badge-table" if ctype == "table"
        else "source-badge-image" if ctype == "image"
        else "source-badge-text"
    )
    return (
        f"<div class='source-card'>"
        f"<span class='{badge_class}'>{ctype}</span>"
        f"<span class='source-filename'>📄 {filename or 'Document'}</span>"
        f"<span class='source-page-pill'>Page {page}</span>"
        f"<span class='source-id-pill'>[{chunk_id}]</span>"
        f"</div>"
    )


# ── Mermaid Interactive SVG Frame ─────────────────────────────────────────
MERMAID_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <style>
    body {
      background-color: transparent;
      margin: 0;
      padding: 6px;
      display: flex;
      justify-content: center;
      align-items: flex-start;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .flow-wrapper {
      width: 100%;
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 22px 20px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 140px;
      overflow-x: auto;
    }
    .flow-target {
      width: 100%;
      display: flex;
      justify-content: center;
      align-items: center;
    }
    .flow-target svg {
      max-width: 100% !important;
      height: auto !important;
    }
    /* Visual Cards Fallback */
    .flow-cards-container {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: center;
      gap: 16px;
      width: 100%;
      padding: 8px;
    }
    .flow-card {
      background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
      border: 1px solid #3b82f6;
      border-radius: 10px;
      padding: 14px 20px;
      color: #f8fafc;
      font-weight: 600;
      font-size: 14px;
      box-shadow: 0 4px 14px rgba(59, 130, 246, 0.18);
      max-width: 300px;
      text-align: center;
    }
    .flow-arrow {
      display: flex;
      flex-direction: column;
      align-items: center;
      color: #94a3b8;
      font-size: 12px;
      font-weight: 500;
    }
    .flow-arrow-line {
      display: flex;
      align-items: center;
      color: #60a5fa;
      font-size: 20px;
      line-height: 1;
    }
    .flow-arrow-label {
      background: #1e293b;
      border: 1px solid #475569;
      border-radius: 6px;
      padding: 2px 8px;
      margin-bottom: 4px;
      color: #cbd5e1;
      font-size: 11px;
    }
  </style>
</head>
<body>
  <div class="flow-wrapper">
    <div id="target___UID__" class="flow-target">
      <span style="color:#64748b; font-size:13px;">Loading diagram...</span>
    </div>
  </div>

  <script>
    const code = atob("__B64__");
    const target = document.getElementById("target___UID__");

    mermaid.initialize({
      startOnLoad: false,
      securityLevel: 'loose',
      theme: 'dark',
      themeVariables: {
        darkMode: true,
        background: '#0f172a',
        primaryColor: '#1e40af',
        primaryTextColor: '#f8fafc',
        primaryBorderColor: '#60a5fa',
        lineColor: '#94a3b8',
        edgeLabelBackground: '#1e293b',
        secondaryColor: '#059669',
        tertiaryColor: '#1e293b',
        fontSize: '13px'
      }
    });

    function renderFallbackCards(str) {
      const nodes = [];
      const edges = [];
      const lines = str.split('\\n');
      
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        // Extract node text between quotes
        const qMatches = line.match(/"([^"]+)"/g);
        if (qMatches) {
          for (let j = 0; j < qMatches.length; j++) {
            const label = qMatches[j].replace(/^"|"$/g, '').trim();
            if (label && nodes.indexOf(label) === -1 && label !== "Referenced by") {
              nodes.push(label);
            }
          }
        }
        // Extract arrow labels between pipes |label|
        const pipeMatch = line.match(/\\|([^|]+)\\|/);
        if (pipeMatch) {
          edges.push(pipeMatch[1].replace(/["']/g, '').trim());
        }
      }

      if (nodes.length === 0) {
        target.innerHTML = '<div style="color:#94a3b8; font-size:13px; text-align:center;">📊 Visual flow represented in explanation table</div>';
        return;
      }

      let html = '<div class="flow-cards-container">';
      for (let i = 0; i < nodes.length; i++) {
        html += '<div class="flow-card">' + nodes[i] + '</div>';
        if (i < nodes.length - 1) {
          const edgeLabel = edges[i] ? '<span class="flow-arrow-label">' + edges[i] + '</span>' : '';
          html += '<div class="flow-arrow">' + edgeLabel + '<span class="flow-arrow-line">───►</span></div>';
        }
      }
      html += '</div>';
      target.innerHTML = html;
    }

    async function doRender() {
      try {
        const res = await mermaid.render("svg___UID__", code);
        const errDiv = document.getElementById("dsvg___UID__");
        if (errDiv) errDiv.remove();
        target.innerHTML = res.svg;
      } catch (err) {
        console.warn("Mermaid primary render caught:", err);
        const errDiv = document.getElementById("dsvg___UID__");
        if (errDiv) errDiv.remove();

        try {
          const cleanLines = ['flowchart TD'];
          const rawLines = code.split('\\n');
          for (let i = 0; i < rawLines.length; i++) {
            const line = rawLines[i].trim();
            if (line.indexOf('-->') !== -1 && !line.endsWith('--')) {
              cleanLines.push(line.replace(/&/g, 'and').replace(/\\|["'](.*?)["']\\|/g, '|$1|'));
            }
          }
          if (cleanLines.length > 1) {
            const res2 = await mermaid.render("svg_fb___UID__", cleanLines.join('\\n'));
            const errDiv2 = document.getElementById("dsvg_fb___UID__");
            if (errDiv2) errDiv2.remove();
            target.innerHTML = res2.svg;
            return;
          }
        } catch (err2) {
          const errDiv2 = document.getElementById("dsvg_fb___UID__");
          if (errDiv2) errDiv2.remove();
        }

        renderFallbackCards(code);
      }
    }

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", doRender);
    } else {
      doRender();
    }
  </script>
</body>
</html>
"""


def get_mermaid_html_container(raw_code: str, element_key: str) -> str:
    """Generate an isolated, robust HTML document for rendering Mermaid diagrams with zero syntax bombs."""
    uid = f"{element_key}_{uuid.uuid4().hex[:6]}"
    b64_code = base64.b64encode(raw_code.strip().encode("utf-8")).decode("ascii")
    return MERMAID_HTML_TEMPLATE.replace("__B64__", b64_code).replace("__UID__", uid)
