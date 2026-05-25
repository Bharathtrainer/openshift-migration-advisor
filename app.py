"""
OpenShift Virtualization Migration Advisor
-------------------------------------------
A local-first migration assessment tool that ingests legacy hypervisor
configurations (VMware .vmx, libvirt XML, RHV/oVirt exports) and produces
a structured migration report for Red Hat OpenShift Virtualization.

Powered by Gemma 4 26B MoE running locally via Ollama. No configuration
data ever leaves the host machine — a hard requirement for regulated
enterprises where VM inventories contain sensitive infrastructure secrets.

Submission for the DEV Gemma 4 Challenge — Build with Gemma 4.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import ollama

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL = os.environ.get("GEMMA_MODEL", "gemma4:26b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Gemma 4 recommended sampling defaults
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}

client = ollama.Client(host=OLLAMA_HOST)

# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a senior Red Hat OpenShift Virtualization migration architect.
You assess legacy hypervisor configurations (VMware vSphere, libvirt/KVM, Red Hat
Virtualization / oVirt) and produce structured migration reports for moving
workloads to OpenShift Virtualization (KubeVirt-based).

You must reason from the provided configuration ONLY. Do not invent VMs, disks,
networks, or resources that are not in the source. If a field is missing, mark it
as "not specified in source" rather than guessing.

Your output is consumed by infrastructure architects. Be precise, terse, and
opinionated. Flag risks clearly. Reference real OpenShift Virtualization
primitives (VirtualMachine, DataVolume, NetworkAttachmentDefinition, StorageClass,
CDI, MTV / Migration Toolkit for Virtualization) where relevant."""

USER_PROMPT_TEMPLATE = """Analyse the following source hypervisor configuration and produce
a migration assessment report for Red Hat OpenShift Virtualization.

Return the report as Markdown with these exact sections:

## 1. Inventory Summary
A table of every VM/workload found in the source. Columns: Name, vCPU, Memory,
Disk(s), Network(s), Guest OS, Notes.

## 2. OpenShift Virtualization Equivalents
For each VM, map source resources to OpenShift Virt primitives. Show the target
VirtualMachine CR shape (kind, key spec fields — not full YAML), required
DataVolume / StorageClass choices, and NetworkAttachmentDefinition needs.

## 3. Compatibility & Risk Flags
Bullet list. Flag anything that does NOT translate cleanly: legacy SCSI
controllers, raw device mappings, vSphere-specific features (vMotion, DRS,
FT, snapshots), unsupported guest OSes, custom kernel modules, PCI passthrough,
SR-IOV requirements, large memory pages, NUMA pinning.

## 4. Migration Path
Recommend the concrete tooling path: MTV (Migration Toolkit for Virtualization)
plan vs. virt-v2v vs. cold migration with image conversion. Justify in one
sentence per VM.

## 5. Effort & TCO Notes
Rough effort sizing (S / M / L) per VM with one-line justification. List the
OpenShift Virt licensing surface area (subscriptions, node count assumptions).
Call out where consolidation is possible.

## 6. Security & Compliance
Note any source-side security posture that must be preserved (encrypted disks,
isolated networks, secrets, certificates, FIPS mode). Map each to its OpenShift
Virt equivalent (encrypted PVCs, NetworkPolicy, Sealed Secrets, FIPS-enabled
cluster).

---

SOURCE CONFIGURATION ({source_type}):

```
{config}
```
"""


# -----------------------------------------------------------------------------
# Core
# -----------------------------------------------------------------------------
def detect_source_type(text: str) -> str:
    """Best-effort detection of the source hypervisor format."""
    head = text.lstrip()[:400].lower()
    if head.startswith(".encoding") or "virtualhw.version" in head:
        return "VMware vSphere (.vmx)"
    if "<domain" in head and "kvm" in head:
        return "libvirt / KVM (domain XML)"
    if "<ovf:envelope" in head or "<envelope" in head:
        return "OVF / OVA descriptor"
    if "rhv" in head or "ovirt" in head:
        return "Red Hat Virtualization / oVirt export"
    return "Unknown / generic hypervisor config"


def generate_report(config_text: str, source_hint: str | None = None):
    """Stream a migration report token-by-token from Gemma 4."""
    if not config_text or not config_text.strip():
        yield "_Please paste a configuration or upload a file first._"
        return

    source_type = source_hint or detect_source_type(config_text)
    prompt = USER_PROMPT_TEMPLATE.format(source_type=source_type, config=config_text)

    started = time.time()
    output = f"> **Source detected:** `{source_type}`  \n"
    output += f"> **Model:** `{MODEL}`  \n"
    output += f"> **Generated:** `{datetime.utcnow().isoformat()}Z`\n\n---\n\n"
    yield output

    try:
        stream = client.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options=SAMPLING,
            stream=True,
        )
        for chunk in stream:
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                output += piece
                yield output
    except ollama.ResponseError as e:
        yield output + f"\n\n**Ollama error:** `{e}`\n\nIs `ollama serve` running and `{MODEL}` pulled?"
        return
    except Exception as e:
        yield output + f"\n\n**Error:** `{type(e).__name__}: {e}`"
        return

    elapsed = time.time() - started
    output += f"\n\n---\n_Report generated locally in {elapsed:.1f}s. No data left this machine._\n"
    yield output

    # Persist to disk
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    (OUTPUT_DIR / f"report-{ts}.md").write_text(output, encoding="utf-8")


def load_sample(name: str) -> str:
    path = Path(__file__).parent / "samples" / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def on_file_upload(file_obj):
    if file_obj is None:
        return ""
    return Path(file_obj.name).read_text(encoding="utf-8", errors="replace")


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
CSS = """
.gradio-container {max-width: 1200px !important;}
#title-row {border-bottom: 1px solid #eee; padding-bottom: 10px;}
"""

with gr.Blocks(title="OpenShift Virt Migration Advisor") as demo:
    with gr.Row(elem_id="title-row"):
        gr.Markdown(
            "# 🛰️ OpenShift Virtualization Migration Advisor\n"
            "**Local-first migration assessment.** Paste a VMware `.vmx`, libvirt domain XML, "
            "or OVF descriptor — get a structured Red Hat OpenShift Virtualization migration "
            f"report. Powered by **Gemma 4 26B MoE** running on Ollama. Nothing leaves this host."
        )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Source configuration")
            sample_dropdown = gr.Dropdown(
                label="Load a sample",
                choices=[
                    "(none)",
                    "vmware-web-tier.vmx",
                    "libvirt-db-server.xml",
                    "rhv-mixed-inventory.txt",
                ],
                value="(none)",
            )
            file_upload = gr.File(
                label="…or upload a config file",
                file_types=[".vmx", ".xml", ".ovf", ".txt", ".yaml", ".yml"],
            )
            config_input = gr.Textbox(
                label="Configuration text",
                placeholder="Paste your VM configuration here…",
                lines=20,
                max_lines=40,
            )
            source_hint = gr.Dropdown(
                label="Source type (override auto-detect)",
                choices=[
                    "Auto-detect",
                    "VMware vSphere (.vmx)",
                    "libvirt / KVM (domain XML)",
                    "OVF / OVA descriptor",
                    "Red Hat Virtualization / oVirt export",
                ],
                value="Auto-detect",
            )
            run_btn = gr.Button("Generate migration report", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Migration report")
            report_output = gr.Markdown(
                value="_Report will appear here. Streaming token-by-token from local Gemma 4._",
                height=720,
            )

    # Wiring
    sample_dropdown.change(
        lambda name: load_sample(name) if name and name != "(none)" else "",
        inputs=sample_dropdown,
        outputs=config_input,
    )
    file_upload.change(on_file_upload, inputs=file_upload, outputs=config_input)

    def _go(cfg, hint):
        hint_val = None if hint == "Auto-detect" else hint
        yield from generate_report(cfg, hint_val)

    run_btn.click(_go, inputs=[config_input, source_hint], outputs=report_output)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, css=CSS)
