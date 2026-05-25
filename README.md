# OpenShift Virtualization Migration Advisor

> **Local-first migration assessment for moving legacy hypervisor workloads to Red Hat OpenShift Virtualization.** Powered by Gemma 4 26B MoE running on Ollama. No configuration data ever leaves the host.

Submission for the [DEV Gemma 4 Challenge — Build with Gemma 4](https://dev.to/challenges/google-gemma-2026-05-06).

---

## The problem

Enterprises consolidating off VMware vSphere and legacy KVM/RHV onto OpenShift Virtualization have a discovery problem: their VM inventories live in `.vmx` files, libvirt XML, and oVirt exports — and those files contain sensitive infrastructure detail (storage paths, network topology, secrets references, FIPS posture, licensing). Sending them to a hosted LLM is a non-starter for regulated workloads.

This tool does the assessment locally. Paste a VM config or upload an inventory file → get a structured migration report covering inventory mapping, OpenShift Virt primitives, compatibility risks, migration tooling recommendation, effort sizing, and security posture preservation.

## Why Gemma 4 26B MoE — and not 31B Dense

I started on **31B Dense** for maximum reasoning quality on multi-VM inventories. I hit two issues that made it the wrong choice for this workload:

1. **Ollama Flash Attention prefill stall on Dense** ([ollama#15350](https://github.com/ollama/ollama/issues/15350)) hangs the 31B variant on prompts beyond ~3–4K tokens. Real-world datacenter inventories blow past that on the first VM with multiple disks. The bug is specific to the Dense model's hybrid sliding+global attention; the MoE variant handles the same prompts cleanly.
2. **Active-parameter efficiency.** 26B MoE activates ~4B parameters per token versus 31B for Dense. On a 24 GB consumer GPU that's the difference between comfortable headroom for the full 256K context KV cache and constant CPU offload.

What I kept:
- **256K context window** — enough to ingest an entire small-datacenter inventory in one shot, which matters for affinity-group reasoning and dependency mapping across VMs.
- **Native reasoning mode** — enabled via the `<|think|>` system-prompt token for the longer multi-VM reports.
- **Native function-calling support** — used for structured-output enforcement on the inventory table (planned v2).

I would not have picked MoE if the workload was short, single-turn, math/code reasoning where Dense's per-token capacity matters more than throughput. For long, structured, enterprise-document reasoning, 26B MoE is the right tool.

## What it does

Input: a VMware `.vmx`, libvirt domain XML, OVF descriptor, or RHV/oVirt inventory export.

Output: a Markdown migration report with six sections:

1. **Inventory Summary** — table of every VM with vCPU/memory/disk/network/OS
2. **OpenShift Virtualization Equivalents** — mapping to `VirtualMachine`, `DataVolume`, `NetworkAttachmentDefinition`, `StorageClass`
3. **Compatibility & Risk Flags** — flags PCI passthrough, SR-IOV, NUMA pinning, huge pages, legacy guest OSes, IDE controllers, etc.
4. **Migration Path** — MTV (Migration Toolkit for Virtualization) vs `virt-v2v` vs cold image conversion, per VM
5. **Effort & TCO Notes** — S/M/L sizing with justification, OpenShift Virt subscription surface area, consolidation opportunities
6. **Security & Compliance** — encrypted disk → encrypted PVC, isolated VLAN → NetworkPolicy, FIPS mode → FIPS-enabled cluster

## Quickstart

```bash
# 1. Install Ollama from https://ollama.com/download
ollama serve

# 2. Pull Gemma 4 26B MoE (~16 GB)
ollama pull gemma4:26b

# 3. Clone and run
git clone https://github.com/<your-handle>/openshift-migration-advisor.git
cd openshift-migration-advisor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open <http://localhost:7860>, pick a sample from the dropdown (or paste your own config), and click **Generate migration report**.

## Sample inputs included

- `samples/vmware-web-tier.vmx` — VMware web-tier VM with encryption + secure boot + vMotion
- `samples/libvirt-db-server.xml` — libvirt database VM with NUMA pinning, huge pages, PCI passthrough, host-passthrough CPU
- `samples/rhv-mixed-inventory.txt` — six-VM RHV inventory including a Windows 2008 R2 legacy workload, an Oracle DB with SR-IOV + FIPS, and anti-affinity policies

Each sample is designed to trigger a specific class of compatibility risk so judges can see the model's reasoning surface.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Gradio UI      │───▶│  Source detect   │───▶│ Gemma 4 26B MoE     │
│  (paste/upload) │    │  + prompt build  │    │ via Ollama (local)  │
└─────────────────┘    └──────────────────┘    └──────────┬──────────┘
                                                          │
                       ┌──────────────────────────────────▼──────┐
                       │ Streamed Markdown report → UI + on-disk │
                       │              output/                    │
                       └─────────────────────────────────────────┘
```

Single file (`app.py`), single model call, streamed token-by-token. No vector store, no agent loop — the 256K context window means the entire inventory fits in one prompt.

## Ollama configuration recommended for long contexts

```bash
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q4_0
ollama serve
```

These keep the KV cache compact enough to fit the 256K window on consumer GPUs.

## Hardware tested

- NVIDIA RTX-class laptop GPU, 16 GB VRAM
- 32 GB system RAM
- `gemma4:26b` Q4_K_M quantization, ~16 GB on disk

## Limitations & honest caveats

- The model occasionally invents OpenShift Virt API field names. Verify against the [KubeVirt API reference](https://kubevirt.io/api-reference/) before applying generated YAML.
- The TCO section is directional, not authoritative — it does not query Red Hat list prices.
- vSphere advanced features (DRS, FT, vSAN policies) are flagged but not deeply analyzed.
- No automated MTV plan generation yet — planned next.

## Roadmap

- Structured-output mode via Gemma 4's native function calling — emit a JSON migration plan that feeds directly into MTV
- Multi-VM dependency graph extraction (the 256K context already supports it; UI needs to render it)
- Cost calculator that takes the structured plan and produces a TCO delta vs. source

## Licence

Apache 2.0. Gemma 4 model usage is governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
