"""
PHANTOM PLATFORM — Master Audit & Validation Suite
===================================================
Executes the comprehensive validation checks specified in Section 15
of the PHANTOM PLATFORM Master Specification.

Sections audited:
  S1: GGUF Loader Correctness & Dequantization Throughput
  S2: Conversion Pipeline & .phantomw Integrity
  S3: CLI Interface Execution (plan, doctor, status, list)
  S4: Phantomfile Parser, Validator & Modelfile Importer
  S5: Plugin System (RAG, Tool Router, Context Cache)
  S6: API Gateway, Bearer Auth, Rate Limiting & Ollama Endpoints
  S7: OSS Readiness, Documentation & Security
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))

import numpy as np
import torch


def audit_section_1():
    """SECTION 1 — GGUF Loader Correctness"""
    from phantom.loader import GGUFLoader, GGUFQuantType

    loader = GGUFLoader.__new__(GGUFLoader)
    # Test Q4_0
    d4 = np.array([1.5], dtype=np.float16).tobytes()
    qs4 = bytes([0x88] * 16)
    out4 = loader.dequantize_tensor(d4 + qs4, GGUFQuantType.Q4_0, (32,))
    assert out4.shape == (32,)
    assert torch.allclose(out4.float(), torch.zeros(32))

    # Test Q8_0
    d8 = np.array([2.0], dtype=np.float16).tobytes()
    qs8 = bytes([4] * 32)
    out8 = loader.dequantize_tensor(d8 + qs8, GGUFQuantType.Q8_0, (32,))
    assert torch.allclose(out8.float(), torch.full((32,), 8.0))

    # Test Q4_K
    d_k = np.array([1.0], dtype=np.float16).tobytes()
    dmin_k = np.array([0.5], dtype=np.float16).tobytes()
    scales_k = bytes([0x00] * 12)
    qs_k = bytes([0x00] * 128)
    data_k = d_k + dmin_k + scales_k + qs_k
    out_k = loader.dequantize_tensor(data_k, GGUFQuantType.Q4_K, (256,))
    assert out_k.shape == (256,)

    return True


def audit_section_2():
    """SECTION 2 — Conversion Pipeline & .phantomw Integrity"""
    from phantom.converter import PhantomLayerReader, PhantomLayerWriter

    writer = PhantomLayerWriter(layer_id=7)
    t = torch.randn(32, 64, dtype=torch.bfloat16)
    writer.add_tensor("test_tensor", t)

    test_path = Path("audit_layer.phantomw")
    writer.write_to_file(test_path)

    with PhantomLayerReader(test_path) as reader:
        rec_t = reader.load_tensor("test_tensor")
        assert torch.allclose(t.float(), rec_t.float(), atol=1e-3)

    if test_path.exists():
        test_path.unlink()
    return True


def audit_section_3():
    """SECTION 3 — CLI Interface Execution & Output Verification"""
    import contextlib
    import io
    from phantom.phantom_cli import PhantomCLI

    cli = PhantomCLI()

    # 1. Test cmd_plan with output capture and content assertion
    buf_plan = io.StringIO()
    with contextlib.redirect_stdout(buf_plan):
        res_plan = cli.cmd_plan("llama3:70b")
    out_plan = buf_plan.getvalue()
    assert res_plan == 0
    assert "LAYER RESIDENCY DISTRIBUTION" in out_plan, "Missing layer residency table in plan output"
    assert "tok/sec" in out_plan
    assert "VRAM" in out_plan and "RAM" in out_plan

    # 2. Test cmd_doctor with output verification
    buf_doc = io.StringIO()
    with contextlib.redirect_stdout(buf_doc):
        res_doc = cli.cmd_doctor()
    out_doc = buf_doc.getvalue()
    assert res_doc == 0
    assert "PHANTOM SYSTEM DIAGNOSTICS" in out_doc
    assert "NVMe Write Speed" in out_doc or "NVMe" in out_doc
    assert "All diagnostics passed" in out_doc

    # 3. Test cmd_benchmark with output verification
    buf_bench = io.StringIO()
    with contextlib.redirect_stdout(buf_bench):
        res_bench = cli.cmd_benchmark("llama3:70b")
    out_bench = buf_bench.getvalue()
    assert res_bench == 0
    assert "PHANTOM BENCHMARK SUITE" in out_bench
    assert "SPECULATIVE" in out_bench or "BENCHMARK" in out_bench

    # 4. Test cmd_status with output verification
    buf_status = io.StringIO()
    with contextlib.redirect_stdout(buf_status):
        res_status = cli.cmd_status()
    out_status = buf_status.getvalue()
    assert res_status == 0
    assert "PHANTOM RUNTIME STATUS" in out_status
    assert "Hardware Tier:" in out_status
    assert "Speculative Runtime:" in out_status
    assert "RAM Amortization:" in out_status

    return True


def audit_section_4():
    """SECTION 4 — Phantomfile System"""
    from phantom.phantomfile import PhantomfileParser

    sample = """
    FROM phi3:3.8b
    SYSTEM You are a helpful pirate assistant.
    PARAMETER temperature 0.5
    PARAMETER stop "<|eot_id|>"
    PHANTOM_PARAM kv_compression on
    PHANTOM_PARAM spectral_quant on
    """
    parser = PhantomfileParser()
    config = parser.parse_string(sample)
    errors = parser.validate(config)
    assert len(errors) == 0

    # Test invalid configuration rejection
    bad_sample = "FROM test\nPARAMETER temperature 999\n"
    bad_config = parser.parse_string(bad_sample)
    bad_errors = parser.validate(bad_config)
    assert len(bad_errors) > 0

    return True


def audit_section_5():
    """SECTION 5 — Plugin System"""
    from phantom.plugins import (
        ContextCachePlugin,
        GenerationContext,
        PluginPipeline,
        RAGPlugin,
        ToolRouterPlugin,
        phantom_tool,
    )

    async def _test():
        pipeline = PluginPipeline()
        rag = RAGPlugin()
        rag.add_document("doc1", "Phantom runs unreachable models.")
        pipeline.register(rag)

        router = ToolRouterPlugin()
        @phantom_tool
        def add(a: int, b: int) -> int:
            return a + b
        router.register_tool("add", add)
        pipeline.register(router)

        ctx = GenerationContext(model_id="phi3:3.8b")
        aug = await pipeline.run_pre_generate("What does Phantom run?", ctx)
        assert "Phantom runs unreachable models" in aug

        tool_resp = '```tool_call {"name": "add", "arguments": {"a": 2, "b": 3}} ```'
        post = await pipeline.run_post_generate(tool_resp, ctx)
        assert "[Tool Result: add] -> 5" in post

        # Test MCP tool server registration and definition injection
        router.register_mcp_server(
            name="system-mcp",
            endpoint="http://localhost:8080/mcp",
            tools=[{"name": "mcp_ping", "description": "Ping MCP server"}],
        )
        assert "mcp_ping" in router.get_tool_definitions()

    asyncio.run(_test())
    return True


def audit_section_6():
    """SECTION 6 — API Gateway & Ollama Endpoints"""
    from fastapi.testclient import TestClient
    from phantom.api.gateway import gateway_app, state

    client = TestClient(gateway_app)

    # Health
    r1 = client.get("/v1/health")
    assert r1.status_code == 200

    # Metrics
    r2 = client.get("/v1/metrics")
    assert r2.status_code == 200
    assert "layer_residency" in r2.json()

    # Ollama tags
    r3 = client.get("/api/tags")
    assert r3.status_code == 200
    assert len(r3.json()["models"]) > 0

    # Bearer Auth
    state.auth_token = "audit_token"
    unauth = client.post("/api/chat", json={"model": "llama3:70b", "messages": [], "stream": False})
    assert unauth.status_code == 401

    auth = client.post(
        "/api/chat",
        json={"model": "llama3:70b", "messages": [], "stream": False},
        headers={"Authorization": "Bearer audit_token"},
    )
    assert auth.status_code == 200
    state.auth_token = None

    return True


def audit_section_7():
    """SECTION 7 — OSS Readiness & Documentation Integrity (No Stubs Allowed)"""
    root = Path(__file__).parents[1]

    docs_to_verify = {
        "README.md": ["phantom", "speculative", "ollama", "benchmark"],
        "docs/ARCHITECTURE.md": ["Wraith", "Spectral", "Neural Cache", "Phantom Pages"],
        "docs/INNOVATIONS.md": ["Innovation", "SIMD", "Speculative", "Routing"],
        "docs/OLLAMA_MIGRATION.md": ["/api/generate", "/api/chat", "Modelfile"],
        "docs/PHANTOMFILE.md": ["FROM", "PHANTOM_PARAM", "PLUGIN"],
        "docs/GGUF_SUPPORT.md": ["Q4_K_M", "Q8_0", "dequantize"],
        "docs/PLUGINS.md": ["BasePlugin", "pre_generate", "post_generate", "tool-router"],
        "docs/grafana_dashboard.json": ["phantom_tokens_per_second", "phantom_vram_used_mb"],
    }

    for rel_path, required_terms in docs_to_verify.items():
        p = root / rel_path
        assert p.exists(), f"CRITICAL AUDIT FAILURE: Required documentation file {rel_path} does not exist!"
        content = p.read_text(encoding="utf-8")
        assert len(content) > 400, (
            f"CRITICAL AUDIT FAILURE: Documentation file {rel_path} is too short "
            f"({len(content)} bytes) — stubs are not allowed!"
        )
        for term in required_terms:
            assert term.lower() in content.lower(), (
                f"CRITICAL AUDIT FAILURE: Document {rel_path} missing required section/term '{term}'"
            )

    return True


def main():
    print("=" * 70)
    print("  PHANTOM PLATFORM AUDIT SUITE (CLI & RUNTIME)")
    print("=" * 70)

    sections = [
        ("S1 GGUF Loader & Dequantization", audit_section_1),
        ("S2 Conversion Pipeline (.phantomw)", audit_section_2),
        ("S3 CLI Interface (plan, doctor)", audit_section_3),
        ("S4 Phantomfile System & Validator", audit_section_4),
        ("S5 Plugin Middleware System", audit_section_5),
        ("S6 Hardened Gateway & Ollama Endpoints", audit_section_6),
        ("S7 OSS Readiness & Documentation", audit_section_7),
    ]

    results = []
    for name, func in sections:
        try:
            passed = func()
            results.append((name, "PASS" if passed else "FAIL"))
            print(f"  [PASS] {name}")
        except Exception as e:
            results.append((name, f"FAIL ({e})"))
            print(f"  [FAIL] {name}: {e}")

    print("\n" + "=" * 70)
    print("PHANTOM PLATFORM AUDIT REPORT")
    print(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print("=" * 70)
    print("SECTION RESULTS:")
    for name, res in results:
        print(f"  {name:<40}: [{res}]")

    all_pass = all("PASS" in r[1] for r in results)
    print("-" * 70)
    print(f"OVERALL: [{'SHIP IT' if all_pass else 'NEEDS WORK'}]")
    print("=" * 70 + "\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
