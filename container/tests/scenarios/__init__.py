"""Real-scenario end-to-end test framework.

This package wires the production OrchestratorAgent + Dispatcher + agent
registry against deterministic stubs (LLM, HA REST, embeddings) and runs
YAML-defined scenarios end-to-end.

See docs/SubAgent/real_scenario_tests_plan.md for the design.
"""
