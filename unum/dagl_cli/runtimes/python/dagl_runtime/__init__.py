"""DAGL Runtime Layer - Orchestration engine for DAGL workflows.

This Lambda Layer wraps user functions with the DAGL orchestration engine,
enabling direct function-to-function invocation without a central coordinator.

Usage:
  - Attach this Layer to any Lambda function
  - Set Handler to: dagl_runtime.main.handler
  - Set env var DAGL_USER_HANDLER to the original handler (e.g., "app.handler")
  - Set env var DAGL_CONFIG to the JSON unum_config for this function
  - Set env var DAGL_FUNCTION_MAP to JSON mapping of function names → ARNs
"""

__version__ = "0.1.0"
