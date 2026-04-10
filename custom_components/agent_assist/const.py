"""Constants for agent-assist integration."""

DOMAIN = "agent_assist"
DEFAULT_CONTAINER_URL = "http://localhost:8080"
CONF_URL = "url"
CONF_API_KEY = "api_key"
PLATFORMS = ["conversation"]
ATTR_CONVERSATION_ID = "conversation_id"
ATTR_LANGUAGE = "language"
WS_PATH = "/ws/conversation"
HEALTH_PATH = "/api/health"
RECONNECT_BASE_DELAY = 1.0
RECONNECT_MAX_DELAY = 30.0
