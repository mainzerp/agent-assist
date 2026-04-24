"""Constants for HA-AgentHub Home Assistant integration."""

DOMAIN = "ha_agenthub"
# Shown in HA integration picker, config entry title, and device registry.
INTEGRATION_TITLE = "HA-AgentHub"
DEFAULT_CONTAINER_URL = "http://localhost:8080"
CONF_URL = "url"
CONF_API_KEY = "api_key"
# PLATFORMS moved to __init__.py using Platform enum
ATTR_CONVERSATION_ID = "conversation_id"
ATTR_LANGUAGE = "language"
WS_PATH = "/ws/conversation"
HEALTH_PATH = "/api/health"
RECONNECT_BASE_DELAY = 1.0
RECONNECT_MAX_DELAY = 30.0
WS_HEARTBEAT_INTERVAL = 15
WS_IDLE_THRESHOLD = 60

# Native Assist plain-timer delegation (0.25.0).
# Opt-in option key in ConfigEntry.options. When True, the conversation
# entity will route eligible plain-timer requests (start/cancel only)
# directly to Home Assistant's native default conversation agent
# instead of forwarding them to the AgentHub container. Default False
# preserves existing behavior for installations that have not opted in.
CONF_NATIVE_PLAIN_TIMERS = "native_plain_timers"
DEFAULT_NATIVE_PLAIN_TIMERS = False
# Stable identifier of HA's built-in default conversation agent.
# Mirrors ``homeassistant.components.conversation.HOME_ASSISTANT_AGENT``;
# kept as a constant so we can fall back to the literal value on older
# HA cores where the symbol is not exported.
NATIVE_HA_AGENT_ID = "conversation.home_assistant"

# 0.25.1: protocol carriers for the LLM-driven native plain-timer
# classifier. The integration emits the eligibility flag/header when
# the user has explicitly opted in (and suppression is not active);
# the container responds with the directive when the classifier
# decides the utterance should be delegated to native Assist.
NATIVE_PLAIN_TIMER_ELIGIBLE_FIELD = "native_plain_timer_eligible"
NATIVE_PLAIN_TIMER_ELIGIBLE_HEADER = "X-HA-AgentHub-Native-Plain-Timer-Eligible"
NATIVE_PLAIN_TIMER_DIRECTIVE = "delegate_native_plain_timer"
