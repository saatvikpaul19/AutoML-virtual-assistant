from __future__ import annotations

import random

from modules.C_nlu.intent_detection import (
    is_stateful_intent,
    is_stateless_intent,
    is_utility_intent,
)
from modules.D_control.state_manager import get_state_manager
from modules.D_control.experiment_controller import ExperimentController
from modules.F_stateless_info.kaggle_service import (
    search_datasets,
    get_dataset_info,
    show_competitions,
    show_leaderboard,
)
from modules.F_stateless_info.kaggle_kernel_service import search_kernels
from modules.F_stateless_info.weather_service import get_weather, weather_to_text
from modules.E_ml_automl.qwen_llm import QwenAssistant


def _format_duration(seconds: int) -> str:
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    parts = []
    if hours:
        parts.append(f"{hours} hour(s)")
    if mins:
        parts.append(f"{mins} minute(s)")
    if secs or not parts:
        parts.append(f"{secs} second(s)")
    return ", ".join(parts)


class CommandRouter:
    def __init__(self):
        self._sm         = get_state_manager()
        self._controller = ExperimentController(self._sm)

    def route(self, command: dict) -> dict:
        intent = command.get("intent", "out_of_scope")

        # ── Timer intents (checked before stateful lookup) ────
        if intent in {
            "set_timer", "check_timer", "pause_timer", "resume_timer",
            "stop_timer", "restart_timer", "reset_timer",
            "add_time_to_timer", "cancel_timer",
        }:
            result = self._handle_timer(command)
            return {**result, "intent": intent, "category": "stateful"}

        if is_stateful_intent(intent):
            result = self._controller.execute(command)
            return {**result, "intent": intent, "category": "stateful"}

        if is_stateless_intent(intent):
            result = self._handle_stateless(command)
            return {**result, "intent": intent, "category": "stateless"}

        if is_utility_intent(intent):
            result = self._handle_utility(command)
            return {**result, "intent": intent, "category": "utility"}

        # Unknown intent — shouldn't normally reach here
        self._sm.append_log(f"❓ Out-of-scope: {command.get('raw_text', '')}")
        return {
            "success": False,
            "message": (
                "I'm not sure how to help with that. "
                "Try saying *'hey mycroft help'* to see what I can do."
            ),
            "intent":   "out_of_scope",
            "category": "utility",
        }

    # ── Timer ─────────────────────────────────────────────────

    def _handle_timer(self, command: dict) -> dict:
        intent = command.get("intent")
        slots  = command.get("slots", {})

        if intent == "set_timer":
            duration = int(slots.get("duration_seconds", 0) or 0)
            if duration <= 0:
                return {
                    "success": False,
                    "message": (
                        "I couldn't work out the timer duration. "
                        "Try: *'set a timer for 5 minutes'*"
                    ),
                }
            label = slots.get("label", "timer")
            self._sm.start_timer(duration, label)
            self._sm.append_log(f"⏲️ Timer: {label} ({duration}s)")
            return {"success": True, "message": f"Timer started for {_format_duration(duration)}."}

        if intent == "check_timer":
            timer = self._sm.get_timer_info()
            if not timer.get("exists"):
                return {"success": True, "message": "There is no active timer."}
            if timer["status"] == "completed":
                return {"success": True, "message": f"Your {timer.get('label','timer')} is done."}
            return {
                "success": True,
                "message": (
                    f"Your {timer.get('label','timer')} is {timer['status']} "
                    f"with {_format_duration(timer['remaining_seconds'])} remaining."
                ),
            }

        if intent == "pause_timer":
            if self._sm.pause_timer():
                timer = self._sm.get_timer_info()
                return {
                    "success": True,
                    "message": f"Paused the {timer.get('label','timer')} with {_format_duration(timer['remaining_seconds'])} remaining.",
                }
            return {"success": False, "message": "There is no running timer to pause."}

        if intent == "resume_timer":
            if self._sm.resume_timer():
                timer = self._sm.get_timer_info()
                return {"success": True, "message": f"Resumed the {timer.get('label','timer')}."}
            return {"success": False, "message": "There is no paused timer to resume."}

        if intent == "stop_timer":
            if self._sm.stop_timer():
                return {"success": True, "message": "Timer stopped."}
            return {"success": False, "message": "There is no timer to stop."}

        if intent == "restart_timer":
            if self._sm.restart_timer():
                timer = self._sm.get_timer_info()
                return {
                    "success": True,
                    "message": f"Restarted the {timer.get('label','timer')} for {_format_duration(timer['remaining_seconds'])}.",
                }
            return {"success": False, "message": "There is no timer to restart."}

        if intent == "reset_timer":
            if self._sm.reset_timer():
                timer = self._sm.get_timer_info()
                return {
                    "success": True,
                    "message": f"Reset the {timer.get('label','timer')} back to {_format_duration(timer['remaining_seconds'])}. It is now paused.",
                }
            return {"success": False, "message": "There is no timer to reset."}

        if intent == "add_time_to_timer":
            extra = int(slots.get("duration_seconds", 0) or 0)
            if extra <= 0:
                return {"success": False, "message": "Please specify how much time to add, e.g. *'add 2 minutes to timer'*."}
            if self._sm.add_time_to_timer(extra):
                timer = self._sm.get_timer_info()
                return {
                    "success": True,
                    "message": f"Added {_format_duration(extra)} to the {timer.get('label','timer')}. It now has {_format_duration(timer['remaining_seconds'])} remaining.",
                }
            return {"success": False, "message": "There is no timer to extend."}

        if intent == "cancel_timer":
            timer = self._sm.get_timer_info()
            if not timer.get("exists"):
                return {"success": True, "message": "There was no active timer to cancel."}
            self._sm.cancel_timer()
            return {"success": True, "message": f"Cancelled the {timer.get('label','timer')}."}

        return {"success": False, "message": f"No timer handler for '{intent}'."}

    # ── Stateless ─────────────────────────────────────────────

    def _handle_stateless(self, command: dict) -> dict:
        intent = command.get("intent")
        slots  = command.get("slots", {})

        if intent == "search_dataset":
            query  = slots.get("query", "")
            result = search_datasets(query)
            self._sm.set_stateless_results(result.get("results", []))
            self._sm.append_log(f"🔍 Dataset search: {query}")
            return {
                "success": True,
                "message": f"Found {len(result.get('results', []))} dataset result(s).",
                "data":    result.get("results", []),
            }

        if intent == "get_dataset_info":
            ds_q   = slots.get("dataset") or self._sm.get("dataset") or ""
            
            # Check if the query is about the currently loaded workspace dataset
            loaded_info = self._sm.get("dataset_info", {})
            if loaded_info and (not ds_q or ds_q.lower() in self._sm.get("dataset", "").lower() or ds_q.lower() in ["it", "this", "the"]):
                rows, cols = loaded_info.get("rows", "unknown"), loaded_info.get("columns", "unknown")
                msg = f"The currently loaded dataset has {rows} rows and {cols} columns."
                self._sm.append_log("ℹ️ Returned workspace dataset info")
                return {"success": True, "message": msg, "data": loaded_info}

            # Otherwise, query external Kaggle records
            result = get_dataset_info(ds_q)
            payload = (
                {
                    "top_result":     result.get("top_result", {}),
                    "files":          result.get("files", []),
                    "search_results": result.get("search_results", []),
                }
                if result.get("success") else {}
            )
            self._sm.set_stateless_results(payload)
            self._sm.append_log(f"ℹ️ Dataset info: {ds_q}")
            
            msg = "Dataset info retrieved."
            if result.get("success"):
                top_name = result.get("top_result", {}).get("title", "")
                if top_name:
                    msg = f"Found dataset info for '{top_name}'."
            else:
                msg = result.get("error", "Failed to retrieve dataset info.")
                
            return {
                "success": result.get("success", False),
                "message": msg,
                "data":    payload,
            }

        if intent == "show_competition":
            result = show_competitions()
            self._sm.set_stateless_results(result.get("results", []))
            self._sm.append_log("🏆 Competitions requested")
            return {
                "success": True,
                "message": f"Retrieved {len(result.get('results', []))} competition result(s).",
                "data":    result.get("results", []),
            }

        if intent == "show_leaderboard":
            competition = slots.get("query", "titanic")
            result      = show_leaderboard(competition)
            self._sm.set_stateless_results(result.get("results", []))
            self._sm.append_log(f"📊 Leaderboard: {competition}")
            return {
                "success": True,
                "message": f"Leaderboard retrieved for {competition}.",
                "data":    result.get("results", []),
            }

        if intent == "search_code":
            query  = slots.get("query", "")
            result = search_kernels(query)
            self._sm.set_stateless_results(result.get("results", []))
            self._sm.append_log(f"📘 Code search: {query}")
            msg = (
                f"Found {len(result.get('results', []))} notebook(s)."
                if result.get("success")
                else f"No Kaggle notebooks found for '{query}'."
            )
            return {"success": True, "message": msg, "data": result.get("results", [])}

        if intent == "suggest_model":
            profile    = self._sm.get("dataset_info", {}).get("profile", {})
            is_tabular = profile.get("modality", "") == "tabular"
            msg = QwenAssistant.suggest_model(profile, is_tabular)
            self._sm.append_log("🤖 Model suggestion requested")
            return {"success": True, "message": msg, "data": []}

        if intent == "suggest_hyperparameters":
            profile    = self._sm.get("dataset_info", {}).get("profile", {})
            model_name = self._sm.get("model", "Unknown")
            msg = QwenAssistant.suggest_hyperparameters(model_name, profile)
            self._sm.append_log("🤖 Hyperparameter suggestion requested")
            return {"success": True, "message": msg, "data": []}

        if intent == "get_weather":
            city = slots.get("city", "")
            if not city:
                return {
                    "success": False,
                    "message": "Which city's weather would you like? For example: *'weather in Ottawa today'*",
                    "data":    {},
                }
            day = slots.get("day", "today")
            try:
                result = get_weather(city, day)
            except Exception as e:
                return {"success": False, "message": f"Weather lookup failed: {e}", "data": {}}

            if not result.get("success"):
                return {"success": False, "message": result.get("error", "Weather lookup failed."), "data": result}

            self._sm.set_weather_result(result)
            self._sm.set_stateless_results(result)
            self._sm.append_log(f"🌦️ Weather: {city} ({day})")
            return {"success": True, "message": weather_to_text(result), "data": result}

        return {"success": False, "message": f"No handler for stateless intent '{intent}'.", "data": []}

    # ── Utility ───────────────────────────────────────────────

    def _handle_utility(self, command: dict) -> dict:
        intent = command.get("intent")
        slots  = command.get("slots", {})

        if intent == "help":
            help_text = (
                "Here are things I can help with:\n\n"
                "**🗄️ Data**\n"
                "• *hey mycroft load iris dataset*\n"
                "• *hey mycroft search dataset fraud detection*\n"
                "• *hey mycroft clean dataset*\n\n"
                "**🤖 Model & Training**\n"
                "• *hey mycroft use xgboost*\n"
                "• *hey mycroft set learning rate to 0.001*\n"
                "• *hey mycroft train for 20 epochs*\n"
                "• *hey mycroft start training*\n"
                "• *hey mycroft show results*\n\n"
                "**💻 Code**\n"
                "• *hey mycroft load corresponding code*\n"
                "• *hey mycroft run code*\n\n"
                "**⏲️ Timer**\n"
                "• *hey mycroft set a timer for 5 minutes*\n"
                "• *hey mycroft check timer*\n\n"
                "**🌦️ Weather**\n"
                "• *hey mycroft what is the weather in Ottawa today*\n"
            )
            return {"success": True, "message": help_text}

        if intent == "repeat":
            last = self._sm.get("assistant_response", "")
            return {"success": True, "message": last or "Nothing to repeat yet."}

        if intent == "greetings":
            responses = [
                "Hello! I'm Mycroft, your machine learning assistant. How can I help you today?",
                "Hi there! Ready to build some models?",
                "Greetings! What dataset are we working with today?",
                "Good to see you! You can also ask me for weather updates or timer controls.",
                "Hey! I'm Mycroft — here to help you train models, check weather, and manage timers.",
            ]
            self._sm.append_log("👋 Greeted the user")
            return {"success": True, "message": random.choice(responses)}

        if intent == "farewell":
            responses = [
                "Goodbye! Let me know when you're ready to train more models.",
                "See you later! Your workspace will be waiting.",
                "Farewell! Have a great rest of your day.",
                "Bye! Let me know if you need more AI support later.",
                "Good night! Mycroft signing off.",
            ]
            self._sm.append_log("👋 Said goodbye")
            return {"success": True, "message": random.choice(responses)}

        # NEW: thanks
        if intent == "thanks":
            responses = [
                "You're welcome! Is there anything else I can help with?",
                "Happy to help! Let me know if you need anything else.",
                "Of course! Just say *'hey mycroft'* whenever you need me.",
                "Glad I could assist! Anything else on your mind?",
                "Anytime! I'm here whenever you need help with your ML workflow.",
            ]
            self._sm.append_log("🤝 User said thanks")
            return {"success": True, "message": random.choice(responses)}

        # NEW: sleep_va
        if intent == "sleep_va":
            self._sm.set_asr_state("sleep")
            self._sm.set_wake_detected(False)
            self._sm.append_log("💤 VA put to sleep by user command")
            return {
                "success": True,
                "message": "Going to sleep. 💤 Say *'hey mycroft'* to wake me up again.",
            }

        if intent == "out_of_scope":
            original = slots.get("original_request", "")
            if original:
                msg = (
                    f"Sorry, I'm not able to *{original}* — that's outside my area of expertise. "
                    "I specialise in AutoML tasks like loading datasets, training models, and checking the weather. "
                    "Say *'hey mycroft help'* to see what I can do."
                )
            else:
                msg = (
                    "I'm not able to help with that — it's outside my area of expertise. "
                    "I specialise in AutoML tasks. "
                    "Say *'hey mycroft help'* to see what I can do."
                )
            self._sm.append_log("🚫 Out-of-scope command received")
            return {"success": False, "message": msg}

        return {"success": False, "message": f"No handler for utility intent '{intent}'."}


# ── Singleton ─────────────────────────────────────────────────

_router: CommandRouter | None = None


def route_command(command: dict) -> dict:
    global _router
    if _router is None:
        _router = CommandRouter()
    return _router.route(command)