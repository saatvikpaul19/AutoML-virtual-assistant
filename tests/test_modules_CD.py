"""
Tests for Modules C and D — run with:  python -m pytest tests/test_modules_CD.py -v
All tests are pure-Python (no microphone / GPU required).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import pytest


# ═══════════════════════════════════════════════════════════
# MODULE 6 — Intent Detection
# ═══════════════════════════════════════════════════════════

from modules.C_nlu.intent_detection import (
    detect_intent,
    normalize_text,
    is_stateful_intent,
    is_stateless_intent,
    is_utility_intent,
    get_supported_intents,
    get_intent_category,
)


class TestNormalizeText:

    def test_lowercase(self):
        assert normalize_text("HELLO WORLD") == "hello world"

    def test_strip_punctuation(self):
        assert normalize_text("hello!!!") == "hello"

    def test_collapse_whitespace(self):
        assert normalize_text("  hello   world  ") == "hello world"

    def test_asr_correction_xgboost(self):
        result = normalize_text("use x g boost")
        assert "xgboost" in result

    def test_asr_correction_learning_rate(self):
        result = normalize_text("set learning great to 0.01")
        assert "learning rate" in result

    def test_asr_correction_batch_size(self):
        result = normalize_text("set batch eyes to 32")
        assert "batch size" in result

    def test_asr_correction_epochs(self):
        result = normalize_text("set epics to 50")
        assert "epochs" in result

    def test_preserves_numbers(self):
        assert "0.01" in normalize_text("set to 0.01")

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_none_returns_empty(self):
        assert normalize_text(None) == ""


class TestDetectIntent:

    # ── Load Dataset (explicit + paraphrased) ────────────
    def test_load_dataset_explicit(self):
        assert detect_intent("load the titanic dataset") == "load_dataset"

    def test_load_dataset_short(self):
        assert detect_intent("load titanic") == "load_dataset"

    def test_load_dataset_paraphrase(self):
        assert detect_intent("can you load data for me") == "load_dataset"

    def test_load_dataset_import(self):
        assert detect_intent("import the iris dataset") == "load_dataset"

    def test_load_dataset_open(self):
        assert detect_intent("open the mnist data") == "load_dataset"

    def test_load_dataset_with_the(self):
        assert detect_intent("load the wine") == "load_dataset"

    def test_load_dataset_flexible_phrasing(self):
        """Regex fallback catches 'load ... dataset' with words between."""
        assert detect_intent("load my custom dataset please") == "load_dataset"

    # ── Select Model ─────────────────────────────────────
    def test_select_model(self):
        assert detect_intent("use xgboost") == "select_model"

    def test_select_model_paraphrase(self):
        assert detect_intent("change model to random forest") == "select_model"

    def test_select_model_choose(self):
        assert detect_intent("choose logistic regression") == "select_model"

    def test_select_model_switch(self):
        assert detect_intent("switch model to xgboost") == "select_model"

    def test_select_model_regex_fallback(self):
        """Regex catches 'pick the random forest model'."""
        assert detect_intent("pick the random forest model") == "select_model"

    # ── Hyperparameters ──────────────────────────────────
    def test_set_learning_rate(self):
        assert detect_intent("set learning rate to 0.01") == "set_learning_rate"

    def test_set_learning_rate_paraphrase(self):
        assert detect_intent("change the learning rate") == "set_learning_rate"

    def test_set_learning_rate_asr(self):
        """ASR may say 'learning great' — correction should fix it."""
        assert detect_intent("set learning great to 0.05") == "set_learning_rate"

    def test_set_batch_size(self):
        assert detect_intent("set batch size to 64") == "set_batch_size"

    def test_set_batch_size_asr(self):
        """ASR may say 'batch eyes'."""
        assert detect_intent("set batch eyes to 32") == "set_batch_size"

    def test_set_epochs(self):
        assert detect_intent("set epochs to 50") == "set_epochs"

    def test_set_epochs_paraphrase(self):
        assert detect_intent("train for 100 epochs") == "set_epochs"

    def test_set_epochs_asr(self):
        assert detect_intent("set epics to 30") == "set_epochs"

    # ── Training Control ─────────────────────────────────
    def test_start_training(self):
        assert detect_intent("start training") == "start_training"

    def test_start_training_paraphrase(self):
        assert detect_intent("begin the training") == "start_training"

    def test_start_training_run(self):
        assert detect_intent("run training") == "start_training"

    def test_start_training_regex(self):
        assert detect_intent("launch the training process") == "start_training"

    def test_pause_training(self):
        assert detect_intent("pause training") == "pause_training"

    def test_pause_training_hold(self):
        assert detect_intent("hold the training") == "pause_training"

    def test_stop_training(self):
        assert detect_intent("stop training") == "stop_training"

    def test_stop_training_cancel(self):
        assert detect_intent("cancel training") == "stop_training"

    def test_stop_training_abort(self):
        assert detect_intent("abort the training") == "stop_training"

    def test_resume_training(self):
        assert detect_intent("resume training") == "resume_training"

    def test_resume_training_continue(self):
        assert detect_intent("continue the training") == "resume_training"

    def test_resume_training_unpause(self):
        assert detect_intent("unpause training") == "resume_training"

    # ── Status / Metrics ─────────────────────────────────
    def test_show_status(self):
        assert detect_intent("show status") == "show_status"

    def test_show_status_check(self):
        assert detect_intent("check status") == "show_status"

    def test_show_accuracy(self):
        assert detect_intent("what is the accuracy") == "show_accuracy"

    def test_show_accuracy_how(self):
        assert detect_intent("how accurate is the model") == "show_accuracy"

    def test_show_loss_curve(self):
        assert detect_intent("show loss curve") == "show_loss_curve"

    def test_show_loss_history(self):
        assert detect_intent("show loss history") == "show_loss_curve"

    def test_show_loss_plot(self):
        assert detect_intent("plot loss graph") == "show_loss_curve"

    # ── Stateless ────────────────────────────────────────
    def test_search_dataset(self):
        assert detect_intent("search dataset on kaggle") == "search_dataset"

    def test_search_dataset_find(self):
        assert detect_intent("find dataset for nlp") == "search_dataset"

    def test_show_competition(self):
        assert detect_intent("show kaggle competition") == "show_competition"

    def test_show_leaderboard(self):
        assert detect_intent("show leaderboard") == "show_leaderboard"

    def test_get_dataset_info(self):
        assert detect_intent("describe dataset") == "get_dataset_info"

    def test_get_dataset_info_tell(self):
        assert detect_intent("tell me about the dataset") == "get_dataset_info"

    # ── Utility ──────────────────────────────────────────
    def test_help(self):
        assert detect_intent("help") == "help"

    def test_help_what_can_you_do(self):
        assert detect_intent("what can you do") == "help"

    def test_repeat(self):
        assert detect_intent("say that again") == "repeat"

    def test_repeat_pardon(self):
        assert detect_intent("pardon") == "repeat"

    # ── Unknown ──────────────────────────────────────────
    def test_unknown_intent(self):
        assert detect_intent("make me a sandwich") == "unknown_intent"

    def test_empty_string(self):
        assert detect_intent("") == "unknown_intent"

    def test_gibberish(self):
        assert detect_intent("asdfghjkl qwerty") == "unknown_intent"


class TestIntentCategories:

    def test_stateful(self):
        assert is_stateful_intent("start_training") is True
        assert is_stateful_intent("load_dataset") is True
        assert is_stateful_intent("show_status") is True

    def test_stateless(self):
        assert is_stateless_intent("search_dataset") is True
        assert is_stateless_intent("show_competition") is True

    def test_utility(self):
        assert is_utility_intent("help") is True
        assert is_utility_intent("unknown_intent") is True

    def test_category_label(self):
        assert get_intent_category("start_training") == "stateful"
        assert get_intent_category("search_dataset") == "stateless"
        assert get_intent_category("help") == "utility"
        assert get_intent_category("nonexistent") == "unknown"

    def test_supported_intents_list(self):
        intents = get_supported_intents()
        assert "start_training" in intents
        assert "help" in intents
        assert "unknown_intent" in intents
        assert len(intents) >= 19


# ═══════════════════════════════════════════════════════════
# MODULE 7 — Slot Filling
# ═══════════════════════════════════════════════════════════

from modules.C_nlu.slot_filling import extract_slots, get_slot_schema


class TestSlotFilling:

    # ── Learning Rate ────────────────────────────────────
    def test_extract_learning_rate_to(self):
        result = extract_slots("set learning rate to 0.01", "set_learning_rate")
        assert result["slots"]["learning_rate"] == 0.01
        assert result["missing_slots"] == []

    def test_extract_learning_rate_bare(self):
        result = extract_slots("learning rate 0.001", "set_learning_rate")
        assert result["slots"]["learning_rate"] == 0.001

    def test_missing_learning_rate(self):
        result = extract_slots("set learning rate", "set_learning_rate")
        assert "learning_rate" in result["missing_slots"]

    def test_learning_rate_zero_invalid(self):
        result = extract_slots("set learning rate to 0", "set_learning_rate")
        assert "learning_rate" in result["invalid_slots"]

    def test_learning_rate_too_large(self):
        result = extract_slots("set learning rate to 15", "set_learning_rate")
        assert "learning_rate" in result["invalid_slots"]

    # ── Batch Size ──────────────────────────────────────
    def test_extract_batch_size(self):
        result = extract_slots("set batch size to 64", "set_batch_size")
        assert result["slots"]["batch_size"] == 64

    def test_missing_batch_size(self):
        result = extract_slots("set batch size", "set_batch_size")
        assert "batch_size" in result["missing_slots"]

    def test_batch_size_too_large(self):
        result = extract_slots("set batch size to 9999", "set_batch_size")
        assert "batch_size" in result["invalid_slots"]

    # ── Epochs ───────────────────────────────────────────
    def test_extract_epochs(self):
        result = extract_slots("set epochs to 50", "set_epochs")
        assert result["slots"]["epochs"] == 50

    def test_extract_epochs_for(self):
        result = extract_slots("train for 100 epochs", "set_epochs")
        assert result["slots"]["epochs"] == 100

    def test_epochs_too_large(self):
        result = extract_slots("set epochs to 99999", "set_epochs")
        assert "epochs" in result["invalid_slots"]

    # ── Model ────────────────────────────────────────────
    def test_extract_model_xgboost(self):
        result = extract_slots("use xgboost", "select_model")
        assert result["slots"]["model"] == "xgboost"

    def test_extract_model_random_forest(self):
        result = extract_slots("use random forest", "select_model")
        assert result["slots"]["model"] == "random_forest"

    def test_extract_model_logistic(self):
        result = extract_slots("use logistic regression", "select_model")
        assert result["slots"]["model"] == "logistic_regression"

    def test_extract_model_synonym_rf(self):
        result = extract_slots("use rf", "select_model")
        assert result["slots"]["model"] == "random_forest"

    def test_extract_model_synonym_xgb(self):
        result = extract_slots("use xgb", "select_model")
        assert result["slots"]["model"] == "xgboost"

    def test_missing_model(self):
        result = extract_slots("select model", "select_model")
        assert "model" in result["missing_slots"]

    # ── Dataset ──────────────────────────────────────────
    def test_extract_dataset_titanic(self):
        result = extract_slots("load the titanic dataset", "load_dataset")
        assert result["slots"]["dataset"] == "titanic"

    def test_extract_dataset_iris(self):
        result = extract_slots("load iris", "load_dataset")
        assert result["slots"]["dataset"] == "iris"

    def test_extract_dataset_cifar(self):
        result = extract_slots("load cifar 10", "load_dataset")
        assert result["slots"]["dataset"] == "cifar10"

    def test_extract_dataset_breast_cancer(self):
        result = extract_slots("load breast cancer dataset", "load_dataset")
        assert result["slots"]["dataset"] == "breast_cancer"

    # ── No-slot intents ──────────────────────────────────
    def test_start_training_no_slots(self):
        result = extract_slots("start training", "start_training")
        assert result["slots"] == {}
        assert result["missing_slots"] == []

    def test_help_no_slots(self):
        result = extract_slots("help me", "help")
        assert result["slots"] == {}
        assert result["missing_slots"] == []

    # ── Schema lookup ────────────────────────────────────
    def test_schema_exists(self):
        schema = get_slot_schema("set_learning_rate")
        assert schema is not None
        assert len(schema) == 1
        assert schema[0]["name"] == "learning_rate"

    def test_schema_no_slots(self):
        schema = get_slot_schema("start_training")
        assert schema == []

    def test_schema_unknown(self):
        schema = get_slot_schema("nonexistent_intent")
        assert schema is None


# ═══════════════════════════════════════════════════════════
# MODULE C — NLU Pipeline (end-to-end)
# ═══════════════════════════════════════════════════════════

from modules.C_nlu.nlu_pipeline import understand


class TestNLUPipeline:

    def test_full_pipeline_learning_rate(self):
        result = understand("set learning rate to 0.01")
        assert result["intent"] == "set_learning_rate"
        assert result["slots"]["learning_rate"] == 0.01
        assert result["intent_category"] == "stateful"
        assert result["missing_slots"] == []

    def test_full_pipeline_load_dataset(self):
        result = understand("load the titanic dataset")
        assert result["intent"] == "load_dataset"
        assert result["slots"]["dataset"] == "titanic"

    def test_full_pipeline_start_training(self):
        result = understand("start training")
        assert result["intent"] == "start_training"
        assert result["slots"] == {}

    def test_full_pipeline_unknown(self):
        result = understand("fly me to the moon")
        assert result["intent"] == "unknown_intent"
        assert result["intent_category"] == "utility"

    def test_full_pipeline_batch_size(self):
        result = understand("set batch size to 32")
        assert result["intent"] == "set_batch_size"
        assert result["slots"]["batch_size"] == 32

    def test_full_pipeline_select_model(self):
        result = understand("use xgboost")
        assert result["intent"] == "select_model"
        assert result["slots"]["model"] == "xgboost"

    def test_full_pipeline_help(self):
        result = understand("help")
        assert result["intent"] == "help"
        assert result["intent_category"] == "utility"

    def test_full_pipeline_epochs(self):
        result = understand("set epochs to 100")
        assert result["intent"] == "set_epochs"
        assert result["slots"]["epochs"] == 100

    def test_full_pipeline_show_status(self):
        result = understand("show status")
        assert result["intent"] == "show_status"

    def test_full_pipeline_preserves_raw_text(self):
        result = understand("SET LEARNING RATE TO 0.05")
        assert result["raw_text"] == "SET LEARNING RATE TO 0.05"
        assert result["normalised_text"] == "set learning rate to 0.05"

    def test_full_pipeline_asr_correction(self):
        """ASR says 'x g boost' — should still detect and extract."""
        result = understand("use x g boost")
        assert result["intent"] == "select_model"
        assert result["slots"]["model"] == "xgboost"


# ═══════════════════════════════════════════════════════════
# MODULE 9 — State Manager
# ═══════════════════════════════════════════════════════════

from modules.D_control.state_manager import StateManager


class TestStateManager:

    def setup_method(self):
        self.sm = StateManager()

    def test_initial_state(self):
        state = self.sm.get_state()
        assert state["dataset"] is None
        assert state["model"] is None
        assert state["training_status"] == "idle"
        assert state["epoch_current"] == 0

    def test_set_dataset(self):
        self.sm.set_dataset("titanic")
        assert self.sm.get("dataset") == "titanic"

    def test_set_model(self):
        self.sm.set_model("xgboost")
        assert self.sm.get("model") == "xgboost"

    def test_set_learning_rate(self):
        self.sm.set_learning_rate(0.05)
        assert self.sm.get("learning_rate") == 0.05

    def test_set_batch_size(self):
        self.sm.set_batch_size(128)
        assert self.sm.get("batch_size") == 128

    def test_set_epochs(self):
        self.sm.set_epochs(100)
        assert self.sm.get("epochs_total") == 100

    def test_append_loss(self):
        self.sm.append_loss(1.5)
        self.sm.append_loss(1.2)
        assert self.sm.get("loss_history") == [1.5, 1.2]

    def test_append_accuracy(self):
        self.sm.append_accuracy(0.75)
        self.sm.append_accuracy(0.82)
        assert self.sm.get("accuracy_history") == [0.75, 0.82]

    def test_event_log_is_timestamped(self):
        self.sm.append_log("Test event")
        log = self.sm.get_event_log()
        assert len(log) == 1
        assert "Test event" in log[0]
        assert "[" in log[0]  # timestamp bracket

    def test_event_log_last_n(self):
        for i in range(10):
            self.sm.append_log(f"Event {i}")
        last_3 = self.sm.get_event_log(last_n=3)
        assert len(last_3) == 3

    def test_reset_experiment(self):
        self.sm.set_dataset("titanic")
        self.sm.set_model("xgboost")
        self.sm.append_loss(1.5)
        self.sm.set_verified(True)
        self.sm.reset_experiment()
        assert self.sm.get("dataset") is None
        assert self.sm.get("model") is None
        assert self.sm.get("loss_history") == []
        # verified should be preserved
        assert self.sm.get("verified") is True

    def test_reset_all(self):
        self.sm.set_dataset("titanic")
        self.sm.append_log("test")
        self.sm.set_verified(True)
        self.sm.reset_all()
        assert self.sm.get("dataset") is None
        assert self.sm.get_event_log() == []
        assert self.sm.get("verified") is False  # full reset

    def test_ui_state(self):
        self.sm.set_dataset("iris")
        self.sm.set_model("xgboost")
        ui = self.sm.get_ui_state()
        assert ui["dataset"] == "iris"
        assert ui["model"] == "xgboost"
        assert "n_events" in ui
        assert "loss_latest" in ui

    def test_reset_metrics(self):
        self.sm.append_loss(1.0)
        self.sm.append_accuracy(0.5)
        self.sm.set_epoch_current(5)
        self.sm.reset_metrics()
        assert self.sm.get("loss_history") == []
        assert self.sm.get("accuracy_history") == []
        assert self.sm.get("epoch_current") == 0

    def test_deep_copy_isolation(self):
        """Modifying returned state should not affect internal state."""
        state = self.sm.get_state()
        state["dataset"] = "hacked"
        assert self.sm.get("dataset") is None


# ═══════════════════════════════════════════════════════════
# MODULE 10 — Experiment Controller
# ═══════════════════════════════════════════════════════════

from modules.D_control.experiment_controller import ExperimentController


class TestExperimentController:

    def setup_method(self):
        self.sm = StateManager()
        self.ctrl = ExperimentController(self.sm)

    def _cmd(self, intent, **slots):
        """Helper to build a command dict."""
        return {
            "intent": intent,
            "slots": slots,
            "missing_slots": [],
            "invalid_slots": {},
        }

    def test_load_dataset(self):
        result = self.ctrl.execute(self._cmd("load_dataset", dataset="titanic"))
        assert result["success"] is True
        assert self.sm.get("dataset") == "titanic"

    def test_select_model_supported(self):
        result = self.ctrl.execute(self._cmd("select_model", model="xgboost"))
        assert result["success"] is True
        assert self.sm.get("model") == "xgboost"

    def test_select_model_unsupported(self):
        result = self.ctrl.execute(self._cmd("select_model", model="deep_neural_net"))
        assert result["success"] is False
        assert "not supported" in result["message"].lower()

    def test_set_learning_rate(self):
        result = self.ctrl.execute(self._cmd("set_learning_rate", learning_rate=0.05))
        assert result["success"] is True
        assert self.sm.get("learning_rate") == 0.05

    def test_set_batch_size(self):
        result = self.ctrl.execute(self._cmd("set_batch_size", batch_size=128))
        assert result["success"] is True
        assert self.sm.get("batch_size") == 128

    def test_set_epochs(self):
        result = self.ctrl.execute(self._cmd("set_epochs", epochs=50))
        assert result["success"] is True
        assert self.sm.get("epochs_total") == 50

    def test_start_training_no_dataset(self):
        self.sm.set_model("xgboost")
        result = self.ctrl.execute(self._cmd("start_training"))
        assert result["success"] is False
        assert "dataset" in result["message"].lower()

    def test_start_training_no_model(self):
        self.sm.set_dataset("titanic")
        result = self.ctrl.execute(self._cmd("start_training"))
        assert result["success"] is False
        assert "model" in result["message"].lower()

    def test_start_training_success(self):
        self.sm.set_dataset("titanic")
        self.sm.set_model("xgboost")
        self.sm.set_epochs(3)  # very short
        result = self.ctrl.execute(self._cmd("start_training"))
        assert result["success"] is True
        assert self.sm.get("training_status") == "training"
        # Wait for simulation to complete
        time.sleep(4)
        assert self.sm.get("training_status") == "completed"
        assert len(self.sm.get("loss_history")) == 3
        assert len(self.sm.get("accuracy_history")) == 3
        # Verify metrics are realistic
        for loss in self.sm.get("loss_history"):
            assert 0 < loss < 5
        for acc in self.sm.get("accuracy_history"):
            assert 0 < acc <= 1.0

    def test_pause_and_resume_training(self):
        self.sm.set_dataset("titanic")
        self.sm.set_model("xgboost")
        self.sm.set_epochs(20)
        self.ctrl.execute(self._cmd("start_training"))
        time.sleep(1.0)

        # Pause
        result = self.ctrl.execute(self._cmd("pause_training"))
        assert result["success"] is True
        assert self.sm.get("training_status") == "paused"

        # Record epoch right after pause, then wait — should NOT advance
        time.sleep(0.3)  # let pending epoch finish
        epoch_snapshot_1 = self.sm.get("epoch_current")
        time.sleep(1.0)
        epoch_snapshot_2 = self.sm.get("epoch_current")
        assert epoch_snapshot_2 == epoch_snapshot_1, (
            f"Epoch advanced while paused: {epoch_snapshot_1} → {epoch_snapshot_2}"
        )

        # Resume
        result = self.ctrl.execute(self._cmd("resume_training"))
        assert result["success"] is True
        assert self.sm.get("training_status") == "training"

        # Clean up
        self.ctrl.execute(self._cmd("stop_training"))

    def test_pause_when_idle(self):
        result = self.ctrl.execute(self._cmd("pause_training"))
        assert result["success"] is False

    def test_stop_training(self):
        self.sm.set_dataset("titanic")
        self.sm.set_model("xgboost")
        self.sm.set_epochs(20)
        self.ctrl.execute(self._cmd("start_training"))
        time.sleep(0.8)
        result = self.ctrl.execute(self._cmd("stop_training"))
        assert result["success"] is True
        assert self.sm.get("training_status") == "stopped"

    def test_stop_when_idle(self):
        result = self.ctrl.execute(self._cmd("stop_training"))
        assert result["success"] is False

    def test_resume_when_not_paused(self):
        result = self.ctrl.execute(self._cmd("resume_training"))
        assert result["success"] is False

    def test_show_status(self):
        self.sm.set_dataset("iris")
        self.sm.set_model("random_forest")
        result = self.ctrl.execute(self._cmd("show_status"))
        assert result["success"] is True
        assert "iris" in result["message"]
        assert "random_forest" in result["message"]

    def test_show_accuracy_no_data(self):
        result = self.ctrl.execute(self._cmd("show_accuracy"))
        assert result["success"] is True
        assert "no accuracy" in result["message"].lower()

    def test_show_loss_no_data(self):
        result = self.ctrl.execute(self._cmd("show_loss_curve"))
        assert result["success"] is True
        assert "no loss" in result["message"].lower()

    def test_missing_slots_rejected(self):
        result = self.ctrl.execute({
            "intent": "set_learning_rate",
            "slots": {},
            "missing_slots": ["learning_rate"],
            "invalid_slots": {},
        })
        assert result["success"] is False
        assert "missing" in result["message"].lower()

    def test_invalid_slots_rejected(self):
        result = self.ctrl.execute({
            "intent": "set_learning_rate",
            "slots": {},
            "missing_slots": [],
            "invalid_slots": {"learning_rate": "Not a number."},
        })
        assert result["success"] is False
        assert "invalid" in result["message"].lower()

    def test_cannot_load_dataset_during_training(self):
        self.sm.set_dataset("titanic")
        self.sm.set_model("xgboost")
        self.sm.set_epochs(20)
        self.ctrl.execute(self._cmd("start_training"))
        time.sleep(0.5)
        result = self.ctrl.execute(self._cmd("load_dataset", dataset="iris"))
        assert result["success"] is False
        assert "training" in result["message"].lower()
        self.ctrl.execute(self._cmd("stop_training"))

    def test_cannot_change_model_during_training(self):
        self.sm.set_dataset("titanic")
        self.sm.set_model("xgboost")
        self.sm.set_epochs(20)
        self.ctrl.execute(self._cmd("start_training"))
        time.sleep(0.5)
        result = self.ctrl.execute(self._cmd("select_model", model="random_forest"))
        assert result["success"] is False
        self.ctrl.execute(self._cmd("stop_training"))


# ═══════════════════════════════════════════════════════════
# MODULE 8 — Command Router
# ═══════════════════════════════════════════════════════════

from modules.D_control.command_router import CommandRouter


class TestCommandRouter:

    def setup_method(self):
        self.router = CommandRouter()
        self.router._sm.reset_all()

    def test_route_stateful_command(self):
        result = self.router.route({
            "intent": "load_dataset",
            "slots": {"dataset": "titanic"},
            "missing_slots": [],
            "invalid_slots": {},
        })
        assert result["success"] is True
        assert result["category"] == "stateful"

    def test_route_stateless_command(self):
        result = self.router.route({
            "intent": "search_dataset",
            "slots": {"query": "nlp"},
            "missing_slots": [],
            "invalid_slots": {},
        })
        assert result["success"] is True
        assert result["category"] == "stateless"

    def test_route_utility_help(self):
        result = self.router.route({
            "intent": "help",
            "slots": {},
            "missing_slots": [],
            "invalid_slots": {},
        })
        assert result["success"] is True
        assert result["category"] == "utility"
        assert "available commands" in result["message"].lower()

    def test_route_utility_repeat(self):
        self.router._sm.set_assistant_response("Previous response text")
        result = self.router.route({
            "intent": "repeat",
            "slots": {},
            "missing_slots": [],
            "invalid_slots": {},
        })
        assert result["success"] is True
        assert "previous response text" in result["message"].lower()

    def test_route_unknown_command(self):
        result = self.router.route({
            "intent": "fly_to_moon",
            "slots": {},
            "missing_slots": [],
            "invalid_slots": {},
        })
        assert result["success"] is False
        assert result["category"] == "error"

    def test_full_pipeline_via_router(self):
        """End-to-end: NLU → Router → Result."""
        command = understand("load the titanic dataset")
        result = self.router.route(command)
        assert result["success"] is True
        assert result["intent"] == "load_dataset"
        assert self.router._sm.get("dataset") == "titanic"

    def test_full_pipeline_via_router_model(self):
        """End-to-end: NLU → Router → Result for model selection."""
        command = understand("use random forest")
        result = self.router.route(command)
        assert result["success"] is True
        assert result["intent"] == "select_model"
        assert self.router._sm.get("model") == "random_forest"

    def test_full_pipeline_via_router_lr(self):
        """End-to-end: NLU → Router → Result for learning rate."""
        command = understand("set learning rate to 0.05")
        result = self.router.route(command)
        assert result["success"] is True
        assert self.router._sm.get("learning_rate") == 0.05
