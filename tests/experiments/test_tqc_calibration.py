from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oracle_composition.experiments.fixed_reference import (
    ExperimentContractError,
    sha256_file,
)
from oracle_composition.experiments.tqc_calibration import (
    MODEL_DISPOSITION,
    REQUIRED_TRAINING_SIGNALS,
    TQCCalibrationDesign,
    _assert_finite_parameters,
    _make_finite_resource_callback,
    fault_replay_buffer_pages,
    inspect_tqc_model,
    inspect_tqc_runtime,
    load_calibration_design,
    make_tqc_model,
    make_tqc_vector_env,
    replay_buffer_allocation_bytes,
    run_tqc_calibration,
)
from oracle_composition.experiments.tqc_calibration import (
    main as tqc_calibration_main,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DESIGN_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "bootstrap_tqc_humanoid"
    / "configs"
    / "tqc_resource_calibration_v0.study.json"
)


def _write_mutated_design(tmp_path: Path, mutate: object) -> Path:
    payload = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "design.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _fake_runtime() -> dict[str, object]:
    payload: dict[str, object] = {
        "runtime_kind": "bounded_test_double/v1",
        "environment_id": "Humanoid-v5",
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    payload["runtime_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


class _FakeVectorEnvironment:
    def __init__(self, *, close_error: bool = False) -> None:
        self.closed = False
        self.close_error = close_error

    def close(self) -> None:
        self.closed = True
        if self.close_error:
            raise RuntimeError("forced close failure")


class _FakeReplayBuffer:
    def __init__(self) -> None:
        self.pos = 0
        self.full = False
        self.observations = np.zeros((1, 1), dtype=np.float64)
        self.next_observations = np.zeros((1, 1), dtype=np.float64)
        self.actions = np.zeros((1, 1), dtype=np.float32)
        self.rewards = np.zeros((1, 1), dtype=np.float32)
        self.dones = np.zeros((1, 1), dtype=np.float32)
        self.timeouts = np.zeros((1, 1), dtype=np.float32)


class _FakeModel:
    def __init__(
        self,
        design: TQCCalibrationDesign,
        *,
        final_timestep_offset: int = 0,
        nonfinite_parameter: bool = False,
        nonfinite_training_signal: bool = False,
        final_update_offset: int = 0,
        physical_action: float = 0.0,
        buffer_action: float = 0.0,
    ) -> None:
        import torch

        self.design = design
        self.num_timesteps = 0
        self._n_updates = 0
        self.replay_buffer = _FakeReplayBuffer()
        self.policy = torch.nn.Linear(1, 1)
        if nonfinite_parameter:
            self.policy.weight.data.fill_(float("nan"))
        self.log_ent_coef = None
        self.logger = SimpleNamespace(name_to_value={})
        self.final_timestep_offset = final_timestep_offset
        self.nonfinite_training_signal = nonfinite_training_signal
        self.final_update_offset = final_update_offset
        self.physical_action = physical_action
        self.buffer_action = buffer_action

    def learn(
        self,
        *,
        total_timesteps: int,
        callback: object,
        log_interval: int,
        reset_num_timesteps: bool,
        progress_bar: bool,
    ) -> _FakeModel:
        assert log_interval == 1000
        assert reset_num_timesteps is True
        assert progress_bar is False
        callback.init_callback(self)
        callback.on_training_start({}, {})
        n_envs = self.design.environment.n_envs
        local_values = {
            "actions": np.full((n_envs, 17), self.physical_action, dtype=np.float32),
            "buffer_actions": np.full((n_envs, 17), self.buffer_action, dtype=np.float32),
            "new_obs": np.zeros((n_envs, 348), dtype=np.float64),
            "rewards": np.zeros(n_envs, dtype=np.float64),
            "dones": np.zeros(n_envs, dtype=bool),
        }
        for _ in range(total_timesteps // n_envs):
            self.num_timesteps += n_envs
            callback.update_locals(local_values)
            if not callback.on_step():
                break
            if self.num_timesteps > self.design.tqc.learning_starts:
                self._n_updates += self.design.tqc.gradient_steps
                self.logger.name_to_value.update(
                    {
                        "train/actor_loss": 1.0,
                        "train/critic_loss": 2.0,
                        "train/ent_coef": 0.5,
                        "train/ent_coef_loss": -0.25,
                        "train/n_updates": self._n_updates,
                    }
                )
        self.num_timesteps += self.final_timestep_offset
        self._n_updates += self.final_update_offset
        if self.nonfinite_training_signal:
            self.logger.name_to_value["train/actor_loss"] = float("nan")
        callback.on_training_end()
        return self


class _TickClock:
    def __init__(self, delta: float) -> None:
        self.value = 0.0
        self.delta = delta

    def __call__(self) -> float:
        observed = self.value
        self.value += self.delta
        return observed


def _run_fake(
    tmp_path: Path,
    *,
    model: _FakeModel | None = None,
    peak_rss_bytes: int = 64,
    free_disk: int = 100 * 1024**3,
    clock_delta_seconds: float = 1.0,
):
    design = load_calibration_design(DESIGN_PATH).design
    selected_model = model or _FakeModel(design)
    return run_tqc_calibration(
        design_path=DESIGN_PATH,
        output_path=tmp_path / "receipt.json",
        confirm_disposable_resource_probe=True,
        runtime_inspector=lambda _design: _fake_runtime(),
        vector_env_factory=lambda _design: _FakeVectorEnvironment(),
        model_factory=lambda _design, _environment: selected_model,
        clock=_TickClock(clock_delta_seconds),
        peak_rss_reader=lambda: peak_rss_bytes,
        free_disk_reader=lambda _path: free_disk,
    )


def test_committed_design_is_exact_and_excluded() -> None:
    loaded = load_calibration_design(DESIGN_PATH)

    assert loaded.artifact_sha256 == sha256_file(DESIGN_PATH)
    assert loaded.design.seed == 92001
    assert loaded.design.environment.worker_seeds == (92001, 92002, 92003, 92004, 92005)
    assert loaded.design.tqc.required_stable_baselines3_version == "2.9.0"
    assert loaded.design.expected_vector_steps == 20_000
    assert loaded.design.expected_updates == 19_980
    assert loaded.design.tqc.device == "cpu"
    assert loaded.design.environment.normalization == "none/v1"
    assert loaded.design.persistence.model_disposition == MODEL_DISPOSITION
    assert len(loaded.design.semantic_sha256) == 64


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update({"unexpected": 1}), "keys differ"),
        (lambda value: value.update({"seed": 92002}), "frozen values"),
        (
            lambda value: value["environment"].update(
                {"worker_seeds": [92001, 92002, 92003, 92004, 92006]}
            ),
            "frozen values",
        ),
        (
            lambda value: value["tqc"].update({"required_stable_baselines3_version": "2.9.1"}),
            "stable_baselines3_version must equal",
        ),
        (
            lambda value: value["persistence"].update({"checkpoint": True}),
            "persistence must be fully disabled",
        ),
        (
            lambda value: value["environment"].update({"normalization": "VecNormalize"}),
            "normalization must equal",
        ),
        (
            lambda value: value["resource_gates"].update(
                {"minimum_environment_steps_per_second": 1.0}
            ),
            "frozen values",
        ),
    ],
)
def test_design_rejects_semantic_drift(tmp_path: Path, mutate: object, message: str) -> None:
    with pytest.raises(ExperimentContractError, match=message):
        load_calibration_design(_write_mutated_design(tmp_path, mutate))


def test_fake_complete_run_publishes_non_authoritative_receipt(tmp_path: Path) -> None:
    result = _run_fake(tmp_path)
    receipt = json.loads(result.published.path.read_text(encoding="utf-8"))

    assert result.receipt == receipt
    assert receipt["completion_status"] == "complete"
    assert receipt["measured_workload_gates_passed"] is True
    assert receipt["calibration_gate_passed"] is False
    assert receipt["authoritative_execution"] is False
    assert receipt["observed_environment_steps"] == 100_000
    assert receipt["observed_vector_steps"] == 20_000
    assert receipt["observed_gradient_updates"] == 19_980
    assert set(receipt["finite_training_signals"]) == REQUIRED_TRAINING_SIGNALS
    assert receipt["finite_rollout_signal_checks"] == 20_000
    assert len(receipt["sustained_throughput_windows"]) == 9
    assert receipt["replay_buffer_page_write"]["buffer_position_after"] == 0
    assert receipt["replay_buffer_page_write"]["buffer_full_after"] is False
    assert receipt["replay_buffer_page_write"]["total_array_bytes"] == 32
    assert (
        receipt["replay_buffer_page_write"]["claim_boundary"]
        == "page_addresses_written_no_residency_claim/v1"
    )
    assert receipt["checkpoint_emitted"] is False
    assert receipt["replay_buffer_emitted"] is False
    assert receipt["normalizer_emitted"] is False
    assert receipt["behavioral_claim"] is None
    assert receipt["controller_claim"] is None
    assert receipt["controller_artifact"] is None
    assert receipt["rss_threshold_semantics"] == "sampled_failure_threshold_not_os_memory_cap/v1"
    assert receipt["rss_transient_above_threshold_possible"] is True
    assert receipt["oom_before_final_receipt_possible"] is True
    assert result.published.sha256 == sha256_file(result.published.path)


def test_low_disk_fails_before_environment_creation_and_still_publishes(
    tmp_path: Path,
) -> None:
    called = False

    def vector_factory(_design: TQCCalibrationDesign) -> _FakeVectorEnvironment:
        nonlocal called
        called = True
        return _FakeVectorEnvironment()

    result = run_tqc_calibration(
        design_path=DESIGN_PATH,
        output_path=tmp_path / "receipt.json",
        confirm_disposable_resource_probe=True,
        runtime_inspector=lambda _design: _fake_runtime(),
        vector_env_factory=vector_factory,
        model_factory=lambda design, _environment: _FakeModel(design),
        clock=_TickClock(1.0),
        peak_rss_reader=lambda: 1,
        free_disk_reader=lambda _path: 1,
    )

    assert called is False
    assert result.receipt["completion_status"] == "failed"
    assert result.receipt["failure_stage"] == "preflight"
    assert "free disk" in result.receipt["failure_reason"]
    assert result.receipt["calibration_gate_passed"] is False


def test_sampled_peak_rss_threshold_fails_closed(tmp_path: Path) -> None:
    result = _run_fake(tmp_path, peak_rss_bytes=12 * 1024**3 + 1)

    assert result.receipt["completion_status"] == "failed"
    assert result.receipt["failure_stage"] == "preflight"
    assert result.receipt["peak_rss_bytes"] == 12 * 1024**3 + 1
    assert "peak RSS" in result.receipt["failure_reason"]


def test_stale_high_water_does_not_create_a_residency_claim(tmp_path: Path) -> None:
    result = _run_fake(tmp_path, peak_rss_bytes=31)

    assert result.receipt["completion_status"] == "complete"
    page_write = result.receipt["replay_buffer_page_write"]
    assert page_write["total_array_bytes"] == 32
    assert page_write["peak_rss_high_water_bytes_after_write"] == 31
    assert page_write["claim_boundary"] == "page_addresses_written_no_residency_claim/v1"


def test_nonfinite_parameter_fails_closed(tmp_path: Path) -> None:
    design = load_calibration_design(DESIGN_PATH).design
    result = _run_fake(tmp_path, model=_FakeModel(design, nonfinite_parameter=True))

    assert result.receipt["completion_status"] == "failed"
    assert "non-finite" in result.receipt["failure_reason"]
    assert result.receipt["observed_environment_steps"] == 0
    assert result.receipt["checkpoint_emitted"] is False


@pytest.mark.parametrize(
    "model",
    [
        lambda design: _FakeModel(design, physical_action=0.4001),
        lambda design: _FakeModel(design, buffer_action=-1.0001),
    ],
)
def test_action_envelopes_fail_closed(tmp_path: Path, model: object) -> None:
    design = load_calibration_design(DESIGN_PATH).design
    result = _run_fake(tmp_path, model=model(design))

    assert result.receipt["completion_status"] == "failed"
    assert "exact" in result.receipt["failure_reason"]
    assert result.receipt["observed_vector_steps"] == 1


def test_step_mismatch_fails_closed(tmp_path: Path) -> None:
    design = load_calibration_design(DESIGN_PATH).design
    result = _run_fake(tmp_path, model=_FakeModel(design, final_timestep_offset=-5))

    assert result.receipt["completion_status"] == "failed"
    assert result.receipt["observed_environment_steps"] == 99_995
    assert "environment-step budget" in result.receipt["failure_reason"]


def test_update_mismatch_fails_closed(tmp_path: Path) -> None:
    design = load_calibration_design(DESIGN_PATH).design
    result = _run_fake(tmp_path, model=_FakeModel(design, final_update_offset=-1))

    assert result.receipt["completion_status"] == "failed"
    assert result.receipt["observed_gradient_updates"] == 19_979
    assert "update count" in result.receipt["failure_reason"]


def test_nonfinite_training_signal_fails_closed(tmp_path: Path) -> None:
    design = load_calibration_design(DESIGN_PATH).design
    result = _run_fake(tmp_path, model=_FakeModel(design, nonfinite_training_signal=True))

    assert result.receipt["completion_status"] == "failed"
    assert "training signal" in result.receipt["failure_reason"]
    assert result.receipt["calibration_gate_passed"] is False


def test_primary_and_cleanup_failures_are_both_retained(tmp_path: Path) -> None:
    design = load_calibration_design(DESIGN_PATH).design
    result = run_tqc_calibration(
        design_path=DESIGN_PATH,
        output_path=tmp_path / "receipt.json",
        confirm_disposable_resource_probe=True,
        runtime_inspector=lambda _design: _fake_runtime(),
        vector_env_factory=lambda _design: _FakeVectorEnvironment(close_error=True),
        model_factory=lambda _design, _environment: _FakeModel(
            design, nonfinite_training_signal=True
        ),
        clock=_TickClock(1.0),
        peak_rss_reader=lambda: 64,
        free_disk_reader=lambda _path: 100 * 1024**3,
    )

    assert "training signal" in result.receipt["failure_reason"]
    assert len(result.receipt["cleanup_failures"]) == 1
    assert "forced close failure" in result.receipt["cleanup_failures"][0]


def test_throughput_floor_is_a_hard_gate(tmp_path: Path) -> None:
    result = _run_fake(tmp_path, clock_delta_seconds=100.0)

    assert result.receipt["completion_status"] == "failed"
    assert result.receipt["observed_environment_steps"] == 20_000
    assert result.receipt["sustained_throughput_windows"][0][
        "environment_steps_per_second"
    ] == pytest.approx(100.0)
    assert "throughput" in result.receipt["failure_reason"]


def test_existing_output_is_rejected_before_runtime_work(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    output.write_text("keep", encoding="utf-8")
    called = False

    def runtime_inspector(_design: TQCCalibrationDesign) -> dict[str, object]:
        nonlocal called
        called = True
        return _fake_runtime()

    with pytest.raises(ExperimentContractError, match="overwrite"):
        run_tqc_calibration(
            design_path=DESIGN_PATH,
            output_path=output,
            confirm_disposable_resource_probe=True,
            runtime_inspector=runtime_inspector,
        )
    assert called is False
    assert output.read_text(encoding="utf-8") == "keep"


def test_resource_probe_requires_explicit_acknowledgement_before_runtime(
    tmp_path: Path,
) -> None:
    called = False

    def runtime_inspector(_design: TQCCalibrationDesign) -> dict[str, object]:
        nonlocal called
        called = True
        return _fake_runtime()

    output = tmp_path / "receipt.json"
    with pytest.raises(ExperimentContractError, match="acknowledgement"):
        run_tqc_calibration(
            design_path=DESIGN_PATH,
            output_path=output,
            runtime_inspector=runtime_inspector,
        )

    assert called is False
    assert not output.exists()


def test_symlinked_output_ancestor_is_rejected_before_runtime(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    called = False

    def runtime_inspector(_design: TQCCalibrationDesign) -> dict[str, object]:
        nonlocal called
        called = True
        return _fake_runtime()

    with pytest.raises(ExperimentContractError, match="symbolic links"):
        run_tqc_calibration(
            design_path=DESIGN_PATH,
            output_path=linked_parent / "receipt.json",
            confirm_disposable_resource_probe=True,
            runtime_inspector=runtime_inspector,
        )

    assert called is False
    assert not (real_parent / "receipt.json").exists()


def test_runtime_failure_finalizes_the_preflight_reservation(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    observed_provisional: dict[str, object] = {}

    def runtime_inspector(_design: TQCCalibrationDesign) -> dict[str, object]:
        observed_provisional.update(json.loads(output.read_text(encoding="utf-8")))
        raise RuntimeError("runtime inspection stopped")

    result = run_tqc_calibration(
        design_path=DESIGN_PATH,
        output_path=output,
        confirm_disposable_resource_probe=True,
        runtime_inspector=runtime_inspector,
    )

    assert observed_provisional["completion_status"] == "in_progress"
    assert result.receipt["completion_status"] == "failed"
    assert result.receipt["failure_stage"] == "runtime_inspection"
    assert json.loads(output.read_text(encoding="utf-8")) == result.receipt
    assert result.published.sha256 == sha256_file(output)


def test_direct_cli_requires_resource_acknowledgement(tmp_path: Path, capsys) -> None:
    output = tmp_path / "receipt.json"

    code = tqc_calibration_main(["--design", str(DESIGN_PATH), "--output", str(output)])

    assert code == 2
    assert "--confirm-disposable-resource-probe" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.gym
def test_real_humanoid_tqc_ultrashort_smoke(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("sb3_contrib")
    pytest.importorskip("gymnasium.envs.mujoco")
    forbidden_log_directory = tmp_path / "sb3-log-must-not-exist"
    monkeypatch.setenv("SB3_LOGDIR", str(forbidden_log_directory))
    design = load_calibration_design(DESIGN_PATH).design
    smoke = replace(
        design,
        environment=replace(design.environment, n_envs=1, worker_seeds=(92001,)),
        tqc=replace(
            design.tqc,
            total_timesteps=10,
            buffer_size=20,
            learning_starts=5,
            batch_size=2,
            top_quantiles_to_drop_per_net=1,
            n_quantiles=5,
            n_critics=1,
            policy_hidden_layers=(8, 8),
        ),
    )

    runtime = inspect_tqc_runtime(design)
    environment = make_tqc_vector_env(smoke)
    try:
        model = make_tqc_model(smoke, environment)
        model_runtime = inspect_tqc_model(model, smoke)
        residency = fault_replay_buffer_pages(model, resource_check=lambda: None)
        callback = _make_finite_resource_callback(
            design=smoke,
            output_parent=REPOSITORY_ROOT,
            peak_rss_reader=lambda: 1,
            free_disk_reader=lambda _path: 100 * 1024**3,
            clock=_TickClock(1.0),
        )
        model.learn(total_timesteps=10, callback=callback, progress_bar=False)
        assert model.num_timesteps == 10
        assert smoke.expected_updates == 5
        assert model._n_updates == smoke.expected_updates
        assert replay_buffer_allocation_bytes(model) > 0
        assert model_runtime["replay_total_transition_slots"] == 20
        assert model_runtime["pending_worker_seeds"] == [92001]
        assert model_runtime["custom_logger_active"] is True
        assert model_runtime["logger_directory"] is None
        assert model_runtime["logger_output_format_count"] == 0
        assert len(model_runtime["model_semantic_sha256"]) == 64
        assert residency["total_array_bytes"] == replay_buffer_allocation_bytes(model)
        assert callback.signal_checks == 10
        assert _assert_finite_parameters(model) > 0
        assert runtime["environment_id"] == "Humanoid-v5"
        assert runtime["sb3_contrib_version"] == "2.9.0"
        assert runtime["stable_baselines3_version"] == "2.9.0"
        assert runtime["host_hardware"]["logical_cpu_count"] > 0
        assert runtime["host_hardware"]["total_memory_bytes"] > 0
        assert runtime["torch_intraop_thread_count"] > 0
        assert runtime["torch_interop_thread_count"] > 0
        assert len(runtime["runtime_sha256"]) == 64
        assert not forbidden_log_directory.exists()
    finally:
        environment.close()
