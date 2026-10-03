"""End-to-end: a miniature version of the actual experiment."""
import numpy as np
import pytest

from mint.experiments.transfer import ExperimentConfig, report, run


@pytest.mark.slow
def test_experiment_runs_and_oracle_is_perfect(tmp_path):
    cfg = ExperimentConfig(
        run_name="test", source_domains=("skirmish",), target_domains=("clinic",),
        seeds=(0,), spurious=(0.0, 2.0), target_n_train=800, target_epochs=6,
        n_focal=16, n_probe=96, holdout_models=1, feature_ablations=False,
        interpreter={"epochs": 2, "d_model": 32, "n_layers": 1, "n_heads": 2, "batch_size": 16},
        cache_dir=str(tmp_path / "cache"), out_dir=str(tmp_path),
    )
    res = run(cfg, refresh=True, run_executed=True)
    rows = res["rows"]
    assert {r["cell"] for r in rows} == {"skirmish/mlp", "clinic/mlp"}
    methods = {r["method"] for r in rows}
    assert {"A_full", "A_output_only", "A_ceiling_clinic/mlp",
            "correlation", "first_order", "oracle_effect"} <= methods
    assert "A_procedure_only" not in methods, "the legacy vocabulary arm is off by default"

    for r in rows:
        if r["method"] == "oracle_effect":
            assert r["causal.spearman"] > 0.999
            assert r["causal.precision@3"] > 0.999
    for r in rows:
        assert -1.01 <= r["mediation.spearman"] <= 1.01
        assert r["n_models"] == 1
        assert len(r["per_model"]) == 1
    assert isinstance(report(res), str) and "TRANSFER" in report(res)


@pytest.mark.slow
def test_bundle_cache_round_trip(tmp_path):
    from mint.experiments.transfer import build_or_load
    cfg = ExperimentConfig(seeds=(0,), spurious=(1.0,), target_n_train=500, target_epochs=3,
                           n_focal=8, n_probe=64, cache_dir=str(tmp_path / "c"))
    a = build_or_load("skirmish", "mlp", cfg, refresh=True)
    b = build_or_load("skirmish", "mlp", cfg, refresh=False)
    assert len(a) == len(b) == 1
    assert np.allclose(a[0].gt.unit_effect, b[0].gt.unit_effect)


def test_cache_key_tracks_the_population_config():
    from mint.experiments.transfer import ExperimentConfig, cache_key
    a = ExperimentConfig(seeds=(0, 1))
    b = ExperimentConfig(seeds=(0, 1, 2))
    assert cache_key("skirmish", "mlp", a) != cache_key("skirmish", "mlp", b)
    assert cache_key("skirmish", "mlp", a) == cache_key("skirmish", "mlp", ExperimentConfig(seeds=(0, 1)))
    assert cache_key("clinic", "mlp", a) != cache_key("skirmish", "mlp", a)
    assert cache_key("skirmish", "tab_transformer", a) != cache_key("skirmish", "mlp", a)


def test_refuses_to_hold_out_the_whole_population(tmp_path):
    import pytest
    from mint.experiments.transfer import ExperimentConfig, run
    cfg = ExperimentConfig(seeds=(0,), spurious=(1.0,), target_n_train=400, target_epochs=2,
                           n_focal=8, n_probe=48, holdout_models=1, feature_ablations=False,
                           target_domains=("clinic",), cache_dir=str(tmp_path / "c"),
                           out_dir=str(tmp_path))
    with pytest.raises(ValueError, match="no models to fit"):
        run(cfg, refresh=True, run_executed=False)
