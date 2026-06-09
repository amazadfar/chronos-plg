"""Tests for the data pipeline."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.labels import LabelGenerator
from src.data.build_dataset import DatasetBuilder
from src.data.contracts import DataContractError, validate_raw_data_contracts
from src.data.quality_gate import evaluate_degraded_run_gate
from config.settings import get_settings


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 200
    
    dates = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    
    # Simulate price with some patterns
    returns = np.random.normal(0, 0.02, n)
    returns[50:70] += 0.005  # Bull period
    returns[100:120] -= 0.008  # Bear period
    
    price = 40000 * np.exp(np.cumsum(returns))
    
    return pd.DataFrame({
        "open": price * (1 + np.random.uniform(-0.005, 0.005, n)),
        "high": price * (1 + np.random.uniform(0, 0.015, n)),
        "low": price * (1 - np.random.uniform(0, 0.015, n)),
        "close": price,
        "volume": np.random.uniform(1000, 5000, n),
        "quote_volume": np.random.uniform(40000000, 200000000, n),
        "trades": np.random.randint(10000, 50000, n),
    }, index=dates)


@pytest.fixture
def phase2_raw_slice(sample_ohlcv):
    """Create realistic multi-source raw slice for Phase 2 integration tests."""
    idx = sample_ohlcv.index
    funding_idx = idx[::2]
    liq_idx = idx[::3]
    macro_idx = pd.date_range(idx.min().floor("D"), idx.max().ceil("D"), freq="1D", tz="UTC")

    funding_rate = pd.DataFrame(
        {"funding_rate": np.linspace(-0.0002, 0.0002, len(funding_idx))},
        index=funding_idx,
    )
    open_interest = pd.DataFrame(
        {
            "open_interest": np.linspace(10000, 20000, len(idx)),
            "open_interest_value": np.linspace(3e8, 5e8, len(idx)),
        },
        index=idx,
    )
    macro = pd.DataFrame(
        {
            "dxy_return_1d": np.linspace(-0.01, 0.01, len(macro_idx)),
            "spx_return_1d": np.linspace(-0.02, 0.02, len(macro_idx)),
            "vix": np.linspace(15, 25, len(macro_idx)),
            "yield_curve_2_10": np.linspace(-0.002, 0.01, len(macro_idx)),
        },
        index=macro_idx,
    )
    event_flags = pd.DataFrame(
        {
            "is_fomc_day": 0,
            "is_cpi_day": 0,
            "post_fomc_window": 0,
            "post_cpi_window": 0,
            "is_event_window": 0,
        },
        index=idx,
    )
    liquidations = pd.DataFrame(
        {
            "long_liq_usd_est": np.linspace(1e4, 3e4, len(liq_idx)),
            "short_liq_usd_est": np.linspace(2e4, 1e4, len(liq_idx)),
            "liq_imbalance_est": np.linspace(-0.3, 0.3, len(liq_idx)),
            "has_real_liq_data": 1,
            "liq_data_source_code": 1,
        },
        index=liq_idx,
    )
    contract_metadata = pd.DataFrame(
        [
            {
                "exchange": "binance",
                "market_type": "futures",
                "symbol": "BTCUSDT",
                "base_asset": "BTC",
                "quote_asset": "USDT",
                "tick_size": 0.1,
                "lot_size": 0.001,
                "min_qty": 0.001,
                "min_notional": 100.0,
            }
        ],
        index=["BTCUSDT"],
    )

    return {
        "ohlcv": sample_ohlcv.copy(),
        "funding_rate": funding_rate,
        "open_interest": open_interest,
        "macro": macro,
        "event_flags": event_flags,
        "liquidations": liquidations,
        "contract_metadata": contract_metadata,
    }


class TestLabelGenerator:
    """Tests for LabelGenerator."""
    
    def test_forward_returns_shape(self, sample_ohlcv):
        """Forward returns should have same length as input."""
        gen = LabelGenerator()
        returns = gen.compute_forward_returns(sample_ohlcv["close"])
        
        assert len(returns) == len(sample_ohlcv)
        
    def test_forward_returns_last_is_nan(self, sample_ohlcv):
        """Last value should be NaN (no future data)."""
        gen = LabelGenerator()
        returns = gen.compute_forward_returns(sample_ohlcv["close"], horizon=1)
        
        assert pd.isna(returns.iloc[-1])
        
    def test_forward_returns_values(self, sample_ohlcv):
        """Verify forward return calculation is correct."""
        gen = LabelGenerator()
        close = sample_ohlcv["close"]
        returns = gen.compute_forward_returns(close, horizon=1)
        
        # Manual calculation for first few values
        expected = np.log(close.iloc[1] / close.iloc[0])
        actual = returns.iloc[0]
        
        np.testing.assert_almost_equal(actual, expected, decimal=10)
    
    def test_realized_volatility_shape(self, sample_ohlcv):
        """RV should have same length as input."""
        gen = LabelGenerator()
        rv = gen.compute_realized_volatility(sample_ohlcv["close"])
        
        assert len(rv) == len(sample_ohlcv)
    
    def test_realized_volatility_last_is_nan(self, sample_ohlcv):
        """Last values should be NaN (no future data)."""
        gen = LabelGenerator()
        rv = gen.compute_realized_volatility(sample_ohlcv["close"], window=6)
        
        # Last 6 values should be NaN
        assert pd.isna(rv.iloc[-1])
    
    def test_regime_labels_valid(self, sample_ohlcv):
        """Regime labels should be one of the valid values."""
        gen = LabelGenerator()
        regimes = gen.compute_regime_labels(sample_ohlcv["close"])
        
        valid_regimes = {"normal", "trend", "chop", "panic"}
        actual_regimes = set(regimes["regime"].dropna().unique())
        
        assert actual_regimes.issubset(valid_regimes)
    
    def test_generate_all_labels(self, sample_ohlcv):
        """All labels should be generated."""
        gen = LabelGenerator()
        labels = gen.generate_all_labels(sample_ohlcv)
        
        expected_cols = [
            "forward_return",
            "forward_realized_vol",
            "hist_q10",
            "hist_q50",
            "hist_q90",
            "regime",
            "forward_direction",
        ]
        
        for col in expected_cols:
            assert col in labels.columns, f"Missing column: {col}"


class TestAntiLeakage:
    """Tests to verify no future information leakage."""
    
    def test_no_perfect_correlation(self, sample_ohlcv):
        """No feature should have perfect correlation with target."""
        gen = LabelGenerator()
        labels = gen.generate_all_labels(sample_ohlcv)
        
        # Create simple features
        features = pd.DataFrame(index=sample_ohlcv.index)
        features["return_1"] = np.log(sample_ohlcv["close"] / sample_ohlcv["close"].shift(1))
        features["return_6"] = np.log(sample_ohlcv["close"] / sample_ohlcv["close"].shift(6))
        
        # Validate
        result = gen.validate_no_leakage(features, labels)
        
        assert result["passed"], f"Leakage detected: {result['errors']}"
    
    def test_detect_obvious_leakage(self, sample_ohlcv):
        """Should detect if we accidentally include target as feature."""
        gen = LabelGenerator()
        labels = gen.generate_all_labels(sample_ohlcv)
        
        # Create features with obvious leakage
        features = pd.DataFrame(index=sample_ohlcv.index)
        features["leaked_feature"] = labels["forward_return"]  # This is leakage!
        
        result = gen.validate_no_leakage(features, labels)
        
        # Should fail or at least warn
        assert not result["passed"] or len(result["warnings"]) > 0

    def test_detect_shifted_target_leakage(self, sample_ohlcv):
        """Should fail if a feature reproduces shifted future target values."""
        gen = LabelGenerator()
        labels = gen.generate_all_labels(sample_ohlcv)

        features = pd.DataFrame(index=sample_ohlcv.index)
        features["leak_tplus1"] = labels["forward_return"].shift(-1)

        result = gen.validate_no_leakage(features, labels)

        assert not result["passed"]
        assert any("future target" in err for err in result["errors"])

    def test_fail_on_forward_prefixed_feature(self, sample_ohlcv):
        """Forward-prefixed feature columns are disallowed by leakage validator."""
        gen = LabelGenerator()
        labels = gen.generate_all_labels(sample_ohlcv)

        features = pd.DataFrame(index=sample_ohlcv.index)
        features["forward_spoof"] = np.random.normal(0, 0.01, len(features))

        result = gen.validate_no_leakage(features, labels)

        assert not result["passed"]
        assert any("Forward-looking columns found in features" in err for err in result["errors"])

    def test_fail_on_unusable_forward_horizon_boundary(self, sample_ohlcv):
        """If target marks a row valid without future boundary, leakage checks should fail."""
        gen = LabelGenerator()
        labels = gen.generate_all_labels(sample_ohlcv)
        labels.loc[labels.index[-1], "forward_return"] = 0.0

        features = pd.DataFrame(index=sample_ohlcv.index)
        features["return_1"] = np.log(sample_ohlcv["close"] / sample_ohlcv["close"].shift(1))

        result = gen.validate_no_leakage(features, labels)

        assert not result["passed"]
        assert any("Forward target horizon boundary missing future timestamps" in err for err in result["errors"])


class TestFeatureComputation:
    """Tests for feature computation."""
    
    def test_return_features_no_nan_middle(self, sample_ohlcv):
        """Return features should not have NaN in the middle of the series."""
        builder = DatasetBuilder()
        
        data = {
            "ohlcv": sample_ohlcv,
            "funding_rate": pd.DataFrame(),
            "open_interest": pd.DataFrame(),
            "macro": pd.DataFrame(),
            "event_flags": pd.DataFrame(),
            "liquidations": pd.DataFrame(),
        }
        
        features = builder.compute_features(data)
        
        # After warm-up period, return_1 should not have NaN
        assert features["return_1"].iloc[50:150].isna().sum() == 0
    
    def test_missingness_flags_present(self, sample_ohlcv):
        """Missingness flags should be present."""
        builder = DatasetBuilder()
        
        data = {
            "ohlcv": sample_ohlcv,
            "funding_rate": pd.DataFrame(),
            "open_interest": pd.DataFrame(),
            "macro": pd.DataFrame(),
            "event_flags": pd.DataFrame(),
            "liquidations": pd.DataFrame(),
        }
        
        features = builder.compute_features(data)
        
        assert "has_funding" in features.columns
        assert "has_oi" in features.columns
        assert "has_liqs" in features.columns
        assert "has_macro" in features.columns
        assert "has_real_liq_data" in features.columns


class TestPhase2DataPipeline:
    """Integration tests for Phase 2 data hardening."""

    def test_raw_contracts_accept_realistic_slice(self, phase2_raw_slice):
        """Raw contract validation should pass on realistic multi-source slice."""
        validate_raw_data_contracts(phase2_raw_slice)

    def test_raw_contracts_reject_naive_ohlcv_index(self, phase2_raw_slice):
        """Datetime index must be timezone-aware for time-series datasets."""
        bad = dict(phase2_raw_slice)
        bad_ohlcv = phase2_raw_slice["ohlcv"].copy()
        bad_ohlcv.index = bad_ohlcv.index.tz_localize(None)
        bad["ohlcv"] = bad_ohlcv

        with pytest.raises(DataContractError, match="timezone-aware"):
            validate_raw_data_contracts(bad)

    def test_liq_provenance_defaults_for_legacy_liq_frames(self, phase2_raw_slice):
        """
        Legacy liquidation frames without provenance columns should still be usable.

        They are treated as estimated data (source_code=0).
        """
        legacy_liqs = phase2_raw_slice["liquidations"][
            ["long_liq_usd_est", "short_liq_usd_est", "liq_imbalance_est"]
        ]
        data = dict(phase2_raw_slice)
        data["liquidations"] = legacy_liqs

        features = DatasetBuilder().compute_features(data)

        assert set(features["has_real_liq_data"].dropna().unique().tolist()) == {0}
        assert set(features["liq_data_source_code"].dropna().unique().tolist()) == {0}
        assert set(features["has_liqs"].dropna().unique().tolist()) == {1}

    def test_build_dataset_calls_quality_report(self, phase2_raw_slice, monkeypatch):
        """Dataset build should generate quality report payload on every run."""
        builder = DatasetBuilder()
        original = builder.generate_quality_report
        calls = {"count": 0}

        def wrapped(raw_data, dataset):
            calls["count"] += 1
            report = original(raw_data, dataset)
            assert "coverage_windows" in report
            assert "contract_metadata" in report["coverage_windows"]
            assert report["coverage_windows"]["contract_metadata"]["rows"] == 1
            return report

        monkeypatch.setattr(builder, "generate_quality_report", wrapped)
        dataset = builder.build_dataset(phase2_raw_slice, save=False)

        assert len(dataset) == len(phase2_raw_slice["ohlcv"])
        assert calls["count"] == 1

    def test_ohlcv_gap_ratio_guardrail(self, sample_ohlcv):
        """OHLCV integrity check should fail when gaps exceed configured threshold."""
        gappy_ohlcv = sample_ohlcv.drop(sample_ohlcv.index[::5])  # 20% gap rate
        data = {
            "ohlcv": gappy_ohlcv,
            "funding_rate": pd.DataFrame(),
            "open_interest": pd.DataFrame(),
            "macro": pd.DataFrame(),
            "event_flags": pd.DataFrame(),
            "liquidations": pd.DataFrame(),
            "contract_metadata": pd.DataFrame(),
        }

        with pytest.raises(ValueError, match="gap ratio too high"):
            DatasetBuilder().compute_features(data)


class TestPhase11DataQualityGate:
    """Phase 11 data-completeness gate tests."""

    def test_quality_report_includes_key_family_availability(self, phase2_raw_slice):
        builder = DatasetBuilder()
        dataset = builder.build_dataset(phase2_raw_slice, save=False)
        report = builder.generate_quality_report(phase2_raw_slice, dataset)

        assert "key_family_availability" in report
        families = report["key_family_availability"]
        for family in ["funding", "open_interest", "liquidations", "macro"]:
            assert family in families
            assert "availability_ratio" in families[family]

        assert "quality_gate" in report
        assert isinstance(report.get("data_degraded"), bool)
        assert isinstance(report.get("degradation_reasons"), list)

    def test_degraded_gate_fails_when_oi_and_liq_unavailable(self, sample_ohlcv):
        builder = DatasetBuilder()
        raw_data = {
            "ohlcv": sample_ohlcv,
            "funding_rate": pd.DataFrame(index=sample_ohlcv.index, data={"funding_rate": 0.0}),
            "open_interest": pd.DataFrame(),  # missing OI
            "macro": pd.DataFrame(
                index=sample_ohlcv.index,
                data={
                    "dxy_return_1d": 0.0,
                    "spx_return_1d": 0.0,
                    "vix": 20.0,
                    "yield_curve_2_10": 0.01,
                },
            ),
            "event_flags": pd.DataFrame(),
            "liquidations": pd.DataFrame(),  # missing liq
            "contract_metadata": pd.DataFrame(
                [
                    {
                        "exchange": "binance",
                        "market_type": "futures",
                        "symbol": "BTCUSDT",
                        "base_asset": "BTC",
                        "quote_asset": "USDT",
                        "tick_size": 0.1,
                        "lot_size": 0.001,
                        "min_qty": 0.001,
                        "min_notional": 100.0,
                    }
                ],
                index=["BTCUSDT"],
            ),
        }
        dataset = builder.build_dataset(raw_data, save=False)
        report = builder.generate_quality_report(raw_data, dataset)

        gate = evaluate_degraded_run_gate(report, market_type="futures")
        assert gate.passed is False
        assert any("open_interest_availability_below_threshold" in reason for reason in gate.reasons)
        assert any("liquidations_availability_below_threshold" in reason for reason in gate.reasons)

    def test_degraded_gate_passes_for_full_futures_slice(self, phase2_raw_slice):
        builder = DatasetBuilder()
        dataset = builder.build_dataset(phase2_raw_slice, save=False)
        report = builder.generate_quality_report(phase2_raw_slice, dataset)

        gate = evaluate_degraded_run_gate(report, market_type="futures")
        assert gate.passed is True


def test_config_loads():
    """Settings should load without error."""
    settings = get_settings()
    
    assert settings.binance.symbol == "BTCUSDT"
    assert settings.binance.interval == "4h"
    assert settings.target.horizon_candles == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
