# 99 - Appendix: File-Level Inventory

Generated from AST scan of Python source files in `config/`, `scripts/`, `src/`, and `tests/`.

## config/

| File                           | Module Doc                                                              | Top-Level Classes                                                                                                                        | Top-Level Functions   |
|--------------------------------|-------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|
| `config/__init__.py`           | Configuration module for chronos-plg.                                   | -                                                                                                                                        | -                     |
| `config/baseline_protocols.py` | Immutable baseline evaluation protocols for reproducible comparability. | BaselineModelSpec, BaselineProtocol                                                                                                      | get_baseline_protocol |
| `config/cost_profiles.py`      | Exchange and market-specific fee/cost profiles.                         | MarketFeeProfile, ExchangeCostProfile                                                                                                    | get_cost_profile      |
| `config/scenario_profiles.py`  | Named scenario profiles used for benchmark and backtest runs.           | TradingScenarioProfile                                                                                                                   | get_scenario_profile  |
| `config/settings.py`           | Centralized configuration for the Chronos-2 BTC trading system.         | DataPaths, BinanceConfig, MacroConfig, TargetConfig, FeatureConfig, WalkForwardConfig, CostConfig, StrategyConfig, ModelConfig, Settings | get_settings          |

## scripts/

| File                           | Module Doc                                                                 | Top-Level Classes | Top-Level Functions                                                                                                           |
|--------------------------------|----------------------------------------------------------------------------|-------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `scripts/__init__.py`          | Scripts package.                                                           | -                 | -                                                                                                                             |
| `scripts/benchmark.py`         | Comprehensive benchmark script with visualization.                         | -                 | run_benchmark, run_robustness_analysis, create_equity_curves_plot, create_regime_analysis_plot, generate_summary_report, main |
| `scripts/build_features.py`    | Build processed dataset/features from locally cached raw data.             | -                 | main                                                                                                                          |
| `scripts/download_data.py`     | Download all data for the Chronos-2 trading system.                        | -                 | -                                                                                                                             |
| `scripts/run_backtest.py`      | Run complete backtest with all models.                                     | -                 | main                                                                                                                          |
| `scripts/run_baselines.py`     | Run Phase 6 baseline protocol with frozen folds and net-cost engine.       | -                 | _safe_name, _write_json, parse_args, main                                                                                     |
| `scripts/run_chronos2.py`      | Run Phase 7 Chronos/meta validation with leakage/net-cost gating.          | -                 | _safe_name, _write_json, _build_cost_model, parse_args, main                                                                  |
| `scripts/run_paper_trading.py` | Run Phase 10 paper-trading replay with monitoring and deployment policies. | -                 | _select_feature_columns, _write_json, main                                                                                    |
| `scripts/smoke_check.py`       | Fast smoke check for core pipeline wiring.                                 | -                 | _synthetic_dataset, main                                                                                                      |

## src/

| File                                        | Module Doc                                                                          | Top-Level Classes                                                                                                                        | Top-Level Functions                                                                                                                                                                                                                           |
|---------------------------------------------|-------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `src/__init__.py`                           | Chronos-PLG: Chronos-2 based probabilistic trading system for BTC.                  | -                                                                                                                                        | -                                                                                                                                                                                                                                             |
| `src/backtest/__init__.py`                  | Backtest package.                                                                   | -                                                                                                                                        | -                                                                                                                                                                                                                                             |
| `src/backtest/costs.py`                     | Execution-event trading cost model.                                                 | TransitionLegs, TradeCosts, ExecutionCostEvent, CostModel                                                                                | main                                                                                                                                                                                                                                          |
| `src/backtest/engine.py`                    | Backtest engine for walk-forward strategy evaluation.                               | BacktestResult, BacktestEngine                                                                                                           | main                                                                                                                                                                                                                                          |
| `src/backtest/report.py`                    | -                                                                                   | BacktestReport                                                                                                                           | compare_models, main                                                                                                                                                                                                                          |
| `src/common/__init__.py`                    | Shared constants and helpers used across modules.                                   | -                                                                                                                                        | -                                                                                                                                                                                                                                             |
| `src/common/metrics.py`                     | Shared metric names, thresholds, and metric helpers.                                | MetricName, SuccessThresholds                                                                                                            | safe_div, profit_factor_from_returns, sharpe_ratio, recent_vs_early_sharpe_ratio                                                                                                                                                              |
| `src/data/__init__.py`                      | Data fetching and processing modules.                                               | -                                                                                                                                        | __getattr__                                                                                                                                                                                                                                   |
| `src/data/binance_fetcher.py`               | Binance Futures data fetcher for BTCUSDT perpetual.                                 | BinanceFetcher                                                                                                                           | -                                                                                                                                                                                                                                             |
| `src/data/build_dataset.py`                 | Dataset builder for the Chronos-2 trading system.                                   | DatasetBuilder                                                                                                                           | -                                                                                                                                                                                                                                             |
| `src/data/contracts.py`                     | Data contracts and validation helpers for raw and processed datasets.               | DataContractError, IndexGapStats                                                                                                         | validate_datetime_index, validate_required_columns, validate_raw_data_contracts, compute_index_gap_stats                                                                                                                                      |
| `src/data/labels.py`                        | Label generator for the Chronos-2 trading system.                                   | LabelGenerator                                                                                                                           | main                                                                                                                                                                                                                                          |
| `src/data/liquidation_collector.py`         | Binance liquidation data collector.                                                 | LiquidationEvent, AggregatedLiquidations, LiquidationCollector                                                                           | -                                                                                                                                                                                                                                             |
| `src/data/macro_fetcher.py`                 | Macro data fetcher using yfinance.                                                  | MacroFetcher                                                                                                                             | main                                                                                                                                                                                                                                          |
| `src/data/market_metadata.py`               | Exchange-specific symbol/contract metadata.                                         | ContractMetadata                                                                                                                         | get_contract_metadata                                                                                                                                                                                                                         |
| `src/evaluation/__init__.py`                | Evaluation package.                                                                 | -                                                                                                                                        | -                                                                                                                                                                                                                                             |
| `src/evaluation/metrics.py`                 | Evaluation metrics for forecast and trading performance.                            | QuantileMetrics, TradingMetrics                                                                                                          | pinball_loss, compute_quantile_metrics, compute_trading_metrics, main                                                                                                                                                                         |
| `src/evaluation/phase6_baselines.py`        | Phase 6 baseline protocol utilities.                                                | -                                                                                                                                        | infer_feature_columns, _stable_payload_hash, write_protocol_freeze, freeze_fold_schedule, resolve_model_configs, effective_profit_factor, build_leaderboard, write_leaderboard_artifacts, build_chronos_advancement_gate, write_gate_artifact |
| `src/evaluation/phase7_chronos.py`          | Phase 7 Chronos/meta validation utilities.                                          | -                                                                                                                                        | determine_recent_regime_start, compute_recent_regime_metrics, build_phase7_candidate_gate                                                                                                                                                     |
| `src/evaluation/walk_forward.py`            | Walk-Forward Evaluation Harness.                                                    | FoldResult, WalkForwardResults, WalkForwardEvaluator                                                                                     | main                                                                                                                                                                                                                                          |
| `src/models/__init__.py`                    | Models package.                                                                     | -                                                                                                                                        | __getattr__                                                                                                                                                                                                                                   |
| `src/models/baselines/__init__.py`          | Baseline models package.                                                            | -                                                                                                                                        | __getattr__                                                                                                                                                                                                                                   |
| `src/models/baselines/ewma.py`              | EWMA (Exponentially Weighted Moving Average) Baseline.                              | EWMABaseline, ARBaseline                                                                                                                 | main                                                                                                                                                                                                                                          |
| `src/models/baselines/lightgbm_quantile.py` | LightGBM Quantile Baseline.                                                         | LightGBMQuantileBaseline                                                                                                                 | main                                                                                                                                                                                                                                          |
| `src/models/baselines/random_walk.py`       | Random Walk Baseline.                                                               | BaselineModel, RandomWalkBaseline                                                                                                        | main                                                                                                                                                                                                                                          |
| `src/models/chronos2_runner.py`             | Chronos-2 model runner with strict rolling out-of-sample inference.                 | _ChronosConfig, Chronos2Runner, Chronos2ForReturns                                                                                       | _is_numeric_dtype                                                                                                                                                                                                                             |
| `src/models/meta_model.py`                  | Meta-model stacking Chronos quantile forecasts with tabular features.               | MetaModel                                                                                                                                | -                                                                                                                                                                                                                                             |
| `src/paper_trading/__init__.py`             | Paper-trading replay, monitoring, and deployment policy helpers.                    | -                                                                                                                                        | -                                                                                                                                                                                                                                             |
| `src/paper_trading/engine.py`               | Paper-trading replay engine using the same execution-cost assumptions as backtests. | PaperTradingConfig, PaperTradingReplay, PaperTradingEngine                                                                               | -                                                                                                                                                                                                                                             |
| `src/paper_trading/monitoring.py`           | Monitoring dashboards for paper-trading metrics and cost decomposition.             | -                                                                                                                                        | _max_drawdown_from_returns, summarize_returns_window, build_monitoring_dashboard, build_daily_weekly_dashboards                                                                                                                               |
| `src/paper_trading/policy.py`               | Paper-trading kill-switch, readiness, and capital ramp policy definitions.          | KillSwitchThresholds, KillSwitchEvent, DeploymentReadinessPolicy, DeploymentReadiness, RampStage, CapitalRampPolicy, CapitalRampDecision | _to_float, _row_has_activity, evaluate_kill_switch, evaluate_deployment_readiness, default_capital_ramp_policy, _stage_map, _meets_stage_thresholds, recommend_capital_action, render_capital_ramp_policy, serialize_kill_events              |
| `src/reporting/__init__.py`                 | Reporting package.                                                                  | -                                                                                                                                        | -                                                                                                                                                                                                                                             |
| `src/reporting/decision.py`                 | Decision framework and uncertainty bands for Phase 9 reporting.                     | DecisionOutcome, UncertaintyBand, DecisionReport                                                                                         | effective_profit_factor, _quantile_band, _fold_band, _block_bootstrap_band, compute_uncertainty_bands, build_decision_report, save_decision_artifacts                                                                                         |
| `src/robustness/__init__.py`                | Robustness package.                                                                 | -                                                                                                                                        | -                                                                                                                                                                                                                                             |
| `src/robustness/kill_criteria.py`           | Kill criteria for strategy validation.                                              | CriterionStatus, CriterionResult, KillCriteriaResult, KillCriteria                                                                       | main                                                                                                                                                                                                                                          |
| `src/robustness/stress_tests.py`            | Stress testing for strategy robustness.                                             | StressTestResult, StressTestSuite, StressTester                                                                                          | _sharpe_degradation                                                                                                                                                                                                                           |
| `src/robustness/summary.py`                 | Robustness summary generator.                                                       | RobustnessReport, RobustnessSummary                                                                                                      | main                                                                                                                                                                                                                                          |
| `src/strategy/__init__.py`                  | Strategy package.                                                                   | -                                                                                                                                        | -                                                                                                                                                                                                                                             |
| `src/strategy/execution_intent.py`          | Execution intent abstraction between position targets and cost/execution engine.    | ExecutionPolicy, ExecutionIntent, ExecutionIntentBuilder                                                                                 | classify_transition                                                                                                                                                                                                                           |
| `src/strategy/position_sizing.py`           | Position sizing based on predictions and risk.                                      | PositionConstraints, PositionSizer, KellyCriterionSizer                                                                                  | main                                                                                                                                                                                                                                          |
| `src/strategy/regime_detector.py`           | Market regime detection and strategy gating.                                        | Regime, RegimeDetector                                                                                                                   | main                                                                                                                                                                                                                                          |
| `src/strategy/signals.py`                   | Quantile-based trading signal generator.                                            | Signal, ForecastSnapshot, TradeDecision, QuantileSignalGenerator                                                                         | main                                                                                                                                                                                                                                          |
| `src/strategy/strategy.py`                  | Integrated trading strategy combining all components.                               | TradeRecord, StrategyRiskConstraints, TradingStrategy                                                                                    | main                                                                                                                                                                                                                                          |
| `src/utils/__init__.py`                     | Utility helpers used by scripts and runtime tooling.                                | -                                                                                                                                        | -                                                                                                                                                                                                                                             |
| `src/utils/experiment.py`                   | Experiment/run metadata helpers for CLI scripts.                                    | -                                                                                                                                        | now_utc_iso, set_global_seed, _safe_git_commit, _load_manifest, _save_manifest, start_experiment_run, finalize_experiment_run                                                                                                                 |

## tests/

| File                                  | Module Doc                                                  | Top-Level Classes                                                                                                                                    | Top-Level Functions                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|---------------------------------------|-------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `tests/test_baselines.py`             | Tests for baseline models and walk-forward evaluation.      | TestRandomWalkBaseline, TestEWMABaseline, TestLightGBMBaseline, TestMetrics, TestWalkForward                                                         | sample_data                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `tests/test_costs_phase4.py`          | Phase 4 cost engine tests.                                  | -                                                                                                                                                    | test_transition_classification_legs, test_reverse_transition_fee_legs, test_funding_cashflow_sign_long_vs_short, test_margin_interest_accrual_uses_holding_and_time, test_execution_cost_audit_table_contains_all_components                                                                                                                                                                                                                                             |
| `tests/test_data_pipeline.py`         | Tests for the data pipeline.                                | TestLabelGenerator, TestAntiLeakage, TestFeatureComputation, TestPhase2DataPipeline                                                                  | sample_ohlcv, phase2_raw_slice, test_config_loads                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `tests/test_phase0_phase1.py`         | Tests for Phase 0/1 shared definitions and runtime tooling. | -                                                                                                                                                    | test_binance_futures_discounted_taker_fee, test_kucoin_spot_discounted_taker_fee, test_default_scenario_is_valid, test_profit_factor_and_decay_helpers, test_run_manifest_start_and_finalize                                                                                                                                                                                                                                                                             |
| `tests/test_phase10_paper_trading.py` | Phase 10 paper-trading readiness tests.                     | -                                                                                                                                                    | _sample_data, _run_replay, test_paper_trading_replay_uses_cost_engine_with_audit_columns, test_monitoring_dashboards_include_phase10_metrics, test_kill_switch_triggers_on_threshold_violations, test_kill_switch_ignores_inactive_windows, test_kill_switch_skips_pf_sharpe_when_window_trade_count_is_too_low, test_readiness_uses_recent_active_week_for_pf_check, test_readiness_and_capital_ramp_policy_promote_and_rollback, test_ramp_policy_text_mentions_stages |
| `tests/test_phase3_leakage.py`        | Phase 3 leakage and walk-forward boundary guardrail tests.  | -                                                                                                                                                    | sample_walkforward_data, _wf_config, test_walk_forward_enforces_strict_feature_lag, test_fold_boundaries_artifact_snapshot_written, test_generate_folds_rejects_duplicate_index                                                                                                                                                                                                                                                                                          |
| `tests/test_phase6_baselines.py`      | Phase 6 baseline protocol and leaderboard utility tests.    | -                                                                                                                                                    | _sample_result, test_baseline_protocol_fingerprint_stable, test_infer_feature_columns_filters_leakage_columns, test_freeze_protocol_and_fold_schedule, test_resolve_model_configs_injects_lgb_features, test_leaderboard_and_gate_artifacts                                                                                                                                                                                                                              |
| `tests/test_phase7_chronos.py`        | Phase 7 Chronos/meta validation tests.                      | _DummyChronos                                                                                                                                        | test_chronos_runner_strict_rolling_prediction_updates_from_predicted_q50, test_meta_model_uses_oof_chronos_then_final_full_fit, test_recent_regime_split_prefers_2024_anchor_and_candidate_gate_uses_ratio                                                                                                                                                                                                                                                               |
| `tests/test_phase8_robustness.py`     | Phase 8 robustness protocol tests.                          | -                                                                                                                                                    | _build_base_result, test_block_bootstrap_and_rolling_subperiod_protocols_produce_contiguous_stress_outputs, test_regime_exclusion_and_adverse_window_protocols_use_phase8_logic, test_run_all_includes_cost_grid_and_parameter_sweep_without_random_subsample, test_robustness_summary_requires_stress_pass_rate_for_viability                                                                                                                                           |
| `tests/test_phase9_reporting.py`      | Phase 9 reporting and decision framework tests.             | -                                                                                                                                                    | _build_result, test_decision_report_primary_pf_gate_drives_no_go, test_decision_report_go_when_required_checks_pass, test_uncertainty_bands_include_fold_and_block_bootstrap, test_save_decision_artifacts_writes_json_and_text, test_compare_models_uses_pf_first_selection, test_backtest_report_uses_shared_kill_criteria_text, test_decision_report_warning_is_not_treated_as_severe_fail, test_decision_report_low_win_rate_is_advisory_not_hard_fail               |
| `tests/test_strategy.py`              | Tests for strategy and backtest modules.                    | TestSignalGenerator, TestPositionSizer, TestExecutionIntentBuilder, TestRegimeDetector, TestCostModel, TestBacktestEngine, TestTradingStrategyPhase5 | sample_predictions, sample_data                                                                                                                                                                                                                                                                                                                                                                                                                                          |

## Class Method Index (Non-Empty Classes)

### `config/baseline_protocols.py`

- `BaselineModelSpec`: kwargs_dict
- `BaselineProtocol`: walk_forward_config, to_dict, fingerprint

### `config/cost_profiles.py`

- `MarketFeeProfile`: fee_rate
- `ExchangeCostProfile`: market, fee_rate

### `config/settings.py`

- `DataPaths`: raw, processed, features, ensure_dirs
- `WalkForwardConfig`: effective_train_days, effective_test_days, effective_step_days
- `Settings`: __post_init__

### `src/backtest/costs.py`

- `TransitionLegs`: traded_notional
- `TradeCosts`: __post_init__, zero
- `CostModel`: __init__, _classify_transition, _slippage_rate, _infer_bar_seconds, calculate_event_costs, calculate_execution_costs, calculate_costs, calculate_costs_series

### `src/backtest/engine.py`

- `BacktestResult`: summary, to_dict
- `BacktestEngine`: __init__, run, _calculate_results

### `src/backtest/report.py`

- `BacktestReport`: __init__, generate_summary, _format_regime_analysis, _check_kill_criteria, save_json, save_csv, save_report, save_all

### `src/data/binance_fetcher.py`

- `BinanceFetcher`: __init__, align_funding_to_4h

### `src/data/build_dataset.py`

- `DatasetBuilder`: __init__, _build_contract_metadata_frame, _normalize_liquidations, _assert_ohlcv_integrity, _assert_data_availability, generate_quality_report, _normalized_data_map, _save_raw_data, load_raw_data, compute_features, build_dataset, get_train_test_split

### `src/data/contracts.py`

- `IndexGapStats`: to_dict

### `src/data/labels.py`

- `LabelGenerator`: __init__, compute_forward_returns, compute_realized_volatility, compute_historical_quantiles, compute_regime_labels, generate_all_labels, validate_no_leakage, _validate_datetime_index, _infer_candle_step, _find_shifted_target_leaks

### `src/data/liquidation_collector.py`

- `AggregatedLiquidations`: total_liq_usd, liq_imbalance
- `LiquidationCollector`: __init__, _get_window_key, _parse_liquidation, _add_to_window, stop, estimate_from_oi_changes

### `src/data/macro_fetcher.py`

- `MacroFetcher`: __init__, fetch_ticker, fetch_all_macro, generate_event_flags, align_to_4h

### `src/data/market_metadata.py`

- `ContractMetadata`: to_dict

### `src/evaluation/metrics.py`

- `QuantileMetrics`: to_dict, __str__
- `TradingMetrics`: to_dict, __str__

### `src/evaluation/walk_forward.py`

- `WalkForwardResults`: compute_aggregates, summary
- `WalkForwardEvaluator`: __init__, _validate_datetime_index, _resolve_feature_columns, _prepare_supervised_frame, _assert_fold_no_contamination, _write_fold_boundaries_snapshot, generate_folds, evaluate_model, compare_models

### `src/models/baselines/ewma.py`

- `EWMABaseline`: __init__, name, fit, predict, update
- `ARBaseline`: __init__, name, fit, predict

### `src/models/baselines/lightgbm_quantile.py`

- `LightGBMQuantileBaseline`: __init__, name, _prepare_features, fit, predict, get_feature_importance

### `src/models/baselines/random_walk.py`

- `BaselineModel`: fit, predict, name
- `RandomWalkBaseline`: __init__, name, fit, predict, predict_single

### `src/models/chronos2_runner.py`

- `Chronos2Runner`: __init__, name, clone_unfitted, _resolve_device, _maybe_init_pipeline, _fit_covariate_adjuster, fit, _predict_quantiles_from_context, _covariate_shift, predict

### `src/models/meta_model.py`

- `MetaModel`: __init__, name, _ensure_chronos_template, _clone_chronos_unfitted, _get_chronos_features, _prepare_raw_features, _prepare_meta_features, _build_oof_splits, _generate_oof_chronos_predictions, fit, predict, get_feature_importance

### `src/paper_trading/engine.py`

- `PaperTradingReplay`: to_dict
- `PaperTradingEngine`: __init__, _resolve_start_index, _select_feature_columns, _build_regime_series, run

### `src/paper_trading/policy.py`

- `KillSwitchEvent`: to_dict
- `DeploymentReadiness`: to_dict
- `CapitalRampPolicy`: to_dict
- `CapitalRampDecision`: to_dict

### `src/reporting/decision.py`

- `UncertaintyBand`: to_dict
- `DecisionReport`: to_dict, to_text

### `src/robustness/kill_criteria.py`

- `CriterionResult`: passed
- `KillCriteriaResult`: all_passed, has_warnings, num_passed, num_failed, summary
- `KillCriteria`: __init__, check, _check_sharpe, _check_drawdown, _check_baseline_beat, _check_regime_stability, _check_win_rate, _check_profit_factor_net, _check_decay

### `src/robustness/stress_tests.py`

- `StressTestResult`: to_dict
- `StressTestSuite`: all_passed, pass_rate, to_dict, summary
- `StressTester`: __init__, run_all, _extract_net_returns, _clone_cost_model, _clone_signal_generator, _clone_position_sizer, _clone_engine, _run_engine, _test_cost_stress_grid, _test_regime_exclusion_protocol, _test_adverse_window_protocol, _test_block_bootstrap_stability, _test_rolling_subperiod_stability, _test_parameter_sensitivity

### `src/robustness/summary.py`

- `RobustnessReport`: to_dict
- `RobustnessSummary`: __init__, generate_report, generate_markdown, save_report

### `src/strategy/execution_intent.py`

- `ExecutionIntentBuilder`: __init__, _order_type_for_action, _side, build_for_positions

### `src/strategy/position_sizing.py`

- `PositionConstraints`: from_mapping
- `PositionSizer`: __init__, _effective_leverage_cap, _turnover_limited_target, _apply_turnover_cap, _enforce_short_constraints, _apply_order_constraints, calculate_size, calculate_sizes
- `KellyCriterionSizer`: __init__, calculate_kelly

### `src/strategy/regime_detector.py`

- `RegimeDetector`: __init__, detect_regime, detect_regimes, get_regime_multiplier, get_regime_multipliers

### `src/strategy/signals.py`

- `Signal`: is_trade
- `TradeDecision`: is_trade
- `QuantileSignalGenerator`: __init__, _validate_prediction_columns, _compute_uncertainty, _compute_confidence, _compute_strength, _resolve_entry_threshold_for_regime, build_forecast_snapshot, build_forecast_snapshots, decide_trade, generate_signal, generate_trade_decisions, generate_signals

### `src/strategy/strategy.py`

- `TradingStrategy`: __init__, _extract_position_constraints, _resolve_short_allowed, _apply_turnover_cap, _apply_cooldown_after_drawdown, _apply_scenario_risk_constraints, fit, generate_positions, calculate_returns

### `tests/test_baselines.py`

- `TestRandomWalkBaseline`: test_fit_predict, test_q50_is_zero, test_quantile_ordering
- `TestEWMABaseline`: test_fit_predict, test_captures_momentum
- `TestLightGBMBaseline`: test_fit_predict, test_feature_importance
- `TestMetrics`: test_pinball_loss_symmetric, test_quantile_metrics_coverage, test_trading_metrics_positive_sharpe
- `TestWalkForward`: test_fold_generation, test_evaluate_model

### `tests/test_data_pipeline.py`

- `TestLabelGenerator`: test_forward_returns_shape, test_forward_returns_last_is_nan, test_forward_returns_values, test_realized_volatility_shape, test_realized_volatility_last_is_nan, test_regime_labels_valid, test_generate_all_labels
- `TestAntiLeakage`: test_no_perfect_correlation, test_detect_obvious_leakage, test_detect_shifted_target_leakage, test_fail_on_forward_prefixed_feature, test_fail_on_unusable_forward_horizon_boundary
- `TestFeatureComputation`: test_return_features_no_nan_middle, test_missingness_flags_present
- `TestPhase2DataPipeline`: test_raw_contracts_accept_realistic_slice, test_raw_contracts_reject_naive_ohlcv_index, test_liq_provenance_defaults_for_legacy_liq_frames, test_build_dataset_calls_quality_report, test_ohlcv_gap_ratio_guardrail

### `tests/test_phase7_chronos.py`

- `_DummyChronos`: __init__, name, clone_unfitted, fit, predict

### `tests/test_strategy.py`

- `TestSignalGenerator`: test_generate_signals, test_long_signal_conditions, test_forecast_and_decision_layers, test_regime_adaptive_entry_thresholds
- `TestPositionSizer`: test_calculate_sizes, test_market_type_leverage_cap, test_precision_minimum_constraints, test_short_borrow_gating, test_turnover_cap
- `TestExecutionIntentBuilder`: test_hybrid_policy_mapping
- `TestRegimeDetector`: test_detect_regimes, test_regime_multipliers
- `TestCostModel`: test_calculate_costs, test_zero_costs_no_trade
- `TestBacktestEngine`: test_run_backtest
- `TestTradingStrategyPhase5`: test_strategy_risk_constraints, test_margin_borrow_and_drawdown_cooldown
