# 02 - Architecture And Pipeline

## System Architecture

```mermaid
flowchart LR
    subgraph Sources[External Sources]
        BFX[Binance Futures API]
        YF[yfinance Macro]
        WS[Binance ForceOrder WS]
    end

    subgraph DataLayer[src/data]
        BF[binance_fetcher.py]
        MF[macro_fetcher.py]
        LC[liquidation_collector.py]
        CT[contracts.py]
        DB[build_dataset.py]
        LB[labels.py]
        MM[market_metadata.py]
    end

    subgraph Storage[data]
        RAW[data/raw/*.parquet]
        PROC[data/processed/btc_4h.parquet]
        META[data/processed/*metadata*.json]
    end

    subgraph Modeling[src/models]
        RW[RandomWalk]
        EW[EWMA]
        LGB[LightGBM Quantile]
        CHR[Chronos2Runner]
        META2[MetaModel]
    end

    subgraph Strategy[src/strategy]
        SIG[QuantileSignalGenerator]
        REG[RegimeDetector]
        SIZ[PositionSizer]
        INT[ExecutionIntentBuilder]
        STRAT[TradingStrategy]
    end

    subgraph Eval[src/evaluation + src/backtest]
        WF[WalkForwardEvaluator]
        BT[BacktestEngine]
        COST[CostModel]
        REP[BacktestReport]
    end

    subgraph Governance[src/robustness + src/reporting + src/paper_trading]
        KILL[KillCriteria]
        STRESS[StressTester]
        DEC[DecisionReport]
        PAPER[PaperTradingEngine]
        POL[Kill Switch + Readiness + Capital Ramp]
    end

    subgraph Ops[scripts/*.py]
        S1[download_data.py]
        S2[build_features.py]
        S3[run_baselines.py]
        S4[run_chronos2.py]
        S5[run_backtest.py]
        S6[run_paper_trading.py]
        S7[benchmark.py]
        S8[smoke_check.py]
    end

    BFX --> BF
    YF --> MF
    WS --> LC

    BF --> DB
    MF --> DB
    LC --> DB
    CT --> DB
    MM --> DB
    LB --> DB

    DB --> RAW
    DB --> PROC
    DB --> META

    PROC --> WF
    PROC --> BT
    PROC --> PAPER

    WF --> RW
    WF --> EW
    WF --> LGB
    WF --> CHR
    WF --> META2

    RW --> SIG
    EW --> SIG
    LGB --> SIG
    CHR --> SIG
    META2 --> SIG

    SIG --> REG
    REG --> SIZ
    SIZ --> INT
    INT --> STRAT

    STRAT --> COST
    BT --> COST
    PAPER --> COST

    BT --> REP
    BT --> KILL
    BT --> STRESS
    STRESS --> DEC
    KILL --> DEC

    PAPER --> POL
    PAPER --> DEC

    S1 --> DB
    S2 --> DB
    S3 --> BT
    S4 --> BT
    S5 --> BT
    S6 --> PAPER
    S7 --> BT
    S8 --> WF
```

## End-To-End Pipeline

```mermaid
flowchart TD
    A[Phase 1: Download raw market/macro/liquidation inputs] --> B[Phase 2: Validate contracts and build feature set]
    B --> C[Phase 3: Generate labels and run leakage/boundary checks]
    C --> D[Phase 6: Freeze protocol and fold schedule]
    D --> E[Run baselines via walk-forward backtest]
    E --> F{Baseline PF/Sharpe gate pass?}
    F -- No --> G[Stop candidate advancement; report NO_GO/ITERATE]
    F -- Yes --> H[Phase 7: Run Chronos2 and MetaModel on frozen folds]
    H --> I[Phase 8: Stress suite]
    I --> J[Phase 9: Decision report + uncertainty bands]
    J --> K[Phase 10: Paper-trading replay with same cost engine]
    K --> L[Daily/weekly monitoring dashboards]
    L --> M[Kill switch + readiness checks]
    M --> N[Capital ramp recommendation]
```

## Architectural Design Characteristics

- Causality-first data and fold semantics: explicit feature lag and anti-contamination checks
- Net-cost-first evaluation: cost engine is central in both backtest and paper replay
- Gate-driven lifecycle: baselines gate candidate models; robustness gates viability; paper policy gates deployment
- Auditable artifacts: fold schedules, per-fold metrics, trade-level outputs, decision JSON/TXT, run manifests
