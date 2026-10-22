"""Concurrency-aware process model forecasting with causal nets.

Modules, in the order the pipeline uses them:

    pipeline            the entry point, from an event log to the weekly models
    data                loading, splitting and windowing event logs
    relations           daily relation series (A, AB, ABA)
    bindings            daily binding series (IB, OB)
    forecasting         Chronos-2 forecasts
    weekly_models       the four model families of every evaluation window
    reconstruction      causal nets rebuilt from forecast or observed counts
    causal_net          causal nets from bindings, Fodina mining, and the JSON format
    structural_metrics  the AND-binding share and the structural-metrics CSV
    petri_net           the expansion of causal nets into WF-nets
    rendering           Graphviz drawings of the WF-nets
    conformance         fitness and precision
    precision           ETC and Align-ETC precision
    entropic_relevance  entropic relevance
    reduction           language-preserving τ-reduction
    guard               memory and time budget per evaluation unit
    families            the model families and their names
    paths               where a run's outputs go
    reproducibility     seeds and deterministic ordering
    fodina              the Fodina miner

Built apart from the pipeline:

    history_models      the history families of every evaluation window
"""
