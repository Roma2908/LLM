flowchart LR
    %% 1. INGESTA ─────────────────────────────────────────
    subgraph STEP1["1️⃣ data_ingestion (✔)"]
        direction TB
        DI_OUT[«Dataset»\n(pd.DataFrame · memoria)]
    end

    %% 2. EDA UNIVARIADO ─────────────────────────────────
    subgraph STEP2["2️⃣ eda_univariado (✔)"]
        direction TB
        EU_OUT["eda_univariado/eda_univar_report.html"]
    end

    %% 3. VALIDACIÓN ─────────────────────────────────────
    subgraph STEP3["3️⃣ data_validation (✔)"]
        direction TB
        DV_OUT["data_validation/validation_report.json"]
    end

    %% 4. SPLIT ──────────────────────────────────────────
    subgraph STEP4["4️⃣ data_split (✔)"]
        direction TB
        DS_OUT["train / test / backtest\n(pd.DataFrame · memoria)"]
    end

    %% 5. SEARCH MODEL ───────────────────────────────────
    subgraph STEP5["5️⃣ search_model (✔)"]
        direction TB
        SM_RANK["model_ranking (DataFrame · mem)"]
        SM_BEST["best_model (objeto · mem)"]
        SM_PARAMS["best_params (dict · mem)"]
    end

    %% 6. SAVE MODEL ─────────────────────────────────────
    subgraph STEP6["6️⃣ save_model (✔)"]
        direction TB
        JOBLIB["models/model_v1.joblib"]
        ONNX["models/model_v1.onnx*"]
    end

    %% Flujo principal
    STEP1 --> STEP2 --> STEP3 --> STEP4 --> STEP5 --> STEP6

    %% Estilo opcional
    classDef opt fill:#fffbe6,stroke:#e0d98c,stroke-width:1px;
    class ONNX opt;

  * model_v1.onnx se crea solo si el modelo es convertible con skl2onnx; de lo contrario, se omite sin romper el flujo.
