"""Single source of truth for the Skrub DataOps coding guideline.

Tiered injection policy:
- Code-writing agents: append `SKRUB_DATAOPS_GUIDELINE`
- Debug/repair agents: append `SKRUB_DATAOPS_DEBUG_GUIDELINE` only (not the full guide)
- Plan-only agents: append `SKRUB_DATAOPS_PLAN_CONSTRAINT`
- Submission agent: append `SKRUB_SUBMISSION_CONSTRAINT`
- Agents with RAG tool access may also append `SKRUB_RAG_HINT`

Wording is intentionally explicit about the DataOps abstraction (`skrub.var`,
`.skb.mark_as_X/y`, `.skb.apply`, `.skb.apply_func`, `.skb.make_learner`,
`learner.predict(...)`) because the legacy `skrub.tabular_pipeline` API is not
the desired target.
"""

from machine_learning_engineering.shared_libraries.config import CONFIG

# Verbatim lines that mention `query_skrub_documentation`. Stripped from
# prompts at module load when `CONFIG.use_rag` is False, because the tool is
# not registered with any agent in that mode.
_RAG_HINT_LINE = (
    "- If you need more knowledge about skrub, use the query_skrub_documentation() tool to retrieve relevant code snippets and API descriptions from the local RAG vector database.\n"
)
_RAG_FAILSAFE_LINE = (
    "3. If an error occurs, you must analyze the traceback and immediately invoke the `query_skrub_documentation` tool with a highly specific query targeting the exact method or class that failed (e.g., \"Skrub DataOps train_test_split correct dictionary layout\").\n"
)
_RAG_FAILSAFE_REPLACEMENT = (
    "3. If an error occurs, you must analyze the traceback and rely on the Skrub DataOps guideline already injected into this prompt to fix the call site.\n"
)
_RAG_DEBUG_GUIDELINE_LINE = (
    "If you need help with the DataOps API, call the `query_skrub_documentation`\n"
    "tool (when available) before rewriting the script.\n"
)


def strip_rag_tool_mentions(text: str) -> str:
    """Remove `query_skrub_documentation` references when RAG is disabled."""
    if CONFIG.use_rag:
        return text
    text = text.replace(_RAG_HINT_LINE, "")
    text = text.replace(_RAG_FAILSAFE_LINE, _RAG_FAILSAFE_REPLACEMENT)
    text = text.replace(_RAG_DEBUG_GUIDELINE_LINE, "")
    return text

SKRUB_DATAOPS_GUIDELINE = """
# MANDATORY: Use the Skrub DataOps abstraction (not plain sklearn, not `skrub.tabular_pipeline`)

Every solution MUST be expressed as a Skrub DataOps computation graph. Do not
emit standalone `model.fit(X, y)`, `sklearn.pipeline.Pipeline`, `make_pipeline`,
or manual `train_test_split` on raw arrays. Do not use `skrub.tabular_pipeline`
or `skrub.preprocessing.*` (the latter does not exist).

## Canonical graph shape (follow this order)

1. **Load** — wrap inputs in `skrub.var` so the same graph replays at test time:
       # Option A: in-memory train table (typical Kaggle `./input` layout)
       df = skrub.var("data", pd.read_csv("./input/train.csv"))
       # Option B: externalized file path (preferred when test is a different file)
       file_path = skrub.var(name="file_path", value="./input/train.csv")
       df = file_path.skb.apply_func(pd.read_parquet)  # or pd.read_csv

2. **Mark target and features** — immediately after load, before any transforms.
   IMPORTANT: `"target"` below is a PLACEHOLDER. Replace every occurrence with the
   ACTUAL target column name from the task description (e.g. `income`,
   `median_house_value`, `Survived`). Never emit the literal string `"target"`
   unless that is genuinely the column name in the dataset.
       # For classification tasks with string/categorical labels, map them to numeric integers first:
       # y = df["target"].skb.apply_func(lambda x: x.map({{'class_0': 0, 'class_1': 1}}))
       # y = y.skb.mark_as_y()
       # For numeric target columns (e.g. regression):
       y = df["target"]
       y = y.skb.mark_as_y()
       X = df.drop(columns=["target"], errors="ignore")
       X = X.skb.mark_as_X()
   Start from all train.csv columns except the target. Do not drop other columns
   immediately after load — wait until the feature-engineering step below.
   IMPORTANT: Double-check column data types before mathematical operations or model inputs. If a column contains string values (like string sequences, categories, or text), you must NOT perform raw mathematical operations (addition, subtraction, etc.) directly on it. If you need features from it, extract numeric features using `.skb.apply_func` first, and exclude the raw string column from any direct mathematical addition.

3. **Feature engineering** on tracked nodes (see patterns below).
   You MAY drop columns here when they are superseded by derived features (e.g.
   drop raw lat/lon after creating `distance_km`) or are genuinely redundant
   after transforms. Do not drop columns pre-emptively before engineering them.

4. **Preprocess + model** — instantiate sklearn/skrub objects, then pass them to `.skb.apply`. Do NOT use sklearn `Pipeline`, `make_pipeline`, or `ColumnTransformer`. Instead, use `TableVectorizer` (the native Skrub equivalent of ColumnTransformer) or Skrub select, apply, and concat to preprocess different columns:
       # Option A: Automatic preprocessing using TableVectorizer (preferred for typical tabular datasets):
       from skrub import TableVectorizer
       X_encoded = X_features.skb.apply(TableVectorizer())
       pred = X_encoded.skb.apply(GradientBoostingClassifier(random_state=0), y=y)

       # Option B: Manual column-specific preprocessing:
       X_num = X_features.skb.select(skrub.selectors.numeric()).skb.apply(StandardScaler())
       X_cat = X_features.skb.select(skrub.selectors.categorical()).skb.apply(OneHotEncoder(handle_unknown='ignore', sparse_output=False))
       X_preprocessed = X_num.skb.concat([X_cat], axis=1)
       pred = X_preprocessed.skb.apply(GradientBoostingClassifier(random_state=0), y=y)

   ### Choosing a categorical encoder (use skrub/sklearn native — never `category_encoders`)
   `TableVectorizer()` already handles this automatically: low-cardinality columns
   go to `OneHotEncoder`, high-cardinality ones to `StringEncoder`. Prefer it. If
   you want explicit control, all of these work inside `.skb.apply(...)`:
   - Low cardinality: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
   - High cardinality / dirty strings: `skrub.StringEncoder()`, `skrub.GapEncoder()`,
     `skrub.MinHashEncoder()`, or `skrub.SimilarityEncoder()`
   - Target/CatBoost-style encoding: `sklearn.preprocessing.TargetEncoder()`
     (cross-fitted and leakage-safe — the native replacement for
     `category_encoders.CatBoostEncoder`)
   Do NOT import `category_encoders`; skrub and sklearn cover every case natively
   and it is not an installed dependency.

   ### Models handling of string/categorical features (e.g. CatBoost)
   If you pass the features to a model like `CatBoostClassifier`/`CatBoostRegressor` (or custom wrapper estimator) instead of preprocessing the entire table with `TableVectorizer()` beforehand:
   - CatBoost will fail with a float conversion error if it encounters any string/object columns (like text or categorical columns) that have not been explicitly declared as categorical features (`cat_features=[...]`) or text features (`text_features=[...]`).
   - If you concat engineered text features (e.g. `X_seq` containing a joined string sequence) with the original features, you MUST drop the original string/object columns or list them in `cat_features`/`text_features` before passing the DataFrame to CatBoost. Failing to do so will cause a crash because CatBoost will treat the undeclared string columns as numeric.

5. **Evaluate** — split, learner, and metrics all come from the last DataOp:
       from sklearn.metrics import mean_squared_error
       learner = pred.skb.make_learner()
        # IMPORTANT: Always call train_test_split on the final combined prediction/ensemble node (e.g. `pred`).
        # Do NOT split `y` or `X` separately (e.g. `y.skb.train_test_split` is invalid and will crash), and NEVER split a raw feature/data node (e.g. `X_fe.skb.train_test_split`).
        data = pred.skb.train_test_split(test_size=0.2, random_state=0)
        learner.fit(data["train"])                         # IMPORTANT: Only pass the single "train" environment dict, never pass data["y_train"] as a second argument!
        y_pred = learner.predict(data["test"])             # predictions come ONLY from learner.predict

       y_true = data["y_test"]                             # ground-truth labels come ONLY from this key
       final_validation_score = np.sqrt(mean_squared_error(y_true, y_pred))
       print(f"Final Validation Performance: {{final_validation_score}}")

   ### CRITICAL: how to read predictions and labels back out of the split
   `data["train"]` and `data["test"]` are ENVIRONMENT dicts you FEED to the learner
   (`learner.fit(data["train"])`, `learner.predict(data["test"])`). They are NOT
   result frames — do NOT index them for predictions or labels. The keys
   `_skrub_predict` and `_skrub_y` do NOT exist inside `data["test"]`; indexing
   them (e.g. `data["test"]["_skrub_predict"]`, `data["test"]["_skrub_y"]`) raises
   `KeyError: '_skrub_predict'`.
   - Predictions: `y_pred = learner.predict(data["test"])` — never `data["test"][...]`.
   - Ground-truth labels: `data["y_test"]` (top-level split key), never `data["test"]["_skrub_y"]`.
   # split dict keys: "train", "test", "y_train", "y_test"
   # Note: data["y_test"] is a pandas Series. If it contains string target labels (e.g. classification), you must map them to numeric integers (e.g. 0/1) before calculating performance metrics. Never call .astuple() on it, and never cast raw string targets using .astype(int).

## Multi-target tasks (task predicts MORE THAN ONE target column)

Some tasks require predicting several target columns at once (e.g.
`y1` AND `y2`). There are TWO valid
patterns — pick based on the estimator you are using:

**Pattern A — one multi-output learner (preferred when the estimator natively
supports 2-D `y`):** mark ALL target columns as `y`, apply the estimator once,
and split the multi-column prediction by column.
    y = df[["target_a", "target_b"]].skb.mark_as_y()
    X = df.drop(columns=["target_a", "target_b"], errors="ignore").skb.mark_as_X()
    pred = X_features.skb.apply(TableVectorizer()).skb.apply(model, y=y)
    learner = pred.skb.make_learner()
    # test_preds is an (n, 2) array → split columns: test_preds[:, 0], test_preds[:, 1]
This works ONLY if the estimator accepts 2-D `y` — native multi-output models
(`RandomForestRegressor`, `ExtraTreesRegressor`, `Ridge`) or any model wrapped in
`sklearn.multioutput.MultiOutputRegressor`.

**Pattern B — one learner per target (required when the estimator is
single-output):** if your custom estimator calls `np.asarray(y).ravel()`,
rank-transforms `y`, or otherwise assumes a single column, Pattern A will
misalign and crash. Train and KEEP one learner per target in a dict:
    learners = {{}}
    for target in target_cols:
        yt = df[target].skb.mark_as_y()
        pred = X_features.skb.apply(TableVectorizer()).skb.apply(model, y=yt)
        learner = pred.skb.make_learner()
        learner.fit(data["train"])
        learners[target] = learner            # keep EVERY learner, not just the last
Print the per-target metric AND their mean (multi-target leaderboards usually
average a per-column metric).

**Invariant for BOTH patterns:** every target column in the final submission must
be produced by a model actually trained on THAT target. The classic bug is a
`for target in target_cols:` loop that rebinds a single `learner`/`pred` variable
each iteration — after the loop only the LAST target's model survives, the others
are silently lost, and the submission writes one model's predictions into every
target column. NEVER reuse one target's learner or predictions for a different
target column, and NEVER assign the same prediction array to two columns.

## Feature-engineering patterns (critical)

**Double check column datatypes before numeric operations:**
- Never assume a column is numeric just because of its name or position. Always verify the task description schema.
- If a column contains string values (e.g. string sequence, characters, category names), you must NOT perform raw mathematical operations on it directly (such as adding it to numeric features: `f07 + f27` where `f27` is a string sequence).
- If you need features from a string/categorical column, extract numeric representations (like string length, counts of characters, ordinal sums) using `.skb.apply_func` first, and ensure the raw string column itself is excluded from any direct mathematical addition or numeric transformations.

**Arithmetic on tracked columns is valid** — stay in the graph, no UDF needed:
    d_lat = lat2 - lat1
    interaction = distance_km * X_features["passenger_count"]

**`.assign()` adds columns** — preferred for straightforward feature creation:
    X_fe = X_features.assign(distance_km=distance_km, hour_sin=hour_col.skb.apply_func(np.sin))

**`.skb.apply_func(fn)`** — for numpy/pandas functions on Series or chained ops:
    # IMPORTANT: Never pass a DataOp (including compound expressions) directly to a raw numpy/pandas function
    # (e.g. np.sqrt(x), np.log(x), np.abs(x)). Always wrap the function using .skb.apply_func.
    # This applies ONLY to named numpy/pandas FUNCTIONS. Arithmetic operators
    # (+ - * /), comparisons (< > == etc.), and .dt/.str accessors already work
    # DIRECTLY on DataOps and return DataOps — do NOT wrap them. In particular,
    # NEVER write an identity wrapper like `(a / b).skb.apply_func(lambda x: x)`:
    # `a / b` is already a valid DataOp on its own.
    #   CORRECT:   ratio = X["total_rooms"] / X["households"]
    #   WRONG:     ratio = (X["total_rooms"] / X["households"]).skb.apply_func(lambda x: x)
    cos_val = angle.skb.apply_func(np.cos)
    sqrt_val = (x_col * x_col + y_col * y_col).skb.apply_func(np.sqrt)
    date_col = X["pickup_datetime"].skb.apply_func(pd.to_datetime)
    month_sin = (date_col.dt.month * 2 * np.pi / 12).skb.apply_func(np.sin)

**Custom transforms** — write a function that takes a DataOp and returns a DataOp:
    def add_distance(data_op):
        lat1 = data_op["pickup_latitude"]
        lat2 = data_op["dropoff_latitude"]
        # ... haversine math using .skb.apply_func where needed ...
        return data_op.assign(distance_km=distance_km)

    X_dist = add_distance(X_features).drop(
        columns=["pickup_latitude", "pickup_longitude", ...], errors="ignore"
    )

**Datetime cyclical encoding** — parse once, then sin/cos via `.skb.apply_func`:
    date_col = X["timestamp"].skb.apply_func(pd.to_datetime)
    hour_col = date_col.dt.hour * 2 * np.pi / 24
    X_dt = X_scaled.assign(
        hour_sin=hour_col.skb.apply_func(np.sin),
        hour_cos=hour_col.skb.apply_func(np.cos),
    )

**Join feature blocks** with `.skb.concat`:
    X_final = X_scaled.skb.concat([date_features], axis=1)

**Column subsets** with `.skb.select`:
    X_numeric = X.skb.select(skrub.selectors.numeric())

## CRITICAL: the DataOp-vs-pandas boundary inside functions

There are TWO different function patterns and they receive DIFFERENT objects.
Mixing them up is the most common error — read this carefully:

- A function passed to **`.skb.apply_func(fn)`** receives a CONCRETE pandas
  object (DataFrame/Series), NOT a DataOp. Inside it you may use ONLY plain
  pandas/numpy. NEVER call `.skb`, `skrub.selectors`, `.skb.apply`, or
  `.skb.select` inside such a function — the concrete object has no `.skb`
  namespace and it will raise `AttributeError: 'DataFrame' object has no
  attribute 'skb'`.
      # CORRECT — plain pandas inside apply_func:
      def freq_encode(s):            # s is a concrete pandas Series
          return s.map(s.value_counts(normalize=True))
      encoded = X["city"].skb.apply_func(freq_encode)

- A function that USES `.skb.*` must take a DataOp and be called DIRECTLY on
  the DataOp (`build_preprocessor(X_features)`), NEVER wrapped in
  `.skb.apply_func`:
      # CORRECT — DataOp in, DataOp out, called directly:
      def build_preprocessor(X_op):          # X_op is a DataOp
          num = X_op.skb.select(skrub.selectors.numeric()).skb.apply(StandardScaler())
          cat = X_op.skb.select(skrub.selectors.categorical()).skb.apply(TableVectorizer())
          return num.skb.concat([cat], axis=1)
      X_preprocessed = build_preprocessor(X_features)   # direct call, NOT apply_func

Keep UDFs small; prefer `.assign` / `.skb.apply_func` chains over one large function.

## Row filtering that must not run at predict time

When filtering bad rows (outliers, invalid coords, negative targets), use
`skrub.eval_mode()` so raw test data passes through unchanged:

    def clean_rows(df):
        mask = (df["target"] >= 0) & (df["lat"].between(40.5, 41.8))
        filtered = df[mask].reset_index(drop=True)
        return (skrub.eval_mode() != "predict").skb.if_else(filtered, df)

    df = clean_rows(df)

## Hyperparameter / feature search inside the graph

Use `skrub.choose_from` for alternatives, then `.as_data_op()` to re-enter the graph:

    X_variant = skrub.choose_from(
        {{"basic": X_v1, "extended": X_v2}},
        name="features",
    ).as_data_op()

    pred = skrub.choose_from(
        {{
            "linear": X_final.skb.apply(LinearRegression(), y=y),
            "ridge": X_final.skb.apply(Ridge(alpha=1.0), y=y),
        }},
        name="models",
    ).as_data_op()

    from sklearn.model_selection import ShuffleSplit
    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    results = pred.skb.make_grid_search(cv=cv, scoring="r2", fitted=True, refit=False)

Cross-validate a single graph node (no grid) with:
    score = pred.skb.cross_validate(cv=cv, scoring="r2")

Subsample for fast search: `df = df.skb.subsample(5000)`

## How to fit/train on the full training set (without validation split)

If you want to train on 100% of the training data (e.g. for the final solution script), do NOT pass symbolic Skrub DataOp objects (like `X`, `y`, `X_features`, etc.) in the environment dictionary. Passing an empty dictionary {{}} will fail because default variable values are not automatically resolved in fit mode. Instead, pass the concrete training DataFrame/Series under the corresponding variable name or `_skrub_X` / `_skrub_y` placeholders:

# Option A: Pass concrete training data to the variable name used in the graph
train_df = pd.read_csv("./input/train.csv")
learner.fit({{"data": train_df}})  # if "data" var was used
# or: learner.fit({{"file_path": "./input/train.csv"}})  # if "file_path" var was used

# Option B: Pass concrete feature and target placeholders directly
train_df = pd.read_csv("./input/train.csv")
train_X = train_df.drop(columns=["target"])
train_y = train_df["target"]
learner.fit({{"_skrub_X": train_X, "_skrub_y": train_y}})

# INCORRECT (will raise TypeError - passing symbolic DataOps):
learner.fit({{"_skrub_X": X_features, "_skrub_y": y}})  # X_features and y are DataOps!

## Test-time prediction (pick the pattern that matches your graph)

**A. Graph loaded via a file-path variable** — pass an env dict with the same key:
    y_test_pred = learner.predict({{"file_path": "./input/test.csv"}})

**B. Graph used `.skb.mark_as_X()`** — pass the raw feature table (no manual preprocessing):
    test_df = pd.read_csv("./input/test.csv")
    y_test_pred = learner.predict({{"_skrub_X": test_df}})

**C. Graph used `skrub.var("data", train_df)` without `mark_as_X`** — pass under that name:
    y_test_pred = learner.predict({{"data": test_df}})

Do NOT create a second `skrub.var` for test data. Do NOT re-run preprocessing
manually; the learner replays the full graph.

## Banned patterns (treat as failures, not options)

- `model.fit(X_train, y_train)` / `model.predict(X_val)` outside `.skb.apply`
- `scaler.fit_transform(X_train)` / `scaler.transform(X_val)` outside `.skb.apply`
  (instantiating `StandardScaler()` and passing it to `.skb.apply(scaler)` is correct)
- `from sklearn.pipeline import Pipeline`, `make_pipeline`, or `ColumnTransformer` (strictly banned; use Skrub select, apply, and concat instead)
- `vectorizer.fit_transform(some_tracked_node)` (must be inside `.skb.apply(...)`)
- sklearn.model_selection.train_test_split(X, y) on raw arrays or DataFrames
- Calling `.skb.train_test_split()` separately on target `y` or features `X` (always split the combined prediction/ensemble node)
- Passing `stratify` keyword argument (like `stratify=y` or `stratify=y.skb.apply_func(...)`) to `.skb.train_test_split()`. The `stratify` keyword argument is NOT supported on lazy DataOp nodes and will crash with an error containing `na_value`. Simply omit `stratify` or use a standard non-stratified split.
- Eagerly iterating over or checking elements in `df.columns` or `X.columns` (e.g. `[c for c in X.columns]` or `if col in X.columns`), because columns on a `DataOp` are lazy nodes and not concrete lists. Use `.drop(columns=[...])` or `.skb.select()` to modify columns.
- Calling `.to_pandas()` or pandas DataFrame methods on `data["train"]` or `data["test"]` (they are environment dicts, not DataFrames).
- `skrub.tabular_pipeline(...)`, `skrub.preprocessing.*`
- Importing from private modules with underscores, e.g. `skrub._dataframe`
- Calling `.skb`, `.skb.apply`, `.skb.select`, or `skrub.selectors` on the
  argument of a function passed to `.skb.apply_func(...)` (that argument is a
  concrete pandas object, not a DataOp — see the DataOp-vs-pandas boundary above)
- Inventing constructor arguments for skrub/sklearn objects. In particular,
  `TableVectorizer()` has NO `scaler=` argument. To scale numerics, pass
  `TableVectorizer(numeric=StandardScaler())` or apply `StandardScaler()` to the
  numeric block separately. If unsure of a class's signature, instantiate it with
  defaults rather than guessing keyword arguments.
- Inventing data loaders such as `skrub.datasets.*` for the local `./input` data;
  always load with `pd.read_csv("./input/train.csv")`.
- Importing `category_encoders` (e.g. `CatBoostEncoder`). Use skrub's native
  encoders (`TableVectorizer`, `StringEncoder`, `GapEncoder`, `MinHashEncoder`,
  `SimilarityEncoder`) or `sklearn.preprocessing.TargetEncoder` instead.
- Hard-coded row filters without `skrub.eval_mode().skb.if_else(...)` when the
  same graph must predict on unfiltered test rows
- Calling raw numpy/pandas functions directly on DataOp objects (e.g., np.sqrt(data_op), np.log(data_op), np.abs(data_op)) instead of wrapping them in `.skb.apply_func`
- Wrapping plain arithmetic (`+ - * /`), comparisons, or `.dt`/`.str` accessors in `.skb.apply_func` — these already work directly on DataOps. In particular, NEVER add a no-op identity wrapper like `.skb.apply_func(lambda x: x)`: a bare expression such as `X["a"] / X["b"]` is already a valid DataOp and needs no wrapping.
- Passing symbolic Skrub DataOp objects (such as `X`, `y`, `X_features`, `df`, etc.) inside environment dictionaries to `learner.fit()`, `learner.predict()`, or `learner.score()`. These dictionaries must only contain concrete Python/pandas objects (like DataFrames, Series, or strings).
- Loading test data (e.g. `test.csv`) or generating a submission file (e.g. `submission.csv`). The solution script must ONLY load the train data, train the model, evaluate validation score, and print the validation metric. Do NOT include any test-prediction or submission generation code in this script. Focus exclusively on validation performance. Submission generation is handled in a separate phase at the end of the pipeline.
- Printing model training logs or iteration progress (e.g. setting `verbose=True`, `verbose=100`, etc. for XGBoost, LightGBM, CatBoost, or other estimators). You must ALWAYS set `verbose=False` or `verbose=0` to keep execution outputs completely silent. Verbose training prints accumulate in the conversation history and will exceed the context length limit, causing the campaign to crash.
- Accessing pandas attributes directly on Skrub DataOp node variables (such as calling `df.columns`, `df.index`, `df.dtypes`, or using double brackets `df[[...]]` to select columns). A Skrub DataOp variable (e.g. created by `skrub.var()`) does NOT have these pandas attributes. To select or drop columns on the graph, use `df.drop(columns=[...], errors="ignore")` or retrieve the column names from the raw pandas DataFrame before wrapping it in `skrub.var()`.

## When integrating, ensembling, or combining models

Stay inside the DataOps graph by implementing a custom Scikit-learn estimator class (e.g. an averaging or stacking regressor/classifier). This is the cleanest approach because it avoids fit/predict mode-duality errors on graph nodes. Do NOT use `@skrub.deferred` or `.skb.apply_func` to combine model prediction nodes directly.

CRITICAL — apply the ensemble estimator to ONE node:
- Do NOT build two separate `.skb.apply(model)` prediction branches and then try
  to combine them. A tuple of DataOps has NO `.skb` namespace, so
  `(branch_a, branch_b).skb.apply(...)` raises `AttributeError`.
- Instead, the custom ensemble estimator must hold BOTH sub-models, fit them
  internally inside its own `fit`, and be applied ONCE to the single preprocessed
  feature node: `pred = X_preprocessed.skb.apply(ensemble_model, y=y)`.

CRITICAL — custom-estimator semantics:
- `fit(self, X, y)` does ALL training. `predict`/`transform` must NEVER call
  `.fit`/`.fit_transform`, and must NEVER reference `y` (it is not available at
  predict time — doing so causes leakage and runtime errors).
- Match the mixin to the task: `ClassifierMixin` for classification,
  `RegressorMixin` for regression. Do NOT inherit `RegressorMixin` for a
  classifier (or vice versa) even if you also set `self._estimator_type`.
- Never mutate constructor parameters in `fit` (e.g. do NOT do `self.model_a.fit(X, y)` directly). In scikit-learn, constructor parameters must remain unfitted and unmutated so they can be cloned or have `set_params` called safely. Instead, import `clone` from `sklearn.base`, clone the estimators in `fit`, and store the fitted models as new attributes ending with an underscore (e.g., `self.model_a_ = clone(self.model_a).fit(X, y)` and `self.model_b_ = clone(self.model_b).fit(X, y)`). Use these fitted clones (e.g., `self.model_a_`) inside `predict`/`predict_proba`.


Example of a custom averaging estimator applied in the graph for regression tasks:

    from sklearn.base import BaseEstimator, RegressorMixin, clone

    class EnsembleRegressor(BaseEstimator, RegressorMixin):
        def __init__(self, model_a, model_b):
            self.model_a = model_a
            self.model_b = model_b

        def fit(self, X, y):
            self.model_a_ = clone(self.model_a).fit(X, y)
            self.model_b_ = clone(self.model_b).fit(X, y)
            return self

        def predict(self, X):
            return 0.5 * self.model_a_.predict(X) + 0.5 * self.model_b_.predict(X)

    ensemble_model = EnsembleRegressor(model_original, model_expanded)
    pred_ensemble = X_features.skb.apply(ensemble_model, y=y)
    final_learner = pred_ensemble.skb.make_learner()

Example of a custom averaging estimator applied in the graph for classification tasks:

    from sklearn.base import BaseEstimator, ClassifierMixin, clone

    class EnsembleClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, model_a, model_b):
            self.model_a = model_a
            self.model_b = model_b
            self.classes_ = None

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            self.model_a_ = clone(self.model_a).fit(X, y)
            self.model_b_ = clone(self.model_b).fit(X, y)
            return self

        def predict_proba(self, X):
            proba_a = self.model_a_.predict_proba(X)
            proba_b = self.model_b_.predict_proba(X)
            return 0.5 * proba_a + 0.5 * proba_b

        def predict(self, X):

            probas = self.predict_proba(X)
            return self.classes_[np.argmax(probas, axis=1)]

    ensemble_model = EnsembleClassifier(model_original, model_expanded)
    pred_ensemble = X_features.skb.apply(ensemble_model, y=y)
    final_learner = pred_ensemble.skb.make_learner()
"""

SKRUB_DATAOPS_DEBUG_GUIDELINE = """
# DEBUG FAIL-SAFE: do NOT solve errors by removing Skrub DataOps

If the failing code uses Skrub DataOps (`skrub.var`, `.skb.*`), the fix MUST
remain inside the DataOps graph. Stripping out Skrub and falling back to plain
`model.fit(X, y)` / `sklearn.Pipeline` / manual `train_test_split` is treated
as a fatal regression, even if it makes the traceback go away.

Make the MINIMAL change that fixes the reported traceback. The following are
NOT bugs — preserve them verbatim, do not "improve" or strip them while fixing:
- the model family / algorithm (do NOT swap e.g. XGBoost for RandomForest),
- the feature-engineering and preprocessing logic,
- the target column name already used in the working parts of the script,
- the XAI instrumentation block (the `import xai_probes`,
  `xai_probes.run_leakage_suite(...)` call, and any `.skb.set_name("xai_features")`
  / `.skb.set_name("xai_model")` tags on the graph). Never delete these to clear
  an error; fix the failing line in place. `xai_probes` is a local module staged
  next to the script — do NOT add a `pip install` guard for it.
Changing any of these to dodge an error is a regression even if the script then
runs.

Make the SMALLEST edit that clears the reported traceback. Change only the
line(s) named in the error; re-emit every other line verbatim. Do NOT rewrite
the script from scratch, re-architect a section that already runs, or switch to a
different paradigm (`sklearn.pipeline.make_pipeline`, `model.fit(X, y)` on raw
arrays). A full rewrite is how working solutions silently regress.

NEVER invent imports to make an error go away. If the traceback is an
`ImportError` / "cannot import name X", that symbol does not exist in the
installed version — do NOT guess another path for it (e.g. private modules like
`skrub._utils`, or `skrub.preprocessing.*`) and do NOT import a symbol you are
not certain ships in the installed package. Remove the bad import and use only
public, documented skrub/sklearn APIs. Only import names you know exist.

An error message that names a skrub function is telling you that function was
used WRONGLY — it is never a signal to switch to pandas/sklearn. Replacing
`.skb.concat` with `pd.concat`, or `.skb.apply`/`.skb.make_learner` with
`model.fit`/`train_test_split`, is ALWAYS the wrong fix and a fatal regression,
even when the traceback mentions the skrub call.

Common real fixes that DO preserve DataOps:
- `skrub.concat(...)` AttributeError: `concat` is a METHOD on a DataOp node, not a
  module function. Use `first_node.skb.concat([other_node], axis=1)`. Do NOT
  "fix" it with `pd.concat(...)` — pandas concat does not work on DataOp nodes.
- Combining/stacking/averaging MODEL predictions: do NOT build separate prediction
  nodes (`pred1 = X.skb.apply(model_a, y=y)`, `pred2 = ...`) and then try to
  combine them with `.skb.concat` / `pd.concat` / `np.column_stack`. Prediction
  nodes are single-column Series and cannot be stacked that way. Instead apply ONE
  custom estimator (that trains the base models AND the meta-learner inside its
  own `fit`) to the FEATURE matrix:
  `stacked = X_features.skb.apply(StackedEnsemble(base_models=[...], meta=...), y=y)`.
- Wrong import: use `skrub.TableVectorizer`, not `skrub.preprocessing.TableVectorizer`.
- `AttributeError` on a tracked node: wrap the call in `.skb.apply_func(...)` or
  `.skb.apply(<BaseEstimator>)` instead of calling sklearn/numpy directly on the node.
- Feature math on columns: use normal arithmetic (`a - b`, `a * b`) or chain
  `.skb.apply_func(np.sin)` — do not call `np.sin(series)` or `np.sqrt(series)`
  directly on any tracked Series or compound expressions without `.skb.apply_func` (always wrap them, e.g. `(a * a + b * b).skb.apply_func(np.sqrt)`).
- `choose_from` result used as a node: call `.as_data_op()` before `.skb.apply` or
  `.skb.make_learner()`.
- Need a train/val split: `last_node.skb.train_test_split(...)`, not
  `sklearn.model_selection.train_test_split` on raw arrays. Use keys
  `data["train"]`, `data["test"]`, `data["y_train"]`, `data["y_test"]`.
  Always call `.skb.train_test_split(...)` on the final prediction/ensemble node (e.g. `pred`), NEVER on a feature/data node (like `X_fe` or `X`).
  If you see `ValueError: DataOp should have a node marked with "mark_as_X()"`, it is because you called `.skb.train_test_split()` on a target/feature node directly; instead, ensure you call it on the final prediction/ensemble node.
- `learner.fit(...)` argument mismatch / KeyError: 'y_train': Never pass a second argument to `learner.fit()` (such as `learner.fit(data["train"], data["y_train"])`). The learner's fit method only accepts a single argument: the training split environment dictionary (e.g. `learner.fit(data["train"])`). The target labels are already tracked inside the lazy DataOps graph.

- `KeyError: '_skrub_predict'` (or `'_skrub_y'`) at the metric line: the script tried
  to index the split environment, e.g. `data["test"]["_skrub_predict"]` /
  `data["test"]["_skrub_y"]`. Those keys do not exist — `data["test"]` is fed to the
  learner, not indexed. Replace with `y_pred = learner.predict(data["test"])` for the
  predictions and `data["y_test"]` for the ground-truth labels, then compute the metric
  on those two (e.g. `np.sqrt(mean_squared_error(data["y_test"], y_pred))`).
- Test-time prediction: match how the graph was built —
  `learner.predict({{"file_path": "..."}})` if load used a path var;
  `learner.predict({{"_skrub_X": test_df}})` if `.skb.mark_as_X()` was used.
  The env-dict value is a raw `pd.read_csv(...)` DataFrame. Do NOT wrap test data
  in a new `skrub.var(...)`, do NOT re-`mark_as_X()` it, and do NOT call
  `.skb.eval()` to manufacture the value — pass the DataFrame directly.
- Row filters that drop test rows: wrap with
  `(skrub.eval_mode() != "predict").skb.if_else(filtered_df, df)`.
- Premature submission generation (TRAINING/VALIDATION scripts ONLY): If a *training/validation* script contains test data loading (e.g. `test.csv`) or submission file writing (e.g. `submission.csv`), remove it entirely and focus on training + validation score. This does NOT apply to a submission script whose job is to produce `./final/submission.csv` — there, test loading and submission writing are REQUIRED; keep them and predict via the existing learner (`learner.predict({{...}})`), never by re-running preprocessing by hand.
- String target columns in classification: If the target column (accessed as `data["y_test"]` / `data["y_train"]` from split) contains string/categorical labels, map them to numeric integers (e.g. `data["y_test"].map({{'class_0': 0, 'class_1': 1}})` ) before calculating validation performance metrics. Never call `.astuple()` on a Series object, and never cast raw string target labels directly with `.astype(int)` (as this will raise a ValueError).
- AttributeError: 'numpy.dtypes.Int64DType' object has no attribute 'na_value' or Evaluation of '.na_value' failed: this is caused by passing `stratify=y` (or another `DataOp` node) to `.skb.train_test_split(...)`. Skrub's `train_test_split` does not support `stratify` on lazy `DataOp` nodes. To fix it, remove the `stratify` keyword argument from `.skb.train_test_split(...)`.
- TypeError: This object is a DataOp ... it is not possible to eagerly iterate over it now or eagerly use its Boolean value now: this is caused by trying to iterate over `df.columns` / `X.columns` or checking column membership (e.g. `c in X.columns`) in a list comprehension. Columns of a `DataOp` are lazy and cannot be iterated eagerly. To fix it, use `.drop(columns=[...])` or `.skb.select()` instead of iterating `df.columns`.
- AttributeError: 'dict' object has no attribute 'to_pandas' when trying to use `X_train.to_pandas()`: `X_train` or `data["train"]` are environment dictionaries, not DataFrames. Do not call `.to_pandas()` on them.
- TypeError: operation 'radd' not supported for dtype 'str' with dtype 'float64': this is caused by performing arithmetic addition/subtraction/math on a column containing string data (like a string sequence or category name). To fix this, inspect the columns in the addition, find the one containing strings (e.g. a character sequence feature), and exclude it from the addition or convert/extract numeric features from it first using `.skb.apply_func()`.
- CatBoost float conversion error / string features crash: when fitting `CatBoostClassifier` or `CatBoostRegressor` (or custom wrappers) on a DataFrame `X`, CatBoost crashes if any columns are strings/objects unless they are explicitly declared in `cat_features` or `text_features`. To fix this, either drop all original string/object columns from the DataFrame passed to the model, preprocess the DataFrame using `TableVectorizer()` before fitting, or declare all remaining string/object columns in `cat_features`/`text_features` of the model.
- CatBoostError: You can't change params of fitted model / clone / set_params error: in custom ensemble or stacking estimators, this is caused by calling `.fit()` directly on the sub-estimator parameter (e.g. `self.model_a.fit(...)`), which mutates it in-place. Because scikit-learn standard requires constructor parameters to remain unmutated, call `clone(estimator)` on each sub-model in `fit()` and assign the fitted instance to an attribute with a trailing underscore (e.g. `self.model_a_ = clone(self.model_a).fit(...)`), then use the fitted underscore-named attribute (e.g. `self.model_a_`) in `predict()` / `predict_proba()`.



Even if you conclude the code is already correct or "needs no changes", you MUST
still re-emit the COMPLETE script in a single ```python block. Prose-only replies
("no changes needed", "the script has been revised", status summaries) are
discarded and waste a debug round — never reply with prose instead of the script.

If you need help with the DataOps API, call the `query_skrub_documentation`
tool (when available) before rewriting the script.
"""

SKRUB_DATAOPS_PLAN_CONSTRAINT = """
# Skrub DataOps plan constraint

- All proposed changes MUST stay inside the existing Skrub DataOps graph (`skrub.var`, `.skb.*`).
- Do NOT propose reverting to plain sklearn (`model.fit(X, y)`, `sklearn.Pipeline`) or using sklearn Pipelines/ColumnTransformers. Use Skrub select, apply, and concat instead.
- When combining models, describe implementing a custom Scikit-learn estimator class to average/stack predictions and applying it via `.skb.apply(ensemble_model, y=y)`. Do NOT use `@skrub.deferred` or `.skb.apply_func` to combine model prediction nodes directly.
- NEVER build separate prediction nodes (`pred1 = X.skb.apply(model_a, y=y)`, `pred2 = X.skb.apply(model_b, y=y)`) and then concatenate/stack them with `.skb.concat`, `pd.concat`, or `np.column_stack` — prediction nodes are single-column and cannot be combined that way. The custom estimator must train the base models AND the meta-learner inside its own `fit`, and be applied ONCE to the feature matrix: `X_features.skb.apply(StackedEnsemble(...), y=y)`.
- PRESERVE THE FULL TARGET SET. If the current solution marks more than one target column as `y` (e.g. `y = df[["target_a", "target_b"]].skb.mark_as_y()`), the task is multi-target: every plan and every extracted code block MUST keep predicting ALL of those targets. Do NOT narrow `y` to a single column, and do NOT drop a target, even when the change you propose is only about feature engineering, hyperparameters, or the model. Keep marking all targets at once and use a natively multi-output estimator or one wrapped in `sklearn.multioutput.MultiOutputRegressor`. Never reuse one target's model or predictions for another target column.
"""

SKRUB_SUBMISSION_CONSTRAINT = """
# Skrub DataOps submission constraint

- Reuse the existing `learner = <last_node>.skb.make_learner()` from the provided solution.
- Do NOT call `model.fit(X, y)` or `model.predict(X)` on raw arrays.
- Do NOT create a new `skrub.var` for test data or re-run preprocessing manually.

- The value behind every env-dict key (`"data"`, `"file_path"`, `_skrub_X`,
  `_skrub_y`) is ALWAYS a concrete object loaded right there with
  `pd.read_csv(...)` (optionally with a plain `.drop(columns=[...])`), or a path
  string. It is NEVER a `skrub.var`, NEVER a DataOp/`.skb.*` node, and NEVER the
  output of `.skb.eval()`. The learner already holds the full graph; you only
  feed it raw data.

- If you choose to fit the learner on the full training dataset, do NOT pass symbolic Skrub DataOp objects (like `X`, `y`, `X_features`, etc.) in the dictionary. You MUST pass actual, concrete pandas DataFrame/Series:
    `learner.fit({{"data": pd.read_csv("./input/train.csv")}})` if `"data"` var was used;
    `learner.fit({{"file_path": "./input/train.csv"}})` if a `"file_path"` var was used.
    Alternatively, pass concrete features and targets using: `learner.fit({{"_skrub_X": train_X_df, "_skrub_y": train_y_series}})` where both are concrete pandas objects.
- Predict by passing an env dict that matches how the solution loads data:
    `learner.predict({{"file_path": "./input/test.csv"}})` if a path var was used;
    `learner.predict({{"_skrub_X": test_df}})` if `.skb.mark_as_X()` was used;
    `learner.predict({{"data": test_df}})` if `skrub.var("data", ...)` was used without `mark_as_X`.
  Ensure that you NEVER pass symbolic DataOps in these dictionaries.

- For classification metrics that need probabilities (AUC, logloss), use
  `learner.predict_proba({{...}})[:, 1]` (binary) with the SAME env dict shape as
  `predict` above — e.g. `learner.predict_proba({{"_skrub_X": test_df}})[:, 1]`.

- Wrong vs right at inference (the value is a raw DataFrame, nothing else):
    # WRONG — never wrap test data in a new var or call .skb.eval() to feed the env dict:
    #   test_X = skrub.var("test_data", test_df).skb.mark_as_X()
    #   preds = learner.predict_proba({{"_skrub_X": test_X.skb.eval()}})
    # RIGHT — the value is the raw DataFrame straight from read_csv:
    #   test_df = pd.read_csv("./input/test.csv")
    #   preds = learner.predict_proba({{"_skrub_X": test_df}})[:, 1]

- Banned at inference time (treat as failures): creating any `skrub.var(...)` for
  test/eval data; calling `.skb.mark_as_X()` / `.skb.mark_as_y()` on test data;
  calling `.skb.eval()` to produce a value for a `learner.fit/predict/score` env
  dict.

- MULTI-TARGET tasks (more than one target column): each target column in
  `submission.csv` MUST come from a model trained on THAT target. If the solution
  marked all targets as `y` and trained one multi-output learner, split the
  prediction array by column (`preds[:, 0]`, `preds[:, 1]`). If it trained one
  learner per target (a dict keyed by target name), predict each column with its
  matching learner. NEVER assign the same prediction array to two target columns,
  and NEVER rely on a single leftover `learner` variable from a training loop — it
  holds only the LAST target's model.
- Save predictions to `./final/submission.csv` (create the dir first, e.g.
  `os.makedirs("./final", exist_ok=True)`). Do not drop any test rows.
"""

def strip_xai_mentions(text: str) -> str:
    """Remove XAI-related guidelines when XAI is not used in the run."""
    if CONFIG.use_xai_correction or CONFIG.use_xai_refinement:
        return text
    xai_block = (
        "- the XAI instrumentation block (the `import xai_probes`,\n"
        "  `xai_probes.run_leakage_suite(...)` call, and any `.skb.set_name(\"xai_features\")`\n"
        "  / `.skb.set_name(\"xai_model\")` tags on the graph). Never delete these to clear\n"
        "  an error; fix the failing line in place. `xai_probes` is a local module staged\n"
        "  next to the script — do NOT add a `pip install` guard for it.\n"
    )
    return text.replace(xai_block, "")


SKRUB_RAG_HINT = _RAG_HINT_LINE

def is_gpt_model() -> bool:
    """Check if the configured agent model is in the GPT-3/4/5 family."""
    model = CONFIG.agent_model.lower()
    return any(x in model for x in ("gpt-3", "gpt-4", "gpt-5"))

def strip_gpt_mentions(text: str) -> str:
    """Remove GPT-specific guidelines when not using a GPT model."""
    if is_gpt_model():
        return text
    
    gpt_bullets = [
        '''- Passing `stratify` keyword argument (like `stratify=y` or `stratify=y.skb.apply_func(...)`) to `.skb.train_test_split()`. The `stratify` keyword argument is NOT supported on lazy DataOp nodes and will crash with an error containing `na_value`. Simply omit `stratify` or use a standard non-stratified split.\n''',
        '''- Eagerly iterating over or checking elements in `df.columns` or `X.columns` (e.g. `[c for c in X.columns]` or `if col in X.columns`), because columns on a `DataOp` are lazy nodes and not concrete lists. Use `.drop(columns=[...])` or `.skb.select()` to modify columns.\n''',
        '''- Calling `.to_pandas()` or pandas DataFrame methods on `data["train"]` or `data["test"]` (they are environment dicts, not DataFrames).\n''',
        '''- AttributeError: 'numpy.dtypes.Int64DType' object has no attribute 'na_value' or Evaluation of '.na_value' failed: this is caused by passing `stratify=y` (or another `DataOp` node) to `.skb.train_test_split(...)`. Skrub's `train_test_split` does not support `stratify` on lazy `DataOp` nodes. To fix it, remove the `stratify` keyword argument from `.skb.train_test_split(...)`.\n''',
        '''- TypeError: This object is a DataOp ... it is not possible to eagerly iterate over it now or eagerly use its Boolean value now: this is caused by trying to iterate over `df.columns` / `X.columns` or checking column membership (e.g. `c in X.columns`) in a list comprehension. Columns of a `DataOp` are lazy and cannot be iterated eagerly. To fix it, use `.drop(columns=[...])` or `.skb.select()` instead of iterating `df.columns`.\n''',
        '''- AttributeError: 'dict' object has no attribute 'to_pandas' when trying to use `X_train.to_pandas()`: `X_train` or `data["train"]` are environment dictionaries, not DataFrames. Do not call `.to_pandas()` on them.\n'''
    ]
    for bullet in gpt_bullets:
        text = text.replace(bullet, "")
    return text


SKRUB_DATAOPS_GUIDELINE = strip_gpt_mentions(strip_rag_tool_mentions(SKRUB_DATAOPS_GUIDELINE))
SKRUB_DATAOPS_DEBUG_GUIDELINE = strip_gpt_mentions(
    strip_xai_mentions(
        strip_rag_tool_mentions(SKRUB_DATAOPS_DEBUG_GUIDELINE)
    )
)
SKRUB_RAG_HINT = strip_rag_tool_mentions(SKRUB_RAG_HINT)
