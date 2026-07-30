from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from pipeline import FakeReviewDetector, compute_scores, generate_report, scrape_and_preprocess


st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️", layout="wide")


APP_CSS = """
<style>
  .block-container { padding-top: 2rem; padding-bottom: 2.5rem; }
  [data-testid="stMetricValue"] { font-size: 1.6rem; }
  .small-note { font-size: 0.9rem; opacity: 0.75; }
  .card {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 14px 16px;
    background: rgba(255,255,255,0.03);
  }
</style>
"""


def _rule_only_decision(fake_score: float) -> str:
    # Rough mapping to keep UX consistent when ML artifacts are missing.
    # Scores are typically in a small range (sum of weighted heuristics).
    if fake_score >= 3.5:
        return "REMOVE"
    if fake_score >= 2.5:
        return "REVIEW"
    return "CLEAN"


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _ensure_session_state():
    st.session_state.setdefault("df_raw", None)
    st.session_state.setdefault("df_scored", None)
    st.session_state.setdefault("df_out", None)
    st.session_state.setdefault("selected_ids", None)


def _get_id_column(df: pd.DataFrame) -> str:
    for col in ["review_id", "reviewId", "id"]:
        if col in df.columns:
            return col
    return df.columns[0]


def _safe_datetime_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _charts_block(df_out: pd.DataFrame):
    st.subheader("Charts")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Decisions (count)**")
        decision_counts = df_out["decision"].value_counts().reindex(["CLEAN", "REVIEW", "REMOVE"]).fillna(0).astype(int)
        st.bar_chart(decision_counts)

    with c2:
        st.markdown("**Ratings (Original vs Clean)**")
        if "rating" in df_out.columns:
            all_r = df_out["rating"].value_counts().sort_index()
            clean_r = df_out[df_out["decision"] == "CLEAN"]["rating"].value_counts().sort_index()
            ratings_df = pd.DataFrame({"All": all_r, "Clean": clean_r}).fillna(0).astype(int)
            st.bar_chart(ratings_df)
        else:
            st.info("No `rating` column found for rating charts.")

    c3, c4 = st.columns([1, 1])
    with c3:
        st.markdown("**Ensemble score distribution**")
        if "ensemble_score" in df_out.columns:
            st.bar_chart(df_out["ensemble_score"].fillna(0).clip(lower=0))
        else:
            st.info("No `ensemble_score` available.")

    with c4:
        st.markdown("**Reviews over time**")
        if "timestamp" in df_out.columns:
            ts = _safe_datetime_series(df_out["timestamp"])
            by_day = (
                pd.DataFrame({"date": ts.dt.date, "count": 1})
                .dropna()
                .groupby("date")["count"]
                .sum()
                .sort_index()
            )
            if len(by_day) > 0:
                st.line_chart(by_day)
            else:
                st.info("Timestamps could not be parsed into dates.")
        else:
            st.info("No `timestamp` column found for timeline chart.")


def main():
    st.markdown(APP_CSS, unsafe_allow_html=True)
    _ensure_session_state()

    st.title("Fake Review Detection (Google Play)")
    st.caption("Step 1: Fetch reviews → Step 2: Select reviews → Step 3: Apply detection model + charts")

    with st.sidebar:
        st.header("Input")
        app_name_or_id = st.text_input(
            "App name or App ID",
            placeholder="e.g. whatsapp or com.whatsapp",
        )

        st.header("Scraping filters")
        max_reviews = st.number_input("Max reviews (0 = no limit)", min_value=0, value=500, step=100)
        days_back = st.number_input("Days back (0 = all time)", min_value=0, value=30, step=7)
        rating_filter = st.selectbox("Rating filter", options=["All", "1", "2", "3", "4", "5"], index=0)

        st.header("Model")
        models_dir = st.text_input("Models directory", value="models")
        use_ml = st.toggle("Apply ML ensemble (LightGBM + TF-IDF)", value=True)

        st.divider()
        fetch = st.button("Fetch reviews", type="primary", use_container_width=True)
        clear = st.button("Clear results", use_container_width=True)

    if clear:
        st.session_state["df_raw"] = None
        st.session_state["df_scored"] = None
        st.session_state["df_out"] = None
        st.session_state["selected_ids"] = None
        st.rerun()

    if fetch:
        if not app_name_or_id or not app_name_or_id.strip():
            st.error("Please provide an app name or app ID.")
            return

        max_reviews_arg = None if int(max_reviews) == 0 else int(max_reviews)
        days_back_arg = None if int(days_back) == 0 else int(days_back)
        rating_filter_arg = None if rating_filter == "All" else int(rating_filter)

        status_box = st.empty()
        progress_bar = st.progress(0)
        sample_box = st.empty()

        def status_callback(msg: str):
            status_box.info(msg)

        def progress_callback(collected, target, iteration, added, final=False):
            if target:
                pct = min(int(collected / max(target, 1) * 100), 100)
                progress_bar.progress(pct)
            else:
                progress_bar.progress(min((iteration * 7) % 100, 95))

        def review_callback(review, count):
            text = review.get("content", "")
            user = review.get("userName", "Anonymous")
            stars = review.get("score", "")
            if text:
                sample_box.caption(f"Latest: {user} ({stars}★) — {text[:220]}{'…' if len(text) > 220 else ''}")

        try:
            with st.spinner("Scraping and preprocessing reviews..."):
                df = scrape_and_preprocess(
                    app_name_or_id=app_name_or_id.strip(),
                    max_reviews=max_reviews_arg,
                    days_back=days_back_arg,
                    rating_filter=rating_filter_arg,
                    progress_callback=progress_callback,
                    review_callback=review_callback,
                    status_callback=status_callback,
                )
        except Exception as e:
            st.exception(e)
            return

        st.session_state["df_raw"] = df
        st.session_state["df_scored"] = None
        st.session_state["df_out"] = None
        st.session_state["selected_ids"] = None
        st.rerun()

    if st.session_state["df_raw"] is None:
        st.info("Use the sidebar to **Fetch reviews** from Google Play.")
        return

    df = st.session_state["df_raw"]
    id_col = _get_id_column(df)

    st.markdown(
        f"""
<div class="card">
  <div><b>Fetched reviews:</b> {len(df):,}</div>
  <div class="small-note">Next: choose <b>Check all reviews</b> or select a subset, then apply detection.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.subheader("Step 2 — Select reviews")
    left, right = st.columns([1, 2])
    with left:
        check_all = st.checkbox("Check all reviews", value=True)
        max_preview = st.number_input("Preview rows", min_value=50, max_value=5000, value=500, step=50)
        show_cols = st.multiselect(
            "Columns to show",
            options=list(df.columns),
            default=[c for c in ["timestamp", "username", "rating", "text_original", "text_cleaned"] if c in df.columns][:5]
            or list(df.columns)[:6],
        )

    if check_all:
        selected_ids = df[id_col].astype(str).tolist()
    else:
        candidates = df.copy()
        candidates[id_col] = candidates[id_col].astype(str)
        options = candidates[id_col].head(2000).tolist()
        selected_ids = st.multiselect(
            "Select review IDs (showing first 2000 IDs)",
            options=options,
            default=options[: min(50, len(options))],
        )

    st.session_state["selected_ids"] = selected_ids

    preview_df = df.copy()
    preview_df[id_col] = preview_df[id_col].astype(str)
    preview_df = preview_df[preview_df[id_col].isin(selected_ids)]

    with right:
        st.markdown("**Selected reviews preview**")
        st.dataframe(preview_df[show_cols].head(int(max_preview)), use_container_width=True, height=420)
        st.caption(f"Selected: {len(preview_df):,} / {len(df):,}")

    st.subheader("Step 3 — Apply fake review detection model")
    apply_model = st.button("Apply detection", type="primary")

    if apply_model or st.session_state["df_out"] is not None:
        if st.session_state["df_out"] is None:
            if len(preview_df) == 0:
                st.error("No reviews selected.")
                return

            with st.spinner("Computing rule-based scores..."):
                df_scored = compute_scores(preview_df.copy())
            st.session_state["df_scored"] = df_scored

            if not use_ml:
                df_out = df_scored.copy()
                df_out["rule_score"] = df_out["fake_score"]
                df_out["lgbm_proba"] = None
                df_out["ensemble_score"] = df_out["fake_score"]
                df_out["decision"] = df_out["fake_score"].apply(_rule_only_decision)
                df_out["confidence"] = df_out["decision"].map({"REMOVE": "high", "REVIEW": "medium", "CLEAN": "high"})
                df_out["processed_at"] = datetime.now()
            else:
                ml_available = True
                detector_error = None
                try:
                    detector = FakeReviewDetector(models_dir=models_dir.strip() or "models")
                except Exception as e:
                    ml_available = False
                    detector_error = str(e)
                    detector = None

                if not ml_available:
                    st.warning(
                        "ML ensemble is unavailable (missing model artifacts). Falling back to **rule-only** decisions.\n\n"
                        f"Details: {detector_error}"
                    )
                    df_out = df_scored.copy()
                    df_out["rule_score"] = df_out["fake_score"]
                    df_out["lgbm_proba"] = None
                    df_out["ensemble_score"] = df_out["fake_score"]
                    df_out["decision"] = df_out["fake_score"].apply(_rule_only_decision)
                    df_out["confidence"] = df_out["decision"].map({"REMOVE": "high", "REVIEW": "medium", "CLEAN": "high"})
                    df_out["processed_at"] = datetime.now()
                else:
                    with st.spinner("Running ML ensemble detection..."):
                        results = detector.process_reviews(df_scored)
                        df_out = results["all"]

            st.session_state["df_out"] = df_out

        df_out = st.session_state["df_out"]

        st.subheader("Overview")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Selected", f"{len(df_out):,}")
        c2.metric("Clean", f"{(df_out['decision'] == 'CLEAN').sum():,}")
        c3.metric("Review", f"{(df_out['decision'] == 'REVIEW').sum():,}")
        c4.metric("Removed", f"{(df_out['decision'] == 'REMOVE').sum():,}")
        c5.metric("Avg rule score", f"{df_out['fake_score'].mean():.2f}")

        df_clean = df_out[df_out["decision"] == "CLEAN"].copy()
        df_review = df_out[df_out["decision"] == "REVIEW"].copy()
        df_removed = df_out[df_out["decision"] == "REMOVE"].copy()

        st.subheader("Results tables")
        tabs = st.tabs(["Clean", "Needs review", "Removed", "All (selected)"])
        for tab, dfx in zip(tabs, [df_clean, df_review, df_removed, df_out]):
            with tab:
                st.dataframe(
                    dfx.sort_values(["ensemble_score", "fake_score"], ascending=False, na_position="last"),
                    use_container_width=True,
                    height=520,
                )

        _charts_block(df_out)

        st.subheader("Downloads")
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("Clean CSV", data=_to_csv_bytes(df_clean), file_name="clean_reviews.csv", mime="text/csv")
        d2.download_button("Review CSV", data=_to_csv_bytes(df_review), file_name="needs_review.csv", mime="text/csv")
        d3.download_button("Removed CSV", data=_to_csv_bytes(df_removed), file_name="removed_reviews.csv", mime="text/csv")
        d4.download_button("All CSV", data=_to_csv_bytes(df_out), file_name="all_scored.csv", mime="text/csv")

        st.divider()
        st.subheader("Report")

        try:
            with st.spinner("Generating report..."):
                report = generate_report(df_clean, df_removed, df_out, output_dir="outputs")
            st.write(report["report_text"])
            st.caption(f"Saved report file: `{report['report_file']}`")

            if report.get("summary_plot") is not None:
                st.pyplot(report["summary_plot"])
            if report.get("score_plot") is not None:
                st.pyplot(report["score_plot"])
        except Exception as e:
            st.warning("Report generation failed (this does not affect detection).")
            st.exception(e)
    else:
        st.info("Click **Apply detection** in Step 3 to run scoring and see charts, tables, and downloads.")


if __name__ == "__main__":
    # Streamlit runs this as a script; keep main() import-safe.
    main()

