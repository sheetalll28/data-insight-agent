"""Data pipeline behaviour on messy and edge-case inputs."""
import numpy as np
import pandas as pd

from pipeline import run_cleaning_pipeline, detect_identifier_columns, read_tabular_file


def messy():
    return pd.DataFrame({
        "Patient ID": [1, 2, 3, 4, 5, 6, 7, 8],
        "Age": ["30", "40", "UNKNOWN", "35", "999", "33", "38", None],
        "Score": [1.0, 2.0, 2.5, np.nan, 3.0, 2.0, 1.5, 2.2],
        "Group": ["a", "b", None, "a", "a", "b", "a", "a"],
    })


def test_original_dataframe_is_not_modified():
    raw = messy()
    snapshot = raw.copy()
    run_cleaning_pipeline(raw)
    pd.testing.assert_frame_equal(raw, snapshot)


def test_mostly_numeric_text_is_coerced_and_everything_imputed():
    clean, rep = run_cleaning_pipeline(messy(), return_report=True)
    assert pd.api.types.is_numeric_dtype(clean["Age"])
    assert rep["coerced_to_numeric"] == {"Age": 1}
    assert clean.drop(columns=["Patient ID"]).isna().sum().sum() == 0
    assert rep["imputed"]["Group"]["method"] == "mode" and clean.loc[2, "Group"] == "a"


def test_identifier_detected_and_left_untouched():
    clean, rep = run_cleaning_pipeline(messy(), return_report=True)
    assert rep["identifier_columns"] == ["Patient ID"]
    assert clean["Patient ID"].tolist() == list(range(1, 9))


def test_outliers_clipped_or_only_flagged():
    clean, rep = run_cleaning_pipeline(messy(), return_report=True)
    assert "Age" in rep["outliers"] and clean["Age"].max() < 999
    kept, rep2 = run_cleaning_pipeline(messy(), clip_outliers=False, return_report=True)
    assert kept["Age"].max() == 999 and rep2["outliers"]["Age"]["action"] == "flagged only"


def test_only_exact_duplicates_removed():
    df = pd.DataFrame({"id": [1, 2, 3, 3], "v": [5, 5, 6, 6]})
    clean, rep = run_cleaning_pipeline(df, return_report=True)
    assert rep["exact_duplicates_removed"] == 1 and len(clean) == 3
    # rows 1 and 2 differ only by id -> counted, NOT dropped
    df2 = pd.DataFrame({"record_id": [1, 2, 3], "v": [5, 5, 6], "w": ["x", "x", "y"]})
    clean2, rep2 = run_cleaning_pipeline(df2, return_report=True)
    assert len(clean2) == 3 and rep2["near_duplicates_ignoring_ids"] == 1


def test_non_id_unique_numeric_not_flagged():
    df = pd.DataFrame({"revenue": np.random.default_rng(0).normal(size=50)})
    assert detect_identifier_columns(df) == []


def test_edge_cases_do_not_crash():
    for df in [pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [np.nan, np.nan]}),
               pd.DataFrame({"t": ["x", "y", "z"]}), pd.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]})]:
        run_cleaning_pipeline(df)


def test_bundled_messy_dataset_cleans_fully():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "messy_data.csv")
    clean, rep = run_cleaning_pipeline(pd.read_csv(path), return_report=True)
    assert clean.isna().sum().sum() == 0
    assert rep["identifier_columns"] == ["Patient ID"]


def test_identifier_found_even_when_rows_are_duplicated():
    df = pd.DataFrame({"Customer ID": [f"C{i}" for i in range(30)] + ["C0"],
                       "v": list(range(30)) + [0]})
    clean, rep = run_cleaning_pipeline(df, return_report=True)
    assert rep["exact_duplicates_removed"] == 1 and rep["identifier_columns"] == ["Customer ID"]



def test_read_tsv_and_csv_give_same_frame():
    import io
    tsv = io.StringIO("name\tcity\tscore\nSmith, John\tNew York, NY\t3.5\nLee\tParis\t4\n")
    csv = io.StringIO('name,city,score\n"Smith, John","New York, NY",3.5\nLee,Paris,4\n')
    a, b = read_tabular_file(tsv, "people.TSV"), read_tabular_file(csv, "people.csv")
    pd.testing.assert_frame_equal(a, b)
    assert a.loc[0, "name"] == "Smith, John"       # comma inside a TSV value is kept


def test_read_rejects_bad_files():
    import io
    import pytest
    with pytest.raises(ValueError, match="unsupported"):
        read_tabular_file(io.StringIO("a,b\n1,2\n"), "data.xlsx")
    with pytest.raises(ValueError, match="tab-separated"):
        read_tabular_file(io.StringIO("a\tb\n1\t2\n"), "mislabelled.csv")
    with pytest.raises(ValueError):
        read_tabular_file(io.StringIO("a,b\n"), "empty.csv")
