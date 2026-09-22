"""Tests for reading a comparison on a slice of the validation set.

A family of transformations can be adopted for what it does to part of
the data. Measuring that requires reading a breakdown of the scoring
artefact rather than its pooled figures, and two things can go wrong
quietly. The reported number can come from the wrong snapshot, because
a run does not peak on a subset where it peaks overall; and the noise
floor a difference is judged against can be computed over a different
set of snapshots than the difference itself. Either produces a result
that looks ordinary and means nothing, so the tests below pin which
snapshot each reading comes from rather than only that a number
appears.

The fixture is built so that the two readings must disagree: the slice
declines as training goes on while the pooled metric improves, and the
three baseline seeds peak at three different epochs.
"""
import json
import sys

import pytest

import scripts.compare_to_baseline as cli

# Where each run peaks on the pooled metric. Deliberately three
# different epochs: it is the spread of these positions that makes a
# slice read at the pooled peak so much noisier than one read at its
# own.
OVERALL_CURVES = {
    "b0_a": {1: 0.800, 2: 0.820, 3: 0.850, 4: 0.840},
    "b0_b": {1: 0.800, 2: 0.810, 3: 0.830, 4: 0.851},
    "b0_c": {1: 0.810, 2: 0.853, 3: 0.840, 4: 0.830},
    "cand": {1: 0.810, 2: 0.830, 3: 0.860, 4: 0.855},
}

# The same snapshots scored on the slice, which gets worse with
# training. Every run peaks here in the first epoch.
FINE_CURVES = {
    "b0_a": {1: 0.900, 2: 0.880, 3: 0.860, 4: 0.840},
    "b0_b": {1: 0.901, 2: 0.880, 3: 0.860, 4: 0.840},
    "b0_c": {1: 0.899, 2: 0.880, 3: 0.860, 4: 0.840},
    "cand": {1: 0.910, 2: 0.890, 3: 0.870, 4: 0.850},
}


def _figures(f1):
    """One snapshot's figures, with the fields a reading needs."""
    return {
        "f1": f1,
        "boundary_f1": {"0.1": 0.80},
        "pore_count_error": -0.03,
        "merges_per_100_gt": 2.5,
        "splits_per_100_gt": 2.0,
    }


def _artefact(path, runs):
    """Write a scoring artefact holding the given runs."""
    payload = {}
    for run in runs:
        for epoch, f1 in OVERALL_CURVES[run].items():
            payload[f"{run}/epoch-{epoch}"] = {
                "overall": _figures(f1),
                "per_scale_bin": [
                    {"label": "scale_bin=fine",
                     **_figures(FINE_CURVES[run][epoch])},
                    {"label": "scale_bin=coarse", **_figures(f1)},
                ],
            }
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def artefacts(tmp_path):
    """A candidate artefact and a baseline artefact of three seeds."""
    return (
        _artefact(tmp_path / "candidate.json", ["cand"]),
        _artefact(tmp_path / "baseline.json", ["b0_a", "b0_b", "b0_c"]),
    )


def _run(monkeypatch, artefacts, out, extra=()):
    """Run the comparison and return its summary entries."""
    candidate, baseline = artefacts
    monkeypatch.setattr(sys, "argv", [
        "compare_to_baseline.py",
        "--candidate", str(candidate),
        "--baseline", str(baseline),
        "--out", str(out),
        *extra,
    ])
    assert cli.main() == cli.EXIT_OK
    return json.loads(out.read_text())


def _entry(summary, view, reading="Full-budget reading"):
    """The one entry of a summary matching a view and a reading."""
    matching = [
        item for item in summary
        if item["view"] == view and item["reading"] == reading
    ]
    assert len(matching) == 1, f"{view}/{reading}: {len(matching)} entries"
    return matching[0]


def test_a_slice_is_read_at_its_own_peak(monkeypatch, artefacts, tmp_path):
    """The slice picks the snapshot that is best on the slice.

    The candidate peaks in the third epoch on the pooled metric and in
    the first on this slice. Reporting the pooled peak here would
    describe a model that is not the best this policy produced on the
    images the family exists for.
    """
    summary = _run(
        monkeypatch, artefacts, tmp_path / "out.json",
        ["--slice", "scale_bin=fine"],
    )

    entry = _entry(summary, "scale_bin=fine")
    assert entry["candidate"]["epoch"] == 1
    assert entry["candidate"]["f1"] == pytest.approx(0.910)


def test_a_slice_read_at_the_pooled_peak_reports_that_snapshot(
    monkeypatch, artefacts, tmp_path
):
    """The stricter reading takes the snapshot chosen on everything.

    The figures must still be the slice's own. Reporting the pooled
    number under a slice's name is the failure this pins: both are
    plausible values on the same scale, so nothing downstream would
    reveal the substitution.
    """
    summary = _run(
        monkeypatch, artefacts, tmp_path / "out.json",
        ["--slice", "scale_bin=fine"],
    )

    entry = _entry(summary, "scale_bin=fine at overall peak")
    assert entry["candidate"]["epoch"] == 3
    assert entry["candidate"]["f1"] == pytest.approx(0.870)


def test_reading_a_slice_at_the_pooled_peak_widens_the_floor(
    monkeypatch, artefacts, tmp_path
):
    """The two readings carry different noise floors, and must.

    The baseline seeds peak at three different epochs on the pooled
    metric, so reading the slice there reads it at three different
    points of its own declining curve. The spread that results is a
    property of where the pooled peaks happened to fall, not of the
    seeds, and it is several times the spread the slice shows when each
    seed is allowed its own peak. Both numbers are reported precisely
    because the gap between them is this large.
    """
    summary = _run(
        monkeypatch, artefacts, tmp_path / "out.json",
        ["--slice", "scale_bin=fine"],
    )

    own = _entry(summary, "scale_bin=fine")["significance_floor"]
    pooled = _entry(
        summary, "scale_bin=fine at overall peak"
    )["significance_floor"]
    assert own == pytest.approx(0.002)
    assert pooled == pytest.approx(0.040)


def test_asking_for_a_slice_does_not_move_the_pooled_reading(
    monkeypatch, artefacts, tmp_path
):
    """The comparison every earlier experiment was decided on is fixed.

    Slices are an addition to a tool whose results are already
    published inside the study. A change here that shifted the pooled
    numbers would silently invalidate those, so the pooled entries are
    required to be identical whether or not a slice is also requested.
    """
    without = _run(monkeypatch, artefacts, tmp_path / "a.json")
    with_slice = _run(
        monkeypatch, artefacts, tmp_path / "b.json",
        ["--slice", "scale_bin=fine"],
    )

    pooled = [item for item in with_slice if item["view"] == "overall"]
    assert pooled == without


def test_a_slice_the_artefact_does_not_hold_names_what_it_does(
    monkeypatch, artefacts, tmp_path
):
    """A mistyped slice fails loudly and says what was available.

    The alternative - skipping what cannot be found - would report a
    comparison on fewer views than were asked for, and the missing one
    is easy to overlook in a log that otherwise looks healthy.
    """
    summary_path = tmp_path / "out.json"
    candidate, baseline = artefacts
    monkeypatch.setattr(sys, "argv", [
        "compare_to_baseline.py",
        "--candidate", str(candidate),
        "--baseline", str(baseline),
        "--out", str(summary_path),
        "--slice", "scale_bin=medium",
    ])

    assert cli.main() == cli.EXIT_FAILED
    assert not summary_path.exists()


def test_a_missing_section_is_distinguished_from_a_missing_row(tmp_path):
    """An artefact scored before a breakdown existed says so.

    Both cases are a slice that cannot be read, but only one is a typo,
    and the fix differs: re-score the checkpoints, or correct the
    label.
    """
    entry = {"overall": _figures(0.85)}

    with pytest.raises(KeyError, match="per_scale_bin"):
        cli.figures_for(entry, "scale_bin=fine")


def test_every_slice_is_reported_from_both_sides():
    """Each slice yields both readings, and the pooled view comes
    first, so that a summary file is read in the order the argument
    list asked for."""
    views = cli.views_to_compare(["scale_bin=fine", "material=AS"])

    assert views == [
        (cli.OVERALL, None),
        ("scale_bin=fine", None),
        ("scale_bin=fine", cli.OVERALL),
        ("material=AS", None),
        ("material=AS", cli.OVERALL),
    ]


# A subset a family is adopted for need not be a row the scoring layer
# writes out. The second microscope is two materials, and read apart
# they are six and twelve images - each too small to separate anything
# from seed noise. Their union is what the question was about.
#
# The counts below are chosen so that the three ways of combining two
# rows give three different answers: pooling the counts gives 0.7234,
# averaging the rows' F1 gives 0.6464, and weighting that average by
# the number of annotated pores gives 0.7143. Only the first is the F1
# of the union, and a fixture where they coincided would let two of the
# three defects through.
BIG_ROW = {"n_true_positives": 300, "n_false_positives": 90,
           "n_false_negatives": 100, "n_gt": 400, "n_pred": 390}
SMALL_ROW = {"n_true_positives": 40, "n_false_positives": 10,
             "n_false_negatives": 60, "n_gt": 100, "n_pred": 50}
POOLED_F1 = 0.723404
ROW_MEAN_F1 = 0.646414
GT_WEIGHTED_F1 = 0.714262


def _counted(counts, n_images, merges, splits):
    """One breakdown row, carrying the counts a pooled view adds up."""
    n_gt = counts["n_gt"]
    return {
        **counts,
        "n_images": n_images,
        "n_scale_outliers_excluded": 0,
        "n_merges": merges,
        "n_splits": splits,
        "f1": 2 * counts["n_true_positives"] / (
            2 * counts["n_true_positives"]
            + counts["n_false_positives"]
            + counts["n_false_negatives"]
        ),
        "boundary_f1": {"0.1": 0.80},
        "pore_count_error": (counts["n_pred"] - n_gt) / n_gt,
        "merges_per_100_gt": 100 * merges / n_gt,
        "splits_per_100_gt": 100 * splits / n_gt,
    }


def _material_artefact(path, runs, tie=False):
    """An artefact whose material rows carry counts.

    Every run is given the same rows in both epochs except that the
    second is made slightly worse, which pins the pooled peak to the
    first and therefore to the counts named above, so that the figure a
    pooled reading reports can be checked against a number computed by
    hand. Asking for a tie removes that difference, leaving two epochs
    the rule cannot separate.
    """
    payload = {}
    for run in runs:
        for epoch in (1, 2):
            big = dict(BIG_ROW)
            if epoch == 2 and not tie:
                big["n_true_positives"] -= 1
                big["n_false_negatives"] += 1
            payload[f"{run}/epoch-{epoch}"] = {
                "overall": _figures(OVERALL_CURVES[run][epoch]),
                "per_material": [
                    {"label": "material=K", **_counted(big, 12, 30, 20)},
                    {"label": "material=VAB",
                     **_counted(dict(SMALL_ROW), 6, 5, 9)},
                ],
            }
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def material_artefacts(tmp_path):
    """A candidate and three baseline seeds, with material counts."""
    return (
        _material_artefact(tmp_path / "candidate.json", ["cand"]),
        _material_artefact(
            tmp_path / "baseline.json", ["b0_a", "b0_b", "b0_c"]
        ),
    )


def test_a_pooled_view_adds_the_counts_rather_than_the_figures(
    monkeypatch, material_artefacts, tmp_path
):
    """The union's F1 is computed from its totals, not from its rows.

    Averaging the rows is the tempting shortcut and it is wrong in the
    only case that matters: rows holding different numbers of pores.
    Here the rows are four hundred pores and one hundred, so the
    average and the true figure differ in the second decimal - far
    more than the noise floor a decision is read against, and in a
    direction nothing downstream would reveal.
    """
    summary = _run(
        monkeypatch, material_artefacts, tmp_path / "out.json",
        ["--slice", "material=K+VAB"],
    )

    reported = _entry(summary, "material=K+VAB")["candidate"]["f1"]
    assert reported == pytest.approx(POOLED_F1, abs=5e-5)
    assert reported != pytest.approx(ROW_MEAN_F1, abs=1e-3)
    assert reported != pytest.approx(GT_WEIGHTED_F1, abs=1e-3)


def test_a_pooled_view_pools_every_figure_it_reports(
    monkeypatch, material_artefacts, tmp_path
):
    """The counted fields follow the same rule as F1.

    The error in the number of pores and the merge and split rates are
    all ratios to the annotated count, so each one taken from the
    larger row alone - or averaged across rows - would be a different
    number wearing the right name.
    """
    summary = _run(
        monkeypatch, material_artefacts, tmp_path / "out.json",
        ["--slice", "material=K+VAB"],
    )

    candidate = _entry(summary, "material=K+VAB")["candidate"]
    assert candidate["pore_count_error"] == pytest.approx(-0.12, abs=5e-4)
    assert candidate["merges_per_100_gt"] == pytest.approx(7.0, abs=5e-3)
    assert candidate["splits_per_100_gt"] == pytest.approx(5.8, abs=5e-3)


def test_a_pooled_view_omits_boundary_agreement_rather_than_faking_it(
    monkeypatch, material_artefacts, tmp_path
):
    """Boundary agreement is absent, and the reading still completes.

    It is pooled over outline pixels, which the artefact does not keep,
    so there is nothing here to add up. Carrying a weighted average of
    the rows instead would put a number on the same scale as the real
    one under the same name, and a reader comparing it with a pooled
    view's boundary figure would be comparing two different
    measurements.
    """
    summary = _run(
        monkeypatch, material_artefacts, tmp_path / "out.json",
        ["--slice", "material=K+VAB"],
    )

    pooled = _entry(summary, "material=K+VAB")["candidate"]
    single = _entry(summary, "overall")["candidate"]
    assert pooled["boundary_f1"] is None
    assert single["boundary_f1"] == {"0.1": 0.80}


def test_a_pooled_view_that_cannot_break_a_tie_stops_the_comparison(
    monkeypatch, tmp_path
):
    """An undecidable peak fails the run instead of being skipped.

    Without boundary agreement the ranking rule has one link fewer, so
    a tie it would otherwise have broken can survive. The tempting
    behaviour - take the first, or log a warning and carry on - would
    put an arbitrarily chosen snapshot on one side of a paired
    comparison, and the arbitrary part would then be reported as an
    effect of the policy. It has to be louder than the ordinary gap
    where a reading is merely unavailable.
    """
    summary_path = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", [
        "compare_to_baseline.py",
        "--candidate",
        str(_material_artefact(tmp_path / "c.json", ["cand"], tie=True)),
        "--baseline",
        str(_material_artefact(
            tmp_path / "b.json", ["b0_a", "b0_b", "b0_c"], tie=True
        )),
        "--out", str(summary_path),
        "--slice", "material=K+VAB",
    ])

    assert cli.main() == cli.EXIT_FAILED
    assert not summary_path.exists()


def test_one_missing_member_of_a_pooled_view_is_named(tmp_path):
    """A union is only as readable as its least available row.

    Pooling what was found and reporting it under the name of the whole
    union would answer a different question than the one asked, on
    fewer images, with no sign in the output that it had happened.
    """
    entry = {
        "overall": _figures(0.85),
        "per_material": [
            {"label": "material=K", **_counted(dict(BIG_ROW), 12, 30, 20)},
        ],
    }

    with pytest.raises(KeyError, match="material=VAB"):
        cli.figures_for(entry, "material=K+VAB")


def test_naming_one_value_still_reads_the_row_itself(tmp_path):
    """A single value is the row, not a union of one.

    The row carries fields a pooled view cannot reconstruct, and
    routing every slice through the pooling path would silently drop
    them from readings that were correct before pooling existed.
    """
    row = {"label": "material=K", **_counted(dict(BIG_ROW), 12, 30, 20)}
    entry = {"overall": _figures(0.85), "per_material": [row]}

    assert cli.figures_for(entry, "material=K") is row
