"""Tests for reducing judged panels to one decision per family."""
import json

from materials_vision.phase0.review import (STATUS_ACCEPTED, STATUS_PENDING,
                                            STATUS_REJECTED, STATUS_REVISE,
                                            decision_sheet, family_status,
                                            load_review, summarize)


def panel(
    family: str = "F3b_blur",
    level: str = "high",
    image_id: str = "AS1_40_1",
    kind: str = "gate",
    fingerprint: str = "abc",
    repeat: int = 0,
) -> dict:
    """One entry of a panel index."""
    return {
        "panel_id": f"{family}__{level}__{image_id}__r{repeat}",
        "family": family,
        "level": level,
        "kind": kind,
        "fingerprint": fingerprint,
        "image_id": image_id,
    }


def decision(
    panel_entry: dict, status: str = "ok", criteria=(), fingerprint=None
) -> dict:
    """One verdict on one panel."""
    return {
        "panel_id": panel_entry["panel_id"],
        "fingerprint": fingerprint or panel_entry["fingerprint"],
        "status": status,
        "criteria": list(criteria),
    }


def review(decisions=(), verdicts=()) -> dict:
    """The review file, assembled from records."""
    return {
        "decisions": {d["panel_id"]: d for d in decisions},
        "verdicts": {v["key"]: v for v in verdicts},
    }


def verdict(
    key: str = "F3b_blur__high",
    status: str = STATUS_ACCEPTED,
    fingerprint: str = "abc",
    reason: str = "",
) -> dict:
    """One verdict on one family and setting."""
    return {
        "key": key, "status": status, "fingerprint": fingerprint,
        "reason": reason,
        "family": key.split("__")[0], "level": key.split("__")[1],
    }


class TestCounting:
    """What the panels say, before anyone's verdict on the family."""

    def test_panels_are_grouped_by_family_and_level(self) -> None:
        panels = [
            panel(image_id="AS1_40_1"), panel(image_id="AS1_40_2"),
            panel(level="low", image_id="AS1_40_1"),
        ]
        summaries = summarize(panels, review())
        counts = {s.key: s.n_panels for s in summaries}
        assert counts == {"F3b_blur__high": 2, "F3b_blur__low": 1}

    def test_problems_and_their_criteria_are_tallied(self) -> None:
        first, second = panel(image_id="A"), panel(image_id="B")
        summaries = summarize([first, second], review(decisions=[
            decision(first, "problem", criteria=[4]),
            decision(second, "problem", criteria=[4, 7]),
        ]))
        summary = summaries[0]
        assert summary.n_problems == 2
        assert summary.criteria[4] == 2
        assert summary.criteria[7] == 1
        assert summary.problem_images == ["A", "B"]

    def test_a_decision_made_against_other_parameters_is_ignored(
        self,
    ) -> None:
        # It describes a picture that is no longer on screen.
        entry = panel()
        summaries = summarize([entry], review(decisions=[
            decision(entry, "ok", fingerprint="stale")
        ]))
        assert summaries[0].n_decided == 0
        assert not summaries[0].complete


class TestLevelStatus:
    """When a setting counts as decided."""

    def test_a_complete_and_accepted_level_is_accepted(self) -> None:
        entry = panel()
        summaries = summarize([entry], review(
            decisions=[decision(entry)], verdicts=[verdict()],
        ))
        assert summaries[0].status == STATUS_ACCEPTED

    def test_a_level_with_unseen_panels_stays_pending(self) -> None:
        first, second = panel(image_id="A"), panel(image_id="B")
        summaries = summarize([first, second], review(
            decisions=[decision(first)], verdicts=[verdict()],
        ))
        assert summaries[0].status == STATUS_PENDING

    def test_a_level_without_a_verdict_stays_pending(self) -> None:
        entry = panel()
        summaries = summarize(
            [entry], review(decisions=[decision(entry)])
        )
        assert summaries[0].status == STATUS_PENDING

    def test_a_verdict_from_before_a_parameter_change_is_stale(
        self,
    ) -> None:
        entry = panel()
        summaries = summarize([entry], review(
            decisions=[decision(entry)],
            verdicts=[verdict(fingerprint="older")],
        ))
        assert summaries[0].stale
        assert summaries[0].status == STATUS_PENDING


class TestFamilyStatus:
    """One family, from the settings that gate it."""

    def build(self, *statuses, kinds=None) -> list:
        """Summaries with the given verdicts, all complete."""
        kinds = kinds or ["gate"] * len(statuses)
        panels, decisions, verdicts = [], [], []
        for index, (status, kind) in enumerate(zip(statuses, kinds)):
            entry = panel(level=f"l{index}", kind=kind)
            panels.append(entry)
            decisions.append(decision(entry))
            verdicts.append(verdict(
                key=f"F3b_blur__l{index}", status=status
            ))
        return summarize(panels, review(decisions, verdicts))

    def test_every_gate_accepted_accepts_the_family(self) -> None:
        summaries = self.build(STATUS_ACCEPTED, STATUS_ACCEPTED)
        assert family_status(summaries) == STATUS_ACCEPTED

    def test_one_rejected_gate_rejects_the_family(self) -> None:
        # The acceptance criteria speak about the maximum strength, so
        # a strong setting that failed decides the family.
        summaries = self.build(STATUS_ACCEPTED, STATUS_REJECTED)
        assert family_status(summaries) == STATUS_REJECTED

    def test_revise_outranks_accepted(self) -> None:
        summaries = self.build(STATUS_ACCEPTED, STATUS_REVISE)
        assert family_status(summaries) == STATUS_REVISE

    def test_anything_unreviewed_keeps_the_family_pending(
        self,
    ) -> None:
        summaries = self.build(STATUS_ACCEPTED, STATUS_ACCEPTED)
        summaries.append(summarize([panel(level="extra")], review())[0])
        assert family_status(summaries) == STATUS_PENDING

    def test_a_diagnostic_verdict_does_not_gate(self) -> None:
        # The punishing settings lie outside the frozen ranges; a
        # problem there informs a revision, it cannot reject numbers
        # nobody proposed using.
        summaries = self.build(
            STATUS_ACCEPTED, STATUS_REJECTED,
            kinds=["gate", "diagnostic"],
        )
        assert family_status(summaries) == STATUS_ACCEPTED

    def test_a_family_with_no_gate_is_pending(self) -> None:
        summaries = self.build(
            STATUS_ACCEPTED, kinds=["diagnostic"]
        )
        assert family_status(summaries) == STATUS_PENDING


class TestDecisionSheet:
    """The record the experiment's decision table receives."""

    def accepted_blur(self) -> list:
        """A blur family accepted at its one reviewed setting."""
        entry = panel(family="F3b_blur", level="high")
        from materials_vision.phase0.levels import level_by_key
        level = level_by_key("F3b_blur__high")
        entry["fingerprint"] = level.fingerprint
        return summarize([entry], review(
            decisions=[decision(entry)],
            verdicts=[verdict(
                key="F3b_blur__high", fingerprint=level.fingerprint,
                reason="wyglada wiarygodnie",
            )],
        ))

    def test_the_status_and_its_levels_are_recorded(self) -> None:
        sheet = decision_sheet(self.accepted_blur())
        record = sheet["F3b_blur"]["faza0_wizualna"]
        assert record["status"] == STATUS_ACCEPTED
        assert record["rozpoznawalnosc_zachowana"] is True
        assert record["poziomy_sily"][0]["level"] == "high"
        assert record["poziomy_sily"][0]["reason"] == (
            "wyglada wiarygodnie"
        )

    def test_the_approved_parameters_are_the_ones_rendered(
        self,
    ) -> None:
        # Not the family's current defaults, which may have been
        # edited since the panels were judged.
        sheet = decision_sheet(self.accepted_blur())
        approved = sheet["F3b_blur"]["faza0_wizualna"][
            "approved_parameters"
        ]["high"]
        assert approved["sigma_px"] == (0.8, 0.8)

    def test_an_unaccepted_level_approves_nothing(self) -> None:
        entry = panel()
        summaries = summarize([entry], review(
            decisions=[decision(entry, "problem", criteria=[4])],
            verdicts=[verdict(status=STATUS_REVISE)],
        ))
        sheet = decision_sheet(summaries)
        assert sheet["F3b_blur"]["faza0_wizualna"][
            "approved_parameters"
        ] == {}

    def test_the_evidence_names_the_criterion_that_failed(
        self,
    ) -> None:
        entry = panel()
        summaries = summarize([entry], review(
            decisions=[decision(entry, "problem", criteria=[4])],
        ))
        evidence = decision_sheet(summaries)["F3b_blur"]["evidence"]
        assert evidence["n_problems"] == 1
        assert "znikaja cienkie sciany albo male pory" in (
            evidence["criteria"]
        )


class TestLoading:
    """Reading what the review page wrote."""

    def test_a_missing_file_reads_as_empty(self, tmp_path) -> None:
        review_file = load_review(tmp_path / "review.json")
        assert review_file == {"decisions": {}, "verdicts": {}}

    def test_a_partial_file_gets_both_sections(self, tmp_path) -> None:
        path = tmp_path / "review.json"
        path.write_text(json.dumps({"decisions": {}}), encoding="utf-8")
        assert "verdicts" in load_review(path)
