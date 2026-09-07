"""Eligible-first selection, persisted inside the atomic epoch checkpoint."""

import math


def update_selection(selection, snapshot, gated):
    selection = dict(selection)
    score = snapshot["selection_score"]
    if not math.isfinite(score):
        raise ValueError("Checkpoint selection score must be finite")
    best_score = selection.get("best_score")
    best_eligible = selection.get("best_eligible")
    score_improved = best_score is None or score < best_score["selection_score"]
    eligible_improved = (
        gated
        and snapshot["validation_constraints_pass"]
        and (best_eligible is None or score < best_eligible["selection_score"])
    )
    selected_improved = eligible_improved or (best_eligible is None and score_improved)
    if score_improved or eligible_improved:
        # state_dict values otherwise alias parameters changed by the next optimizer step.
        saved = {
            **snapshot,
            "model": {
                key: value.detach().cpu().clone() for key, value in snapshot["model"].items()
            },
        }
        if score_improved:
            selection["best_score"] = saved
        if eligible_improved:
            selection["best_eligible"] = saved
    return selection, bool(selected_improved)


def selected_checkpoint(selection):
    return selection.get("best_eligible") or selection.get("best_score")


def publish_selection(selection, output, save, updated_epoch=None):
    """Publish aliases only after latest.pt commits selection and its model snapshots.

    Resume calls this again, repairing missing/stale aliases without trusting them as
    the authoritative selection state. Only latest.pt is used for training resume.
    """

    def publish(name, snapshot):
        if snapshot is None:
            # latest.pt is authoritative, including the absence of an eligible model.
            (output / name).unlink(missing_ok=True)
            return
        if snapshot is not None and (updated_epoch is None or snapshot["epoch"] == updated_epoch):
            save(output / name, snapshot)

    publish("best_score.pt", selection.get("best_score"))
    publish("best_eligible.pt", selection.get("best_eligible"))
    selected = selected_checkpoint(selection)
    publish("best.pt", selected)


def restore_selection(state, output, load, gated):
    if "selection_state" in state:
        return state["selection_state"]
    # A legacy run can recover its retained best only; do not invent older lost models.
    best = load(output / "best.pt")
    if best.get("contract") != state["contract"] or best["epoch"] > state["epoch"]:
        raise ValueError("Legacy best checkpoint disagrees with the committed resume epoch")
    snapshot = {
        key: best[key]
        for key in (
            "config",
            "contract",
            "model",
            "epoch",
            "step",
            "manifest_sha256",
            "training_scene_hashes",
            "validation_structure",
            "baseline_structure",
            "validation_constraints_pass",
        )
        if key in best
    }
    snapshot.update(
        selection_score=best["best"],
        validation_loss=best.get("best_loss", best["best"]),
        validation_constraints_pass=bool(best.get("validation_constraints_pass", False)),
    )
    return update_selection({}, snapshot, gated)[0]
