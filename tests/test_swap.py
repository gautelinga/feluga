"""`femtools.swap`: a folder swapped in whole, and put back after a kill at any point
between the renames.

    python3 -m pytest tests/test_swap.py
"""
import os

from femtools.swap import new_folder, recover, swap_in


def folder(path, text):
    os.makedirs(path)
    with open(os.path.join(path, "f"), "w") as f:
        f.write(text)


def content(path):
    with open(os.path.join(path, "f")) as f:
        return f.read()


def test_swap_in_replaces_the_folder(tmp_path):
    target, new = str(tmp_path / "out"), str(tmp_path / "out.new")
    folder(new, "1")
    swap_in(new, target)
    folder(new, "2")
    swap_in(new + "/", target + "/")
    assert content(target) == "2"
    assert sorted(os.listdir(tmp_path)) == ["out"]


def test_recover_after_a_kill_between_the_renames(tmp_path):
    """The old folder renamed aside, the new one not yet in: the new one goes in."""
    target, new = str(tmp_path / "out"), str(tmp_path / "out.new")
    folder(target + ".old", "old")
    folder(new, "new")
    messages = []
    assert recover(target, new, messages.append)
    assert content(target) == "new" and sorted(os.listdir(tmp_path)) == ["out"]
    assert "new folder" in messages[0]


def test_recover_falls_back_to_the_old_folder(tmp_path):
    target, new = str(tmp_path / "out"), str(tmp_path / "out.new")
    folder(target + ".old", "old")
    assert recover(target, new)
    assert content(target) == "old" and sorted(os.listdir(tmp_path)) == ["out"]


def test_recover_leaves_a_whole_target_alone(tmp_path):
    """A kill while writing the new folder, or after the swap: nothing to do."""
    target, new = str(tmp_path / "out"), str(tmp_path / "out.new")
    folder(target, "current")
    folder(new, "partial")
    folder(target + ".old", "old")
    assert not recover(target, new)
    assert content(target) == "current" and not recover(str(tmp_path / "none"), new)


def test_recover_with_a_trailing_slash(tmp_path):
    """`<target>/` names the same folders as `<target>`: the new one goes in."""
    target = str(tmp_path / "out") + "/"
    folder(str(tmp_path / "out.old"), "old")
    folder(new_folder(target), "new")
    assert new_folder(target) == str(tmp_path / "out.new")
    assert recover(target) and content(target) == "new"
    assert sorted(os.listdir(tmp_path)) == ["out"]
