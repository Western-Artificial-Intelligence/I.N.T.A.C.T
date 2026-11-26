from circuits import IOI_CIRCUIT, KEEP_POSITION, heads_of

# Wang et al. 2022, Appendix K (Figure 20), copied independently of circuits.py so an accidental edit there fails.
PAPER = {
    "name mover": {(9, 9), (9, 6), (10, 0)},
    "backup name mover": {(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)},
    "negative name mover": {(10, 7), (11, 10)},
    "s2 inhibition": {(7, 3), (7, 9), (8, 6), (8, 10)},
    "induction": {(5, 5), (5, 8), (5, 9), (6, 9)},
    "duplicate token": {(0, 1), (0, 10), (3, 0)},
    "previous token": {(2, 2), (4, 11)},
}


def test_ioi_circuit_matches_the_paper():
    """Every class holds exactly the paper's heads, no head is in two classes, and there are 26 in total."""
    assert {cls: set(heads) for cls, heads in IOI_CIRCUIT.items()} == PAPER
    assert len(heads_of(IOI_CIRCUIT)) == sum(len(h) for h in IOI_CIRCUIT.values()) == 26


def test_keep_position_covers_every_class():
    """Each class acts at end (name movers, S-inhibition), S2 (induction, duplicate token) or S1+1 (previous token)."""
    assert KEEP_POSITION == {"name mover": "end", "backup name mover": "end", "negative name mover": "end",
                             "s2 inhibition": "end", "induction": "S2", "duplicate token": "S2",
                             "previous token": "S1+1"}
