from sources.squad_source import SquadSource


class _FakeDataset(list):
    pass


def test_squad_source_maps_rows(monkeypatch):
    fake_rows = _FakeDataset(
        [
            {
                "id": "abc",
                "title": "Title",
                "context": "Some context",
                "question": "What?",
                "answers": {"text": ["Answer 1", "Answer 2"]},
            }
        ]
    )

    def fake_load_dataset(name, split):
        assert name == "squad"
        assert split == "train[:10]"
        return fake_rows

    monkeypatch.setattr("sources.squad_source.load_dataset", fake_load_dataset)

    source = SquadSource()
    rows = list(source.load_records(split="train", sample_size=10))

    assert len(rows) == 1
    assert rows[0].doc_id == "abc"
    assert rows[0].answers == ["Answer 1", "Answer 2"]
