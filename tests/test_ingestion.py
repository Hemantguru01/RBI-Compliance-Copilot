import shutil

import pymupdf

from app.utils.fixtures import write_corrupted_pdf


def test_ingests_all_fixture_documents(workspace, make_container):
    root, raw = workspace
    c = make_container(root, raw)
    result = c.ingest()
    assert sorted(result.added) == sorted(p.name for p in raw.glob("*.pdf"))
    assert result.total_documents == 3 and result.total_chunks == len(c.store.chunks) > 3
    assert c.store.index_path.exists() and c.store.chunks_path.exists()
    assert len(list((root / "processed").glob("*.json"))) == 3


def test_second_run_is_incremental(workspace, make_container):
    root, raw = workspace
    c = make_container(root, raw)
    c.ingest()
    vectors_before = c.store.index.ntotal
    again = c.ingest()
    assert again.added == [] and again.updated == [] and len(again.unchanged) == 3
    assert c.store.index.ntotal == vectors_before


def test_index_persists_across_restart(workspace, make_container):
    root, raw = workspace
    make_container(root, raw).ingest()
    restarted = make_container(root, raw)
    assert restarted.index_ready and len(restarted.store.documents) == 3
    assert restarted.retriever.retrieve("cooling-off period for digital loans")


def test_modified_and_deleted_files_are_detected(workspace, make_container):
    root, raw = workspace
    c = make_container(root, raw)
    c.ingest()
    target = raw / "fixture_cyber_security.pdf"
    doc = pymupdf.open(target)
    doc[0].insert_text((40, 780), "5. Banks shall conduct a red-team exercise every year.")
    doc.saveIncr()
    doc.close()
    (raw / "fixture_digital_lending.pdf").unlink()
    result = c.ingest()
    assert result.updated == ["fixture_cyber_security.pdf"]
    assert result.removed == ["fixture_digital_lending.pdf"]
    assert result.total_documents == 2
    assert any("red-team exercise" in ch.text for ch in c.store.chunks.values())
    assert not any(ch.source_filename == "fixture_digital_lending.pdf" for ch in c.store.chunks.values())


def test_duplicates_and_corrupted_files_are_skipped(workspace, make_container):
    root, raw = workspace
    shutil.copy(raw / "fixture_grievance_redress.pdf", raw / "zz_copy_of_grievance.pdf")
    write_corrupted_pdf(raw / "broken.pdf")
    c = make_container(root, raw)
    result = c.ingest()
    assert "duplicate of fixture_grievance_redress.pdf" in result.skipped["zz_copy_of_grievance.pdf"]
    assert "Cannot open" in result.skipped["broken.pdf"]
    assert result.total_documents == 3


def test_force_rebuild(workspace, make_container):
    root, raw = workspace
    c = make_container(root, raw)
    c.ingest()
    result = c.ingest(force=True)
    assert len(result.added) == 3 and result.total_documents == 3


def test_corrupted_index_is_rebuilt_on_startup(workspace, make_container):
    root, raw = workspace
    make_container(root, raw).ingest()
    (root / "vector_store" / "index.faiss").write_bytes(b"corrupt")
    c = make_container(root, raw)
    assert c.index_ready and len(c.store.documents) == 3
    assert any("rebuilt" in m for m in c.messages)


def test_empty_directory_yields_empty_index(tmp_path, make_container):
    raw = tmp_path / "raw"
    raw.mkdir()
    c = make_container(tmp_path, raw)
    result = c.ingest()
    assert result.total_documents == 0 and not c.index_ready
