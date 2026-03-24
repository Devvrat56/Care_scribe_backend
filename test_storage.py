import os
import shutil
from storage_utils import save_transcript, save_entities, save_summary

def test_storage():
    test_id = "test-session-123"
    test_transcript = "This is a test transcript."
    test_entities = [{"text": "Aspirin", "label": "CHEMICAL"}]
    test_summary = "# Test Summary\n\n- Advice: Rest more."

    # Test saving
    trans_path = save_transcript(test_id, test_transcript)
    ent_path = save_entities(test_id, test_entities)
    sum_path = save_summary(test_id, test_summary)

    print(f"Transcript saved to: {trans_path}")
    print(f"Entities saved to: {ent_path}")
    print(f"Summary saved to: {sum_path}")

    # Verify existence
    assert os.path.exists(trans_path)
    assert os.path.exists(ent_path)
    assert os.path.exists(sum_path)

    # Verify content
    with open(trans_path, "r") as f:
        assert f.read() == test_transcript
    
    import json
    with open(ent_path, "r") as f:
        assert json.load(f) == test_entities
    
    with open(sum_path, "r") as f:
        assert f.read() == test_summary

    print("All storage tests passed!")

if __name__ == "__main__":
    if os.path.exists("storage/test-session-123"):
        shutil.rmtree("storage/test-session-123")
    test_storage()
