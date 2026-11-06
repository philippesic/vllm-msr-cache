from benchmarks.research_checks.wait_opportunity_report import returning_transcript_floor


def transcript():
    return [dict(session_id='s',turn=0,success=True,payload=dict(prompt=list(range(64))),
                 output_token_ids=list(range(64,82))),
            dict(session_id='s',turn=1,success=True,payload=dict(prompt=list(range(94))),
                 output_token_ids=[94])]


def test_completed_prefix_floor_excludes_last_predicted_token_and_rounds_blocks():
    rows=transcript()
    assert returning_transcript_floor(rows)==14
    rows[1]['payload']['prompt'][20]=999
    assert returning_transcript_floor(rows)==78


def test_missing_or_failed_previous_generation_cannot_establish_a_floor():
    rows=transcript()
    assert returning_transcript_floor(rows[1:]) is None
    rows[0]['success']=False
    assert returning_transcript_floor(rows) is None
