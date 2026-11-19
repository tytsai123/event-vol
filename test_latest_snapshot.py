import pandas as pd
import pytest

import helper as h


def make_atm_df():
    times = pd.date_range('2026-11-19 19:31', periods=3, freq='14min', tz='UTC')
    expiries = pd.to_datetime(['2026-11-20', '2026-11-27', '2026-12-04'], utc=True)
    rows = []
    for i, ts in enumerate(times):
        for j, expiry in enumerate(expiries):
            rows.append({
                'ts_recv': ts,
                'expiration': expiry,
                'total_variance_quadratic': float(i + j + 1),
            })
    return pd.DataFrame(rows)


def test_latest_snapshot_returns_all_expiries_from_last_minute():
    atm_df = make_atm_df()

    snapshot = h.latest_snapshot(atm_df)

    assert snapshot['ts_recv'].nunique() == 1
    assert snapshot['ts_recv'].iloc[0] == atm_df['ts_recv'].max()
    assert set(snapshot['expiration']) == set(atm_df['expiration'])
    assert len(snapshot) == 3


def test_latest_snapshot_skips_minute_with_all_missing_variance():
    atm_df = make_atm_df()
    last_ts = atm_df['ts_recv'].max()
    atm_df.loc[atm_df['ts_recv'] == last_ts, 'total_variance_quadratic'] = float('nan')

    snapshot = h.latest_snapshot(atm_df)

    assert snapshot['ts_recv'].nunique() == 1
    assert snapshot['ts_recv'].iloc[0] == atm_df['ts_recv'].unique()[-2]
    assert len(snapshot) == 3


def test_latest_snapshot_respects_cutoff():
    atm_df = make_atm_df()
    cutoff = atm_df['ts_recv'].iloc[3]

    snapshot = h.latest_snapshot(atm_df, cutoff=cutoff)

    assert snapshot['ts_recv'].nunique() == 1
    assert snapshot['ts_recv'].iloc[0] == cutoff


def test_latest_snapshot_raises_when_nothing_is_eligible():
    atm_df = make_atm_df()
    atm_df['total_variance_quadratic'] = float('nan')

    with pytest.raises(ValueError):
        h.latest_snapshot(atm_df)


def test_latest_snapshot_does_not_mutate_input():
    atm_df = make_atm_df()
    before = atm_df.copy(deep=True)

    snapshot = h.latest_snapshot(atm_df)
    snapshot.iloc[0, snapshot.columns.get_loc('total_variance_quadratic')] = -1

    pd.testing.assert_frame_equal(atm_df, before)
