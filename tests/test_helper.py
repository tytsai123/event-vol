import numpy as np
import pandas as pd
from helpers import event_vol_research as h

def test_pipeline():
    dates=pd.bdate_range('2026-08-26','2026-09-11')
    schedule=pd.DataFrame({'market_open':dates.tz_localize('America/New_York').tz_convert('UTC')+pd.Timedelta(hours=9,minutes=30),'market_close':dates.tz_localize('America/New_York').tz_convert('UTC')+pd.Timedelta(hours=16)},index=dates)
    ts=pd.Timestamp('2026-08-26 19:30',tz='UTC'); S=210.; q=[]; d=[]
    for expiry in ['2026-08-28','2026-09-04','2026-09-11']:
        exp=pd.Timestamp(expiry,tz='UTC'); end=schedule.loc[expiry,'market_close']; _,_,T,tau=h.calculate_variance_time(ts,end,schedule)
        F=S*np.exp(.04*T); D=np.exp(-.04*T); vol=np.sqrt(.004/tau+.12)
        for K in [195,200,205,210,215,220,225]:
            for cp in ['C','P']:
                iid=len(d)+1; p=h.black_price(F,K,D,tau,vol,cp)
                d.append(dict(ts_recv=ts-pd.Timedelta(hours=1),instrument_id=iid,strike_price=K,expiration=exp,instrument_class=cp))
                q.append(dict(ts_recv=ts,instrument_id=iid,bid_px_00=p-.005,ask_px_00=p+.005))
    e=pd.DataFrame([dict(ts_recv=ts,bid_px_00=S-.01,ask_px_00=S+.01)])
    res=h.run_pipeline('2026-08-26',raw_data=(pd.DataFrame(q),pd.DataFrame(d),e),schedule=schedule,progress=False)
    assert len(res['clean_df'])==42
    assert res['clean_df'].mid_iv.notna().all()
    assert np.allclose(res['atm_df'].total_variance_quadratic,.004+.12*res['atm_df'].T_variance)
    result=h.fit_event_variance(res['atm_df'])
    assert np.isclose(result['event_variance'],.004)
    assert np.isclose(result['base_variance'],.12)
    assert len(h.fit_event_series(res['atm_df']))==1
    print('Synthetic pipeline, IV inversion, and event OLS passed.')
    h.plot_feature_heatmaps(res['clean_df'],[('iv_failed','mean','Failure','.1%',1,'YlOrRd')]); h.plot_atm_term_structure(res['atm_df'])
    print('Plot smoke tests passed.')

def test_event_aware_regression():

    ts = pd.Timestamp("2026-11-13 20:30", tz="UTC")
    event_time = ts + pd.Timedelta(days=4)

    true_event_variance = 0.004
    true_base_variance = 0.12

    days = np.array([1., 3., 7., 14., 28.])
    tau = days / 365

    contains_event = days > 4

    df = pd.DataFrame({
        "ts_recv": ts,
        "T_calendar": tau,
        "T_variance": tau,
        "total_variance_quadratic": (
            true_base_variance * tau
            + true_event_variance * contains_event
        )
    })

    result = h.fit_event_variance(
        df,
        event_time=event_time
    )

    assert result["num_expiries"] == 3
    assert np.isclose(
        result["event_variance"], true_event_variance
    )
    assert np.isclose(
        result["base_variance"], true_base_variance
    )
    assert np.isclose(result["r_squared"], 1.0)

    # Test time series function
    series = h.fit_event_series(
        df,
        event_time=event_time
    )

    assert series.iloc[0]["num_expiries"] == 3
    assert np.isclose(
        series.iloc[0]["event_variance"], true_event_variance
    )

    print("Event-aware regression test passed.")

if __name__=='__main__':
    test_pipeline()
    test_event_aware_regression()
