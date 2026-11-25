"""Local option research helpers; no network access or downloads on import/run.
Prices are per share; volatility in decimal units; times in ACT/365 years.
European Black-76 is a baseline approximation to American equity options.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.optimize import brentq

KEYS = ['ts_recv', 'expiration']
ANNUAL_CALENDAR_HOURS = 365 * 24
ANNUAL_TRADING_HOURS = 252 * 6.5


def load_data(date, ticker='NVDA', data_dir='data/raw', *, window='close'):
    """Return (option_quotes, definitions, equity_quotes) from existing files."""
    import databento as db
    day = pd.Timestamp(date).strftime('%Y-%m-%d')
    if window == 'close':
        paths = [Path(data_dir) / f'{ticker}_{dataset}_{schema}_{day}.dbn.zst'
                 for dataset, schema in [('opra.pillar','cbbo-1m'),
                                         ('opra.pillar','definition'), ('equs.mini','bbo-1m')]]
    elif window == 'open':
        from .download_data import build_requests

        paths = [path for path, _ in build_requests(day, ticker, data_dir, window=window)]
    else:
        raise ValueError("window must be 'open' or 'close'")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    # Databento currently uses pandas' deprecated inplace index setter.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The inplace keyword in DataFrame\.set_index is deprecated",
            module=r"databento\.common\.dbnstore$",
        )
        return tuple(db.DBNStore.from_file(path).to_df() for path in paths)


def _quotes(df):
    out = df.copy() if 'ts_recv' in df.columns else df.reset_index()
    out['ts_recv'] = pd.to_datetime(out['ts_recv'], utc=True)
    return out


def data_quality_check(option_df, definition_df):
    q, d = set(option_df.instrument_id.unique()), set(definition_df.instrument_id.unique())
    bid, ask = option_df.bid_px_00, option_df.ask_px_00
    return pd.Series(dict(quote_rows=len(option_df), quote_instruments=len(q),
        definition_instruments=len(d), common_instruments=len(q & d),
        quote_only=len(q-d), definition_only=len(d-q), crossed=int((bid>ask).sum()),
        locked=int((bid==ask).sum()), zero_bid=int((bid==0).sum()), zero_ask=int((ask==0).sum()),
        missing_bid_only=int((bid.isna() & ask.notna()).sum()),
        missing_ask_only=int((ask.isna() & bid.notna()).sum()),
        missing_both=int((bid.isna() & ask.isna()).sum())))


def merge_data(option_df, definition_df, equity_df, spot_tolerance='1min'):
    q, d, e = map(_quotes, (option_df, definition_df, equity_df))
    d = d[['instrument_id','strike_price','expiration','instrument_class','ts_recv']].rename(columns={'ts_recv':'definition_time'})
    if d.duplicated(['instrument_id','definition_time']).any():
        raise ValueError('Duplicate definition keys; inspect before merging')
    out = pd.merge_asof(q.sort_values('ts_recv'), d.sort_values('definition_time'),
        left_on='ts_recv', right_on='definition_time', by='instrument_id', direction='backward')
    e = e[['ts_recv','bid_px_00','ask_px_00']].rename(columns={'ts_recv':'spot_time','bid_px_00':'spot_bid','ask_px_00':'spot_ask'})
    if e.spot_time.duplicated().any():
        raise ValueError('Duplicate equity timestamps')
    valid = (e.spot_bid > 0) & (e.spot_ask >= e.spot_bid) & np.isfinite(e.spot_ask)
    e['spot_mid'] = ((e.spot_bid+e.spot_ask)/2).where(valid)
    out = pd.merge_asof(out.sort_values('ts_recv'), e.sort_values('spot_time'),
        left_on='ts_recv', right_on='spot_time', direction='backward', tolerance=pd.Timedelta(spot_tolerance))
    out['expiration'] = pd.to_datetime(out.expiration, utc=True)
    out['spot_age_seconds'] = (out.ts_recv-out.spot_time).dt.total_seconds()
    return add_features(out)


def add_features(df, moneyness_bins=None):
    out = df.copy()
    phi = out.instrument_class.map({'C':1,'P':-1})
    with np.errstate(divide='ignore', invalid='ignore'):
        out['log_spot_moneyness'] = np.log(out.strike_price/out.spot_mid)
        out['signed_log_spot_moneyness'] = phi*out.log_spot_moneyness
        out['moneyness'] = out.signed_log_spot_moneyness
        base = out.log_spot_moneyness
        if 'forward' in out:
            out['log_forward_moneyness'] = np.log(out.strike_price/out.forward)
            out['signed_log_forward_moneyness'] = phi*out.log_forward_moneyness
            base = out.log_forward_moneyness
    expiry = pd.to_datetime(out.expiration, utc=True).dt.tz_localize(None).dt.normalize()
    quote = pd.to_datetime(out.ts_recv, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
    out['dte'] = (expiry-quote).dt.days
    out['dte_bin'] = pd.cut(out.dte,[0,7,14,30,60,90,np.inf],labels=['0-7D','7-14D','14-30D','30-60D','60-90D','90D+'],include_lowest=True)
    bins = np.r_[-np.inf,np.arange(-5,5.25,.25),np.inf] if moneyness_bins is None else moneyness_bins
    out['moneyness_bin'] = pd.cut(base,bins)
    out['missing_bid'] = out.bid_px_00.isna()
    return out


def filter_quotes(df):
    bid, ask, spot = df.bid_px_00, df.ask_px_00, df.spot_mid
    mask = np.isfinite(bid)&np.isfinite(ask)&np.isfinite(spot)&(bid>0)&(ask>=bid)&(spot>0)&(df.strike_price>0)&np.isfinite(df.strike_price)&df.expiration.notna()&df.instrument_class.isin(['C','P'])
    out = df.loc[mask].copy()
    out['mid_px_00'] = (out.bid_px_00+out.ask_px_00)/2
    out['absolute_spread'] = out.ask_px_00-out.bid_px_00
    out['relative_spread'] = out.absolute_spread/out.mid_px_00
    return out


def add_variance_time(df, weight=.4, calendar='NYSE', schedule=None):
    """Use exchange close as expiry proxy, including early closes."""
    if not 0 <= weight <= 1: raise ValueError('weight must be in [0, 1]')
    out = df.drop(columns=['expiry_time','trading_hours','non_trading_hours','T_calendar','T_variance','variance_time_ratio'],errors='ignore').copy()
    time = out[KEYS].drop_duplicates().copy()
    expiry = pd.to_datetime(time.expiration,utc=True).dt.tz_localize(None).dt.normalize()
    if schedule is None:
        import pandas_market_calendars as mcal
        schedule = mcal.get_calendar(calendar).schedule(start_date=time.ts_recv.min().tz_convert('America/New_York').date(),end_date=expiry.max().date())
    closes = schedule.market_close.copy()
    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None).normalize()
    time['expiry_time'] = expiry.map(closes)
    if time.expiry_time.isna().any():
        raise ValueError(f'No exchange close for expiry dates: {expiry[time.expiry_time.isna()].unique()}')
    cols = ['trading_hours','non_trading_hours','T_calendar','T_variance']
    time[cols] = pd.DataFrame([calculate_variance_time(r.ts_recv,r.expiry_time,schedule,weight) for r in time.itertuples()],index=time.index,columns=cols)
    out = out.merge(time,on=KEYS,how='left',validate='many_to_one')
    out['variance_time_ratio'] = out.T_variance/out.T_calendar
    return out


def build_call_put_pairs(df, spread_floor=.01):
    if spread_floor <= 0: raise ValueError('spread_floor must be positive')
    keys = KEYS+['strike_price']
    values = ['bid_px_00','ask_px_00','mid_px_00','absolute_spread']
    data = df[df.T_calendar.gt(0)]
    pairs = data.pivot(index=keys,columns='instrument_class',values=values)
    pairs = pairs.reindex(columns=pd.MultiIndex.from_product([values,['C','P']],names=pairs.columns.names))
    pairs.columns = [f'{"call" if cp=="C" else "put"}_{field}' for field,cp in pairs.columns]
    pairs = pairs.reset_index().dropna(subset=['call_mid_px_00','put_mid_px_00']).copy()
    pairs['call_minus_put'] = pairs.call_mid_px_00-pairs.put_mid_px_00
    pairs['weight'] = 1/np.maximum(pairs.call_absolute_spread**2+pairs.put_absolute_spread**2,spread_floor**2)
    pairs = pairs.merge(df[KEYS+['T_calendar','T_variance','spot_mid']].drop_duplicates(),on=KEYS,how='left',validate='many_to_one')
    pairs['log_moneyness'] = np.log(pairs.strike_price/pairs.spot_mid)
    return pairs


def calibrate_forward(pair_df, r=.04, cutoff=.10, min_pairs=3):
    rows=[]
    for (ts,expiry), group in pair_df.groupby(KEYS):
        g=group[group.log_moneyness.abs()<=cutoff]
        if g.strike_price.nunique()<min_pairs: continue
        F=estimate_forward_fixed_rate(g,r)
        rows.append(dict(ts_recv=ts,expiration=expiry,forward=F if F>0 else np.nan,
                         discount_factor=np.exp(-r*g.T_calendar.iloc[0]),num_pairs=len(g)))
    if not rows:
        return pair_df[KEYS].iloc[:0].assign(forward=np.nan,discount_factor=np.nan,num_pairs=0)
    return pd.DataFrame(rows)


def calculate_iv(df, progress=True, sigma_max=5.):
    out=df.copy()
    rows=out.itertuples(index=False)
    if progress:
        from tqdm.auto import tqdm
        rows=tqdm(rows,total=len(out),desc='IV solver')
    results=[]
    for row in rows:
        args=(row.forward,row.strike_price,row.discount_factor,row.T_variance,row.instrument_class)
        results.append([implied_vol(p,*args,sigma_max=sigma_max) for p in (row.bid_px_00,row.mid_px_00,row.ask_px_00)])
    out[['bid_iv','mid_iv','ask_iv']]=np.array(results).reshape(-1,3)
    out['iv_spread']=out.ask_iv-out.bid_iv
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        std=out.mid_iv*np.sqrt(out.T_variance)
        d1=np.log(out.forward/out.strike_price)/std+.5*std
        out['vega']=(out.discount_factor*out.forward*np.sqrt(out.T_variance)*np.exp(-.5*d1**2)/np.sqrt(2*np.pi)).where(std>0)
        out['vega_per_vol_point']=out.vega/100
        out['vega_adjusted_spread']=out.absolute_spread/out.vega.where(out.vega>0)
    out['iv_failed']=out.mid_iv.isna()
    return add_features(out)


def build_surface(df):
    return df.loc[(df.signed_log_forward_moneyness>=0)&(df.T_variance>0)&(df.mid_iv>0)&np.isfinite(df.mid_iv)].copy()


def extract_atm_features(surface_df, cutoff=.10, spread_floor=.005):
    rows=[]
    for (ts,expiry),g in surface_df.groupby(KEYS):
        a,b,c=estimate_atm_iv_quadratic(g,cutoff,spread_floor)
        rows.append(dict(ts_recv=ts,expiration=expiry,T_calendar=g.T_calendar.iloc[0],T_variance=g.T_variance.iloc[0],forward=g.forward.iloc[0],atm_iv_linear=estimate_atm_iv_linear(g),atm_iv_quadratic=a if a>0 else np.nan,atm_skew=b,atm_curvature=c))
    cols=KEYS+['T_calendar','T_variance','forward','atm_iv_linear','atm_iv_quadratic','atm_skew','atm_curvature']
    out=pd.DataFrame(rows,columns=cols)
    for method in ['linear','quadratic']:
        out[f'total_variance_{method}']=out[f'atm_iv_{method}']**2*out.T_variance
    return out


def latest_snapshot(atm_df, cutoff=None, method='quadratic'):
    """Return all expiries from the latest valid same-minute ATM snapshot.

    ATM rows are fitted from quotes at each (ts_recv, expiration) cross-section,
    so expiries must come from one shared timestamp rather than independently
    selected per-instrument last quotes. A timestamp is eligible when its
    selected total variance is present, and, if supplied, it is at or before
    the timezone-aware cutoff.
    """
    variance_col = f'total_variance_{method}'
    eligible = atm_df.loc[atm_df[variance_col].notna()]
    if cutoff is not None:
        cutoff = pd.Timestamp(cutoff)
        if cutoff.tzinfo is None:
            raise ValueError('cutoff must be timezone-aware')
        eligible = eligible.loc[eligible['ts_recv'] <= cutoff]
    if eligible.empty:
        raise ValueError('No eligible ATM snapshot rows')
    latest_ts = eligible['ts_recv'].max()
    return atm_df.loc[atm_df['ts_recv'] == latest_ts].copy()


def run_pipeline(date, ticker='NVDA', data_dir='data/raw', *, rate=.04, off_hour_weight=.4, forward_cutoff=.10, atm_cutoff=.10, iv_spread_floor=.005, progress=True, calendar='NYSE', raw_data=None, schedule=None):
    """Return a dict of intermediate frames and diagnostics; raw_data optional for testing."""
    q,d,e=load_data(date,ticker,data_dir) if raw_data is None else raw_data
    quality=data_quality_check(q,d)
    merged=merge_data(q,d,e)
    clean=filter_quotes(merged)
    if clean.empty: raise ValueError('No valid two-sided quotes')
    clean=add_variance_time(clean,off_hour_weight,calendar,schedule)
    pairs=build_call_put_pairs(clean)
    forward=calibrate_forward(pairs,rate,forward_cutoff)
    clean=clean.merge(forward,on=KEYS,how='left',validate='many_to_one')
    clean=calculate_iv(clean,progress)
    surface=build_surface(clean)
    atm=extract_atm_features(surface,atm_cutoff,iv_spread_floor)
    quality=pd.concat([quality,pd.Series(dict(merged_rows=len(merged),missing_definition=int(merged.definition_time.isna().sum()),missing_spot=int(merged.spot_mid.isna().sum()),clean_rows=len(clean),missing_forward=int(clean.forward.isna().sum()),mid_iv_failures=int(clean.iv_failed.sum()),surface_rows=len(surface)))])
    return dict(merged_df=merged,clean_df=clean,pair_df=pairs,forward_df=forward,surface_df=surface,atm_df=atm,quality=quality)


def fit_event_series(atm_df,max_dte=31,method='quadratic',event_time=None):
    """Fit pre-event snapshots only if event_time supplied; no post-event intercept interpreted as event."""
    rows=[]
    if event_time is not None:
        event=pd.Timestamp(event_time)
        if event.tzinfo is None: raise ValueError('event_time must be timezone-aware')
        # Reconstruct model expiry timestamp using calendar time, not UTC midnight label.
        expiry=atm_df.ts_recv+pd.to_timedelta(atm_df.T_calendar*365,unit='D')
        atm_df=atm_df[(atm_df.ts_recv<event)&(expiry>event)]
    for ts,g in atm_df.groupby('ts_recv'):
        try: result=fit_event_variance(g,max_dte,method)
        except ValueError: result=dict(event_variance=np.nan,base_variance=np.nan,event_move=np.nan,base_vol=np.nan,rmse=np.nan,r_squared=np.nan,num_expiries=int(((g.T_calendar*365<=max_dte)&(g.T_variance>0)&g[f'total_variance_{method}'].notna()).sum()))
        rows.append(dict(ts_recv=ts,method=method,max_dte=max_dte,**result))
    return pd.DataFrame(rows,columns=['ts_recv','method','max_dte','event_variance','base_variance','event_move','base_vol','rmse','r_squared','num_expiries'])


def calculate_variance_time(start, end, schedule, weight=0.4):

    if end <= start:
        return np.nan, np.nan, np.nan, np.nan

    calendar_hours = (end - start).total_seconds() / 3600

    overlap_start = schedule["market_open"].clip(lower=start)
    overlap_end = schedule["market_close"].clip(upper=end)

    trading_hours = (
        (overlap_end - overlap_start)
        .dt.total_seconds()
        .clip(lower=0)
        .sum() / 3600
    )

    non_trading_hours = calendar_hours - trading_hours

    variance_hours = (
        trading_hours + weight * non_trading_hours
    )

    annual_variance_hours = (
        ANNUAL_TRADING_HOURS
        + weight * (
            ANNUAL_CALENDAR_HOURS - ANNUAL_TRADING_HOURS
        )
    )

    T_calendar = calendar_hours / ANNUAL_CALENDAR_HOURS
    T_variance = variance_hours / annual_variance_hours

    return (
        trading_hours,
        non_trading_hours,
        T_calendar,
        T_variance
    )


def estimate_forward_fixed_rate(df, r=0.04):

    T = df["T_calendar"].iloc[0]

    F = (
        df["strike_price"]
        + np.exp(r * T) * df["call_minus_put"]
    )

    return np.average(F, weights=df["weight"])


def estimate_forward_wls(df):

    K = df["strike_price"].to_numpy()
    y = df["call_minus_put"].to_numpy()
    w = df["weight"].to_numpy()
    T = df["T_calendar"].iloc[0]

    X = np.column_stack([np.ones(len(K)), K])

    sqrt_w = np.sqrt(w)

    X_w = X * sqrt_w[:, None]
    y_w = y * sqrt_w

    params, _, rank, _ = np.linalg.lstsq(
        X_w, y_w, rcond=None
    )

    a, b = params

    if rank < 2 or b >= 0 or T <= 0:
        return np.nan, np.nan, np.nan

    discount_factor = -b
    F = -a / b
    r = -np.log(discount_factor) / T

    return F, r, discount_factor


from scipy.special import ndtr

def black_price(F, K, D, tau, sigma, cp):

    phi = 1 if cp == "C" else -1

    if tau <= 0 or sigma <= 0:
        return D * max(phi * (F - K), 0.0)

    std = sigma * np.sqrt(tau)

    d1 = np.log(F / K) / std + 0.5 * std
    d2 = d1 - std

    return D * phi * (
        F * ndtr(phi * d1)
        - K * ndtr(phi * d2)
    )


from scipy.optimize import brentq

def implied_vol(price, F, K, D, tau, cp, sigma_max=5.0):

    inputs = [price, F, K, D, tau]

    if not np.isfinite(inputs).all():
        return np.nan

    if min(inputs) <= 0 or cp not in ("C", "P"):
        return np.nan

    phi = 1 if cp == "C" else -1

    lower = D * max(phi * (F - K), 0.0)
    upper = D * (F if cp == "C" else K)

    if price < lower or price >= upper:
        return np.nan

    if price == lower:
        return 0.0

    def objective(sigma):
        return black_price(F, K, D, tau, sigma, cp) - price

    sigma_min = 0.0

    if objective(sigma_max) < 0:
        return np.nan

    return brentq(
        objective,
        sigma_min,
        sigma_max,
        xtol=1e-10,
        maxiter=100
    )


def estimate_atm_iv_linear(df, iv_col="mid_iv"):

    df = df[
        np.isfinite(df[iv_col])
        & (df[iv_col] > 0)
    ].sort_values("log_forward_moneyness")

    grouped = df.groupby("log_forward_moneyness")[iv_col].mean()
    k = grouped.index.to_numpy()
    iv = grouped.to_numpy()

    if len(k) == 0 or k.min() > 0 or k.max() < 0:
        return np.nan

    return np.interp(0.0, k, iv)


def estimate_atm_iv_quadratic(
    df, cutoff=0.10, spread_floor=0.005
):

    subset = df[
        (df["log_forward_moneyness"].abs() <= cutoff)
        & np.isfinite(df["mid_iv"])
        & (df["mid_iv"] > 0)
        & np.isfinite(df["iv_spread"])
        & (df["iv_spread"] >= 0)
    ]

    k = subset["log_forward_moneyness"].to_numpy()
    y = subset["mid_iv"].to_numpy()
    spread = subset["iv_spread"].to_numpy()

    if len(np.unique(k)) < 3 or k.min() > 0 or k.max() < 0:
        return np.nan, np.nan, np.nan

    X = np.column_stack([
        np.ones(len(k)), k, k**2
    ])

    sqrt_w = 1 / np.maximum(spread, spread_floor)

    params, _, rank, _ = np.linalg.lstsq(
        X * sqrt_w[:, None],
        y * sqrt_w,
        rcond=None
    )

    if rank < 3:
        return np.nan, np.nan, np.nan

    a, b, c = params

    return a, b, 2 * c


def fit_event_variance(
    df,
    max_dte=31,
    method="quadratic",
    event_time=None
):

    if df["ts_recv"].nunique() > 1:
        raise ValueError("Use one timestamp or fit_event_series")

    subset = df[
        (df["T_calendar"] * 365 <= max_dte)
        & (df["T_variance"] > 0)
    ].dropna(
        subset=[f"total_variance_{method}"]
    ).copy()

    # Exclude expiries that do not contain the event
    if event_time is not None:

        event = pd.Timestamp(event_time)

        if event.tzinfo is None:
            raise ValueError("event_time must be timezone-aware")

        expiry = subset["ts_recv"] + pd.to_timedelta(
            subset["T_calendar"] * 365,
            unit="D"
        )

        subset = subset[
            (subset["ts_recv"] < event)
            & (expiry > event)
        ].copy()

    x = subset["T_variance"].to_numpy()
    y = subset[f"total_variance_{method}"].to_numpy()

    X = np.column_stack([np.ones(len(x)), x])

    params, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)

    if len(x) < 3 or rank < 2:
        raise ValueError("Insufficient data for regression")

    a, b = params

    y_pred = X @ params
    residual = y - y_pred

    rmse = np.sqrt(np.mean(residual**2))
    sst = np.sum((y - y.mean())**2)
    r_squared = 1 - np.sum(residual**2) / sst if sst > 0 else np.nan

    return {
        "event_variance": a,
        "base_variance": b,
        "event_move": np.sqrt(a) if a >= 0 else np.nan,
        "base_vol": np.sqrt(b) if b >= 0 else np.nan,
        "rmse": rmse,
        "r_squared": r_squared,
        "num_expiries": len(x)
    }


def plot_feature_heatmaps(df, features, figsize=(24,10), moneyness_col='moneyness_bin', ylabel='Log Forward Moneyness'):
    """features: (column, aggregation, title, format, scale, colormap) tuples."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    keys=[moneyness_col,'dte_bin']
    levels=df[keys].dropna()
    idx=pd.Index(levels[moneyness_col].unique()).sort_values()
    cols=pd.Index(levels.dte_bin.unique()).sort_values() if not isinstance(levels.dte_bin.dtype,pd.CategoricalDtype) else pd.Index([v for v in levels.dte_bin.cat.categories if v in set(levels.dte_bin)])
    if not len(idx) or not len(cols): raise ValueError('No observations in heatmap bins')
    fig, axes=plt.subplots(1,len(features),figsize=figsize,squeeze=False,sharey=True)
    matrices=[]
    for feature,agg,title,fmt,scale,cmap in features:
        matrix=pd.pivot_table(df,values=feature,index=keys[0],columns=keys[1],aggfunc=agg,observed=True).reindex(index=idx,columns=cols)*scale
        matrices.append(matrix)
    for ax,(feature,agg,title,fmt,scale,cmap),matrix in zip(axes.flat,features,matrices):
        im=ax.imshow(matrix.to_numpy(dtype=float),origin='lower',aspect='auto',cmap=cmap)
        ax.set_xticks(range(len(cols)),labels=cols,rotation=45)
        ax.set_yticks(range(len(idx)),labels=idx.astype(str))
        ax.set(xlabel='Days to Expiry',title=title)
        for i,j in np.argwhere(np.isfinite(matrix.to_numpy(dtype=float))):
            v=matrix.iloc[i,j]
            ax.text(j,i,format(v,fmt),ha='center',va='center',fontsize=8,color='white' if im.norm(v)>.6 else 'black')
        cb=fig.colorbar(im,ax=ax)
        if fmt.endswith('%'): cb.ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0,0].set_ylabel(ylabel)
    fig.tight_layout()
    return fig,axes


def plot_atm_term_structure(atm_df, timestamp=None, max_dte=None):
    import matplotlib.pyplot as plt
    ts=atm_df.ts_recv.min() if timestamp is None else pd.Timestamp(timestamp)
    s=atm_df[atm_df.ts_recv.eq(ts)].sort_values('T_variance')
    if max_dte is not None: s=s[s.T_calendar*365<=max_dte]
    if s.empty: raise ValueError('No matching snapshot')
    fig,axes=plt.subplots(2,2,figsize=(16,10))
    x=s.T_variance*365
    for method in ['linear','quadratic']:
        axes[0,0].plot(x,s[f'atm_iv_{method}']*100,'o-',label=method)
        axes[0,1].plot(x,s[f'total_variance_{method}'],'o-',label=method)
    axes[1,0].plot(x,s.atm_skew,'o-'); axes[1,1].plot(x,s.atm_curvature,'o-')
    for ax,title in zip(axes.flat,['ATM IV (%)','ATM Total Variance','ATM Skew','ATM Curvature']):
        ax.set(title=title,xlabel='Variance Time (Days)'); ax.grid(alpha=.2)
    for ax in axes[0]: ax.legend()
    fig.tight_layout()
    return fig,axes


def compare_pre_post(atm_df, pre_time, post_time, base_vol, method='quadratic'):
    """Matched expiries; assumes unchanged base variance rate and no other intervening events."""
    if pd.Timestamp(post_time)<=pd.Timestamp(pre_time): raise ValueError('post must follow pre')
    col=f'total_variance_{method}'
    fields=['expiration','T_variance',col,f'atm_iv_{method}','atm_skew','atm_curvature']
    pre=atm_df.loc[atm_df.ts_recv.eq(pd.Timestamp(pre_time)),fields]
    post=atm_df.loc[atm_df.ts_recv.eq(pd.Timestamp(post_time)),fields]
    out=pre.merge(post,on='expiration',suffixes=('_pre','_post'),validate='one_to_one')
    out['base_variance_elapsed']=base_vol**2*(out.T_variance_pre-out.T_variance_post)
    out['event_variance']=out[f'{col}_pre']-out[f'{col}_post']-out.base_variance_elapsed
    out['event_move']=np.sqrt(out.event_variance.where(out.event_variance>=0))
    return out


def forward_sensitivity(pair_df, cutoffs=(.05,.10,.20,np.inf), r=.04):
    """Diagnostic: joint F/r estimation only, not production discounting."""
    rows=[]
    for (ts,expiry),g in pair_df.groupby(KEYS):
        for cutoff in cutoffs:
            s=g[g.log_moneyness.abs()<=cutoff]
            if s.strike_price.nunique()<3: continue
            F,rate,D=estimate_forward_wls(s)
            rows.append(dict(ts_recv=ts,expiration=expiry,T_calendar=s.T_calendar.iloc[0],cutoff=cutoff,F=F,r=rate,D=D,F_fixed=estimate_forward_fixed_rate(s,r),num_pairs=len(s)))
    return pd.DataFrame(rows)


def parity_residuals(pair_df,timestamp,expiration,cutoff=.10):
    s=pair_df[pair_df.ts_recv.eq(pd.Timestamp(timestamp))&pair_df.expiration.eq(pd.Timestamp(expiration))].copy()
    g=s[s.log_moneyness.abs()<=cutoff]
    if g.strike_price.nunique()<3: raise ValueError('Insufficient near-ATM pairs')
    F,r,D=estimate_forward_wls(g)
    s['fitted']=D*(F-s.strike_price)
    s['residual']=s.call_minus_put-s.fitted
    s['residual_lower']=s.call_bid_px_00-s.put_ask_px_00-s.fitted
    s['residual_upper']=s.call_ask_px_00-s.put_bid_px_00-s.fitted
    return s.sort_values('log_moneyness')


def plot_smiles(surface_df,timestamp=None,xlim=(-.3,.3),ncols=4):
    import matplotlib.pyplot as plt
    ts=surface_df.ts_recv.min() if timestamp is None else pd.Timestamp(timestamp)
    s=surface_df[surface_df.ts_recv.eq(ts)]
    groups=list(s.groupby('expiration'))
    if not groups: raise ValueError('No matching snapshot')
    fig,axes=plt.subplots(int(np.ceil(len(groups)/ncols)),ncols,figsize=(5*ncols,4*int(np.ceil(len(groups)/ncols))),squeeze=False)
    for ax,(expiry,g) in zip(axes.flat,groups):
        g=g.sort_values('log_forward_moneyness'); x=g.log_forward_moneyness.to_numpy(dtype=float)
        ax.scatter(x,g.mid_iv.to_numpy()*100,s=10,label='Mid IV')
        ax.fill_between(x,g.bid_iv.to_numpy()*100,g.ask_iv.to_numpy()*100,alpha=.2,label='Bid-Ask IV')
        ax.axvline(0,ls='--',color='gray'); ax.set(title=str(pd.Timestamp(expiry).date()),xlabel='Log Forward Moneyness',ylabel='IV (%)',xlim=xlim); ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(groups):]: ax.set_visible(False)
    fig.tight_layout()
    return fig,axes


def event_cutoff_sensitivity(snapshot,cutoffs=(10,15,24,31,38,52,90,180,365,np.inf),methods=('linear','quadratic')):
    rows=[]
    for cutoff in cutoffs:
        for method in methods:
            try: result=fit_event_variance(snapshot,cutoff,method)
            except ValueError: continue
            rows.append(dict(max_dte=cutoff,method=method,**result))
    return pd.DataFrame(rows)


def plot_event_fit(snapshot,max_dte=31,method='quadratic',display_dte=60):
    import matplotlib.pyplot as plt
    result=fit_event_variance(snapshot,max_dte,method)
    a,b=result['event_variance'],result['base_variance']
    fig,ax=plt.subplots(figsize=(12,6))
    x=snapshot.T_variance*365; y=snapshot[f'total_variance_{method}']; fit=snapshot.T_calendar*365<=max_dte
    ax.scatter(x[fit],y[fit],label='Fit universe'); ax.scatter(x[~fit],y[~fit],label='Outside fit',alpha=.5)
    grid=np.linspace(0,display_dte,200)
    ax.plot(grid,a+b*grid/365,label='OLS fitted variance'); ax.plot(grid,b*grid/365,'--',label='Base variance')
    ax.fill_between(grid,b*grid/365,a+b*grid/365,alpha=.15,label='Intercept component')
    ax.set(xlim=(0,display_dte),xlabel='Variance Time (Days)',ylabel='ATM Total Variance',title='Event Variance Decomposition'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); return fig,ax
