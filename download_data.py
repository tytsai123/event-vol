"""Download Databento research inputs compatible with helper.load_data()."""
from pathlib import Path
import math
import os
import tempfile

import pandas as pd
import pandas_market_calendars as mcal


def build_requests(date, ticker='NVDA', data_dir='data/raw', *, window='close'):
    """Build quote requests for the selected window and one shared definition file."""
    day = pd.Timestamp(date).strftime('%Y-%m-%d')
    ticker = ticker.upper().strip()
    if not ticker or not ticker.replace('.', '').replace('-', '').isalnum():
        raise ValueError('Invalid ticker')
    schedule = mcal.get_calendar('NYSE').schedule(start_date=day, end_date=day)
    if schedule.empty:
        raise ValueError(f'{day} is not a US equity trading day')
    session = schedule.iloc[0]
    minute = pd.Timedelta(minutes=1)
    if window == 'close':
        end = session['market_close'].tz_convert('UTC')
        start = end - 30 * minute
        suffix = ''
    elif window == 'open':
        market_open = session['market_open'].tz_convert('UTC')
        # BBO timestamps mark interval ends and Databento's request end is exclusive.
        start = market_open + minute
        end = market_open + 31 * minute
        suffix = '_open_30m'
    else:
        raise ValueError("window must be 'open' or 'close'")
    midnight = pd.Timestamp(day, tz='UTC')
    specs = [
        ('OPRA.PILLAR', 'cbbo-1m', ticker + '.OPT', 'parent', start, end, suffix),
        ('OPRA.PILLAR', 'definition', ticker + '.OPT', 'parent', midnight,
         midnight + pd.Timedelta(days=1), ''),
        ('EQUS.MINI', 'bbo-1m', ticker, 'raw_symbol', start, end, suffix),
    ]
    return [(Path(data_dir) / f'{ticker}_{dataset.lower()}_{schema}_{day}{tag}.dbn.zst',
             dict(dataset=dataset, schema=schema, symbols=[symbol], stype_in=stype,
                  start=a.isoformat(), end=b.isoformat()))
            for dataset, schema, symbol, stype, a, b, tag in specs]


def download_data(date, ticker='NVDA', data_dir='data/raw', *,
                  confirm=True, dry_run=False, max_cost=None, client=None, window='close'):
    """Estimate missing-file costs, then download after explicit confirmation.

    dry_run=True: estimate only. confirm=False: skip prompt (requires max_cost).
    max_cost: optional USD ceiling on estimated cost; actual charges may differ.
    Existing nonempty files are skipped, not overwritten. Returns a status table.
    client: optional injected Historical client for testing.
    """
    requests = build_requests(date, ticker, data_dir, window=window)
    if max_cost is not None and (not math.isfinite(max_cost) or max_cost < 0):
        raise ValueError('max_cost must be finite and nonnegative')
    rows = []
    pending = []
    for path, params in requests:
        exists = path.is_file() and path.stat().st_size > 0
        row = dict(path=str(path), dataset=params['dataset'], schema=params['schema'],
                   start=params['start'], end=params['end'], estimated_usd=0.0,
                   status='exists' if exists else 'pending')
        rows.append(row)
        if not exists:
            pending.append((path, params, row))
    if not pending:
        print('All files already exist; no API calls made.')
        return pd.DataFrame(rows)
    if client is None:
        import databento as db
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent / '.env', override=False)
        key = os.getenv('DATABENTO_API_KEY')
        if not key:
            raise RuntimeError('Set DATABENTO_API_KEY in .env or environment')
        client = db.Historical(key)
    for path, params, row in pending:
        cost = float(client.metadata.get_cost(**params))
        if not math.isfinite(cost) or cost < 0:
            raise RuntimeError('Invalid cost estimate; download aborted')
        row['estimated_usd'] = cost
        print(f'{path.name}: {params["start"]} -> {params["end"]} | ${cost:.4f}')
    total = sum(row['estimated_usd'] for row in rows)
    print(f'Total estimated cost: USD {total:.4f}')
    if dry_run:
        return pd.DataFrame(rows)
    if max_cost is not None and total > max_cost:
        raise RuntimeError(f'Estimated cost ${total:.4f} exceeds ${max_cost:.4f}')
    if confirm:
        if input('Proceed with download? [y/N] ').strip().lower() != 'y':
            print('Cancelled; no market data downloaded.')
            return pd.DataFrame(rows)
    elif max_cost is None:
        raise ValueError('Set max_cost when confirm=False')
    for path, params, row in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size > 0:
            row['status'] = 'exists'
            continue
        # Download into a temporary directory on the same filesystem; publish
        # only when get_range finishes, so interrupted requests are not cached.
        with tempfile.TemporaryDirectory(dir=path.parent) as temp_dir:
            temp_path = Path(temp_dir) / path.name
            client.timeseries.get_range(**params, path=str(temp_path))
            if not temp_path.is_file() or temp_path.stat().st_size == 0:
                raise RuntimeError(f'Empty download: {path.name}')
            temp_path.replace(path)
        row['status'] = 'downloaded'
        print(f'Saved {path}')
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('date')
    parser.add_argument('--ticker', default='NVDA')
    parser.add_argument('--data-dir', default='data/raw')
    parser.add_argument('--window', choices=('close', 'open'), default='close')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-cost', type=float)
    parser.add_argument('--yes', action='store_true')
    args = parser.parse_args()
    print(download_data(args.date, args.ticker, args.data_dir, confirm=not args.yes,
                        dry_run=args.dry_run, max_cost=args.max_cost,
                        window=args.window).to_string(index=False))
