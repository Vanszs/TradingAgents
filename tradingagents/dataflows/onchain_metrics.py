"""On-chain metrics via Etherscan API.

Supports Ethereum mainnet. Requires ETHERSCAN_API_KEY env var.
Gracefully returns 'data unavailable' if key not set.
All functions return LLM-ready formatted strings.
"""
from __future__ import annotations
import logging
import os
import time
from typing import Optional
import requests

logger = logging.getLogger(__name__)
_CACHE: dict[str, tuple[float, str]] = {}
_CACHE_TTL = 300  # 5 minutes

# Known ERC-20 contract addresses for top tokens
_TOKEN_CONTRACTS: dict[str, str] = {
    'ETH-USD': 'native',
    'USDT-USD': '0xdac17f958d2ee523a2206206994597c13d831ec7',
    'USDC-USD': '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',
    'LINK-USD': '0x514910771af9ca656af840dff83e8264ecf986ca',
    'UNI-USD': '0x1f9840a85d5af5bf1d1762f925bdaddc4201f984',
    'AAVE-USD': '0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9',
    'MKR-USD': '0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2',
    'CRV-USD': '0xd533a949740bb3306d119cc777fa900ba034cd52',
    'SHIB-USD': '0x95ad61b0a150d79219dcf64e1e6cc01f0b64c4ce',
    'PEPE-USD': '0x6982508145454ce325ddbe47a25d4ec3d2311933',
}

ETHERSCAN_BASE = 'https://api.etherscan.io/api'


def _get_api_key() -> Optional[str]:
    key = os.environ.get('ETHERSCAN_API_KEY', '').strip()
    return key if key else None


def _etherscan_get(params: dict) -> Optional[dict]:
    key = _get_api_key()
    if not key:
        return None
    cache_key = str(sorted(params.items()))
    now = time.time()
    if cache_key in _CACHE:
        ts, data = _CACHE[cache_key]
        if now - ts < _CACHE_TTL:
            return data  # type: ignore
    try:
        params['apikey'] = key
        resp = requests.get(ETHERSCAN_BASE, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get('status') == '1':
            _CACHE[cache_key] = (now, data)
            return data
        return None
    except Exception as exc:
        logger.warning('Etherscan request failed: %s', exc)
        return None


def get_onchain_metrics(ticker: str) -> str:
    """Return on-chain metrics for a crypto asset as a formatted string."""
    if not _get_api_key():
        return (
            f'## On-Chain Metrics: {ticker}\n\n'
            'ETHERSCAN_API_KEY not configured. '
            'Set this env var to enable on-chain analysis (active addresses, tx count, gas usage). '
            'Register free at https://etherscan.io/apis'
        )

    contract = _TOKEN_CONTRACTS.get(ticker.upper())
    if not contract:
        # Try base symbol
        base = ticker.split('-')[0].upper()
        for k, v in _TOKEN_CONTRACTS.items():
            if k.startswith(base + '-'):
                contract = v
                break

    if not contract:
        return (
            f'## On-Chain Metrics: {ticker}\n\n'
            f'On-chain metrics not available for {ticker} '
            '(contract address not in supported list or non-EVM chain). '
            'Supported: ETH, USDT, USDC, LINK, UNI, AAVE, MKR, CRV, SHIB, PEPE.'
        )

    if contract == 'native':
        return _get_eth_native_metrics(ticker)
    return _get_erc20_metrics(ticker, contract)


def _get_eth_native_metrics(ticker: str) -> str:
    # ETH supply stats
    supply_data = _etherscan_get({'module': 'stats', 'action': 'ethsupply2'})
    lines = [f'## On-Chain Metrics: {ticker} (Ethereum Native)']
    if supply_data:
        result = supply_data.get('result', {})
        eth_supply = int(result.get('EthSupply', 0)) / 1e18
        burned = int(result.get('BurntFees', 0)) / 1e18
        lines += [
            '',
            f'- Total ETH Supply: {eth_supply:,.0f} ETH',
            f'- Total ETH Burned (EIP-1559): {burned:,.2f} ETH',
            f'- Net Issuance: {eth_supply - burned:,.0f} ETH (approx)',
        ]
    else:
        lines.append('\nSupply data unavailable.')
    return '\n'.join(lines)


def _get_erc20_metrics(ticker: str, contract: str) -> str:
    token_data = _etherscan_get({
        'module': 'stats', 'action': 'tokensupply',
        'contractaddress': contract,
    })
    lines = [f'## On-Chain Metrics: {ticker}']
    if token_data:
        supply = token_data.get('result', 'N/A')
        lines += ['', f'- Token Contract: {contract}', f'- Total Supply (raw): {supply}']
    else:
        lines.append('\nOn-chain data unavailable for this token.')
    return '\n'.join(lines)
